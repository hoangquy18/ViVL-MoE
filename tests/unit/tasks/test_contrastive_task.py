"""ContrastiveTask through the real Trainer on CPU with configs/_smoke/stage2_tiny.yaml."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.conftest import CONFIGS, REPO_ROOT
from vivlmoe.config import load_config
from vivlmoe.config.schema import ConfigError
from vivlmoe.losses.router_losses import RouterLossComputer
from vivlmoe.models.moe.moe_ffn import MoEFFN
from vivlmoe.tasks import build_task, import_builtin_tasks
from vivlmoe.tasks.contrastive import ContrastiveTask, encoder_of
from vivlmoe.tasks.evaluators.retrieval import RetrievalEvaluator, RetrievalLoaders
from vivlmoe.training.distributed import DistContext
from vivlmoe.training.trainer import Trainer
from vivlmoe.utils.ckpt_meta import read_meta

CTX = DistContext.single_process("cpu")
SMOKE = CONFIGS / "_smoke" / "stage2_tiny.yaml"

import_builtin_tasks()


def _overrides(tmp_path: Path, *extra: str) -> list[str]:
    syn = tmp_path / "syn"
    return [
        f"run.dir={tmp_path / 'runs'}",
        f"data.train.root={syn}",
        f"data.evals.ktvic_dev.root={syn}",
        f"data.evals.uitviic_dev.root={syn}",
        "init_from=null",  # no Stage-1 checkpoint in a unit test
        *extra,
    ]


def test_contrastive_two_steps_and_eval(tmp_path: Path) -> None:
    cfg = load_config(SMOKE, _overrides(tmp_path, "train.max_steps=2"))
    task = build_task(cfg, CTX)
    assert isinstance(task, ContrastiveTask)
    run_dir = Path(cfg.run.dir) / cfg.run.name
    trainer = Trainer(cfg, task, CTX, run_dir)
    assert trainer.effective_batch == 8 and trainer.steps_per_epoch == 4  # 32 images / 4 / accum 2
    assert set(trainer.evaluators) == {"ktvic_dev", "uitviic_dev"}
    assert all(isinstance(e, RetrievalEvaluator) for e in trainer.evaluators.values())
    assert all(isinstance(ld, RetrievalLoaders) for ld in trainer.loaders.evals.values())
    assert trainer.selection_metric == "ktvic_dev/rsum" and trainer.selection_mode == "max"
    assert trainer.frozen_paths == ["vision_encoder", "vision_proj"]
    assert not encoder_of(trainer.raw_model).is_moe

    result = trainer.fit()
    assert result["step"] == 2
    last = result["last_eval"]
    for split, (n_img, n_cap) in {"ktvic_dev": (4, 8), "uitviic_dev": (2, 6)}.items():
        ds = trainer.loaders.evals[split].dataset
        assert (len(ds.image_ids), len(ds.captions)) == (n_img, n_cap)
        for k in ("i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10", "rsum"):
            assert 0.0 <= last[f"{split}/{k}"] <= 600.0
        assert last[f"{split}/i2t_r10"] == 100.0  # <= 10 candidates: R@10 is trivially perfect

    dumped = json.loads((run_dir / "metrics" / "ktvic_dev_step2.json").read_text("utf-8"))
    assert dumped["num_images"] == 4 and dumped["num_captions"] == 8
    assert dumped["metrics"]["rsum"] == last["ktvic_dev/rsum"]

    ck = run_dir / "checkpoints"
    assert (ck / "step_0000002.pt").exists() and (ck / "best.pt").exists()
    meta = read_meta(ck / "best.pt")
    assert meta["tower_provenance"] == {"vision_encoder": "trained", "text_encoder": "trained"}
    assert meta["excluded_modules"] == []  # random towers cannot be re-materialised
    assert meta["dev_split_sha1"] == {"ktvic_dev": None, "uitviic_dev": None}
    assert meta["transform_spec"]["size"] == 32 and meta["transform_spec"]["augment"] == "none"


def test_contrastive_step_metrics(tmp_path: Path) -> None:
    cfg = load_config(SMOKE, _overrides(tmp_path, "train.max_steps=1"))
    task = build_task(cfg, CTX)
    run_dir = Path(cfg.run.dir) / cfg.run.name
    trainer = Trainer(cfg, task, CTX, run_dir)
    trainer.sampler.set_epoch(0)
    res = trainer.train_optimizer_step(iter(trainer.train_loader))
    assert torch.isfinite(res.loss) and res.samples == 8 and res.tokens > 0
    assert set(res.metrics) == {"loss_contrastive", "acc_i2t", "acc_t2i", "logit_scale"}
    assert float(res.metrics["loss_contrastive"]) == pytest.approx(float(res.loss), rel=1e-5)
    assert float(res.metrics["logit_scale"]) == pytest.approx(1 / 0.07, rel=1e-2)
    assert task.selection_metric() == ("ktvic_dev/rsum", "max")


def test_contrastive_rejects_parallel_text_eval_split(tmp_path: Path) -> None:
    cfg = load_config(
        SMOKE,
        _overrides(
            tmp_path,
            "data.evals.phomt_dev={kind: synthetic_parallel, split: dev, options: {num_samples: 4}}",
        ),
    )
    task = build_task(cfg, CTX)
    with pytest.raises(ConfigError, match="image-text"):
        task.build_loaders()


def test_contrastive_moe_variant(tmp_path: Path) -> None:
    cfg = load_config(
        SMOKE,
        _overrides(
            tmp_path,
            "train.max_steps=2",
            "model.moe={num_shared: 1, num_routed: 4, top_k: 2, aux_free_bias: true, "
            "bias_update_rate: 0.01, bias_update_every: step}",
        ),
    )
    task = build_task(cfg, CTX)
    run_dir = Path(cfg.run.dir) / cfg.run.name
    trainer = Trainer(cfg, task, CTX, run_dir)
    encoder = encoder_of(trainer.raw_model)
    assert encoder.is_moe and len(encoder.moe_layers()) == cfg.model.encoder.num_layers
    assert all(isinstance(ffn, MoEFFN) for _, ffn in encoder.moe_layers())
    assert int(encoder.dispatch_stats().sum()) == 0  # nothing routed before the first step
    trainer.sampler.set_epoch(0)
    seen: list[dict[str, Any]] = []

    def spy() -> Iterator[dict[str, Any]]:
        for batch in trainer.train_loader:
            seen.append(batch)
            yield batch

    res = trainer.train_optimizer_step(spy())
    assert torch.isfinite(res.loss)
    assert {"loss_aux", "loss_z", "loss_mi"} <= set(res.metrics)
    assert isinstance(task._router_loss, RouterLossComputer)
    # Eq. 22 uses the routed-token counts of EVERY micro-batch of the step: no hook resets the
    # counters after the first backward (the Trainer has no startup forward). Each micro-batch
    # routes B x num_latents visual tokens plus the valid text tokens through every MoE layer.
    assert len(seen) == cfg.train.grad_accum_steps == 2
    num_latents = trainer.raw_model.num_visual_tokens
    tokens = sum(len(b["uids"]) * num_latents + int(b["attention_mask"].sum()) for b in seen)
    stats = encoder.dispatch_stats()  # [L, N_r + 1]
    assert stats[:, -1].tolist() == [tokens] * cfg.model.encoder.num_layers
    assert int(stats[:, :-1].sum()) == tokens * cfg.model.moe.top_k * cfg.model.encoder.num_layers
    before = [b.clone() for b in encoder.router_bias_state()]
    trainer.state.step += 1
    task.post_optimizer_step(trainer, trainer.state)
    after = encoder.router_bias_state()
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))
    assert int(encoder.dispatch_stats().sum()) == 0  # counters reset after the update
    result = trainer.fit()
    assert "ktvic_dev/rsum" in result["last_eval"]


def _load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        f"vivlmoe_script_{name}", REPO_ROOT / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_build_eval_loaders_is_public_and_needs_no_train_source(tmp_path: Path) -> None:
    """scripts/evaluate.py and analyze_routing.py use build_eval_loaders() without data.train."""
    cfg = load_config(SMOKE, _overrides(tmp_path, "data.train=null"))
    task = build_task(cfg, CTX)
    evals = task.build_eval_loaders()
    assert set(evals) == {"ktvic_dev", "uitviic_dev"}
    assert all(isinstance(ld, RetrievalLoaders) for ld in evals.values())
    assert set(task.evaluators()) == {"ktvic_dev", "uitviic_dev"}
    assert task.checkpoint_metadata()["dev_split_sha1"] == {"ktvic_dev": None, "uitviic_dev": None}
    assert task.checkpoint_metadata()["transform_spec"]["size"] == 32
    with pytest.raises(ConfigError, match="data.train"):
        task.build_train_loader()


def test_evaluate_script_recomputes_retrieval_from_checkpoint(tmp_path: Path) -> None:
    cfg = load_config(SMOKE, _overrides(tmp_path, "train.max_steps=2"))
    task = build_task(cfg, CTX)
    run_dir = Path(cfg.run.dir) / cfg.run.name
    result = Trainer(cfg, task, CTX, run_dir).fit()
    best = run_dir / "checkpoints" / "best.pt"
    assert best.exists()

    evaluate = _load_script("evaluate")
    evaluate._populate_registries()
    eval_cfg = load_config(SMOKE, _overrides(tmp_path, "data.train=null"))  # no corpus indexing
    out_dir = tmp_path / "eval_out"
    results = evaluate.evaluate(eval_cfg, CTX, str(best), ["ktvic_dev"], out_dir)
    assert set(results) == {"ktvic_dev"}
    assert results["ktvic_dev"]["rsum"] == pytest.approx(result["last_eval"]["ktvic_dev/rsum"])
    assert (out_dir / "metrics" / "ktvic_dev_step2.json").exists()


def test_real_stage2_and_stage3_configs_load() -> None:
    s2 = load_config(CONFIGS / "pretrain" / "stage2_contrastive.yaml")
    assert s2.task.name == "contrastive" and s2.optim.freeze_preset == "stage2"
    assert s2.init_from is not None and s2.init_from.load == ["text_encoder"]
    # the decontaminated corpus built by scripts/build_pretrain_corpus.py (configs/data/pretrain_corpus.yaml)
    train = s2.data.train
    assert train is not None and train.kind == "jsonl" and train.root == "dataset"
    assert train.splits["train"].file == "corpus/pretrain_corpus.jsonl"
    assert train.image_check == "decode" and train.missing_image_policy == "skip"
    assert set(s2.data.evals) == {"ktvic_dev", "uitviic_dev"}
    assert s2.data.evals["uitviic_dev"].root.endswith("uitvic_dataset")
    assert s2.data.batch_size * s2.train.grad_accum_steps * 4 == 2048
    assert s2.eval.selection_metric == "ktvic_dev/rsum"

    s3 = load_config(CONFIGS / "pretrain" / "stage3_end2end.yaml")
    assert s3.optim.freeze_preset == "stage3" and s3.init_from.load == ["all"]
    assert s3.model.vision.gradient_checkpointing and s3.model.text.gradient_checkpointing
    assert [g.lr for g in s3.optim.groups] == [1.0e-5, 5.0e-5]
    assert s3.optim.groups[0].modules == ["vision_encoder", "text_encoder", "encoder"]
    assert "logit_scale" in s3.optim.groups[1].modules
    assert s3.data.batch_size * s3.train.grad_accum_steps * 4 == 2048
    assert s3.run.name == "stage3_end2end" and s3.data.train.kind == "jsonl"


def test_sigmoid_head_step_metrics(tmp_path: Path) -> None:
    extra = ("train.max_steps=1", "model.contrastive_head=sigmoid", "model.logit_scale_init=0.1")
    cfg = load_config(SMOKE, _overrides(tmp_path, *extra))
    task = build_task(cfg, CTX)
    trainer = Trainer(cfg, task, CTX, Path(cfg.run.dir) / cfg.run.name)
    trainer.sampler.set_epoch(0)
    res = trainer.train_optimizer_step(iter(trainer.train_loader))
    assert torch.isfinite(res.loss)
    assert set(res.metrics) == {
        "loss_contrastive",
        "acc_i2t",
        "acc_t2i",
        "logit_scale",
        "logit_bias",
    }
    assert float(res.metrics["logit_scale"]) == pytest.approx(10.0, rel=1e-2)
    assert float(res.metrics["logit_bias"]) == pytest.approx(-10.0)
    assert (
        trainer.raw_model.logit_bias.grad is not None or trainer.raw_model.logit_bias.requires_grad
    )


def test_infonce_backbone_has_no_logit_bias(tmp_path: Path) -> None:
    from vivlmoe.models.factory import build_task_model

    v1 = build_task_model("backbone", load_config(SMOKE, _overrides(tmp_path)).model)
    assert "logit_bias" not in v1.state_dict()  # v1 checkpoints keep their layout
    v2 = build_task_model(
        "backbone", load_config(SMOKE, _overrides(tmp_path, "model.contrastive_head=sigmoid")).model
    )
    assert float(v2.state_dict()["logit_bias"]) == -10.0
