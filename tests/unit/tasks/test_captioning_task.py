"""CaptioningTask through the Trainer with the tiny smoke config (synthetic captions, CPU)."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import pytest

import vivlmoe.tasks.captioning  # noqa: F401  (registers 'captioning')
import vivlmoe.tasks.vimacsa  # noqa: F401
import vivlmoe.tasks.vqa  # noqa: F401
from vivlmoe.config import load_config
from vivlmoe.config.schema import ConfigError, RunConfig
from vivlmoe.models.freezing import FREEZE_PRESETS
from vivlmoe.tasks.captioning import CaptioningTask
from vivlmoe.tasks.evaluators.composite import CompositeEvaluator
from vivlmoe.tasks.evaluators.dump import read_predictions_jsonl
from vivlmoe.tasks.evaluators.generation import GenerationEvaluator
from vivlmoe.tasks.registry import build_task
from vivlmoe.training.distributed import DistContext
from vivlmoe.training.trainer import Trainer
from vivlmoe.utils.ckpt_meta import read_meta

REPO = Path(__file__).resolve().parents[3]
SMOKE = REPO / "configs" / "_smoke"


def smoke_config(name: str, tmp_path: Path, *extra: str) -> RunConfig:
    """Load ``configs/_smoke/<name>.yaml`` with the run dir and synthetic roots under ``tmp_path``."""
    syn = str(tmp_path / "syn")
    return load_config(
        SMOKE / f"{name}.yaml",
        [
            f"run.dir={tmp_path / 'runs'}",
            f"data.train.root={syn}",
            f"data.evals.dev.root={syn}",
            *extra,
        ],
    )


def run_smoke(name: str, tmp_path: Path, *extra: str) -> tuple[RunConfig, Any, dict[str, Any]]:
    """Train the smoke config in-process (single CPU process); returns ``(cfg, task, fit result)``."""
    cfg = smoke_config(name, tmp_path, *extra)
    ctx = DistContext.single_process("cpu")
    task = build_task(cfg, ctx)
    trainer = Trainer(cfg, task, ctx, Path(cfg.run.dir) / cfg.run.name)
    result = trainer.fit()
    return cfg, task, result


def assert_vietnamese(text: str) -> None:
    assert any(ord(ch) > 127 for ch in text), f"no non-ASCII characters preserved in {text!r}"


def test_captioning_two_steps_eval_dump_and_checkpoint(tmp_path: Path) -> None:
    cfg, task, result = run_smoke("caption_tiny", tmp_path, "train.max_steps=2")
    assert isinstance(task, CaptioningTask)
    run_dir = Path(cfg.run.dir) / cfg.run.name
    assert result["step"] == 2
    assert {"dev/bleu4", "dev/rougeL", "dev/rougeL_paper_compat"} <= set(result["last_eval"])
    assert all(math.isfinite(v) for v in result["last_eval"].values())
    # predictions JSONL in uid order with Vietnamese references preserved
    rows = read_predictions_jsonl(run_dir / "predictions" / "dev_step2.jsonl")
    assert [r["uid"] for r in rows] == list(range(len(rows))) and len(rows) == 4
    assert all(len(r["references"]) == 2 for r in rows)
    assert_vietnamese(" ".join(rows[0]["references"]))
    raw = (run_dir / "predictions" / "dev_step2.jsonl").read_text(encoding="utf-8")
    assert "\\u" not in raw  # ensure_ascii=False
    metrics = json.loads((run_dir / "metrics" / "dev_step2.json").read_text(encoding="utf-8"))
    assert metrics["generation"]["num_beams"] == 2 and metrics["tokenizer_signatures"][
        "bleu_tokenizer"
    ].startswith("13a")
    assert metrics["metrics"]["num_samples"] == 4
    # checkpoint + sidecar with dev split / transform spec metadata
    ckpt_dir = run_dir / "checkpoints"
    assert (ckpt_dir / "best.pt").exists() and (ckpt_dir / "best.pt.meta.json").exists()
    meta = read_meta(ckpt_dir / "best.pt")
    assert meta["transform_spec"]["crop_size"] == 32 and meta["dev_split_sha1"] is None
    assert meta["config"]["task"]["name"] == "captioning"
    # final_metrics.json is written by scripts/train.py; in-process we only see the dict
    assert "final/dev/bleu4" in result["final"] and (run_dir / "eval_history.jsonl").exists()


def test_captioning_task_wiring(tmp_path: Path) -> None:
    cfg = smoke_config("caption_tiny", tmp_path)
    task = build_task(cfg, DistContext.single_process("cpu"))
    assert task.always_frozen() == FREEZE_PRESETS["downstream_caption"]
    assert "backbone.text_encoder" in task.always_frozen()
    assert task.selection_metric() == ("dev/bleu4", "max")
    evs = task.evaluators()
    assert set(evs) == {"dev"} and isinstance(evs["dev"], GenerationEvaluator)
    assert evs["dev"].mode == "joint_caption" and evs["dev"].gen_kwargs["num_beams"] == 2
    loaders = task.build_eval_loaders()
    batch = next(iter(loaders["dev"]))
    assert "labels" not in batch and "refs" in batch and "input_ids" not in batch
    train = task.build_train_loader()
    tb = next(iter(train))
    assert tb["labels"].shape[0] == 2
    assert (tb["labels"] == -100).any()  # variable-length synthetic captions -> -100 padding
    assert task.checkpoint_metadata()["transform_spec"]["size"] == 32
    cfg.task.metrics = ["bogus"]
    with pytest.raises(ValueError, match="unknown metrics"):
        build_task(cfg, DistContext.single_process("cpu")).build_evaluators()


def test_routing_probe_attached_for_moe_regardless_of_eval_cadence(tmp_path: Path) -> None:
    """The probe wraps the dev evaluator for a MoE backbone; the Trainer runs it on its own step
    cadence, so an epoch-based eval config is accepted. A dense model ignores the probe."""
    moe = [
        "model.moe.num_shared=1",
        "model.moe.num_routed=2",
        "model.moe.top_k=1",
        "analysis.routing_probe.split=dev",
        "analysis.routing_probe.every_steps=3",
    ]
    ctx = DistContext.single_process("cpu")
    cfg = smoke_config(
        "caption_tiny", tmp_path, *moe, "eval.every_steps=null", "eval.every_epochs=1"
    )
    assert cfg.model.is_moe
    evs = build_task(cfg, ctx).evaluators()
    assert isinstance(evs["dev"], CompositeEvaluator)
    assert [e.name for e in evs["dev"].evaluators][-1] == "routing_probe"
    with pytest.raises(ConfigError, match="not in data.evals"):
        build_task(
            smoke_config("caption_tiny", tmp_path, *moe, "analysis.routing_probe.split=test"), ctx
        ).evaluators()
    cfg = smoke_config("caption_tiny", tmp_path, "analysis.routing_probe.split=dev")
    assert not cfg.model.is_moe
    assert isinstance(build_task(cfg, ctx).evaluators()["dev"], GenerationEvaluator)


@pytest.mark.hf_cache
def test_captioning_step_on_real_ktvic_with_small_towers(tmp_path: Path) -> None:
    """One optimizer step on 8 real KTVIC captions with the small CLIP/XLM-R towers + ViT5-base."""
    root = Path("/Users/gos/Documents/Moe/moe/ktvic_dataset")
    if not root.exists():
        pytest.skip("KTVIC dataset not available locally")
    logging.getLogger().setLevel(logging.INFO)
    cfg_text = f"""
_include: [configs/_base/train_defaults.yaml, configs/model/small.yaml]
run: {{name: ktvic_small, dir: "{tmp_path / "runs"}", seed: 0}}
task: {{name: captioning, max_target_len: 24, metrics: [bleu4, rougeL], generation: {{num_beams: 2, max_new_tokens: 6}}}}
model: {{generation: {{name: VietAI/vit5-base}}}}
data:
  train: {{kind: ktvic, root: "{root}", split: train, image_check: none, limit: 8}}
  evals: {{dev: {{kind: ktvic, root: "{root}", split: test, image_check: none, limit: 2}}}}
  batch_size: 8
  eval_batch_size: 2
  num_workers: 0
  pin_memory: false
  persistent_workers: false
  tokenizer: {{decoder_max_len: 24}}
optim: {{lr: 3.0e-5}}
train: {{max_steps: 1, grad_accum_steps: 1, amp: {{enabled: false}}, log_every: 1}}
schedule: {{kind: constant, warmup_ratio: 0.0}}
eval: {{every_steps: 1, every_epochs: null, selection_metric: dev/bleu4, final_splits: []}}
checkpoint: {{every_steps: null, every_epochs: null, save_best: false}}
logging: {{tensorboard: false}}
"""
    path = tmp_path / "ktvic_small.yaml"
    path.write_text(cfg_text, encoding="utf-8")
    cfg = load_config(path)
    ctx = DistContext.single_process("cpu")
    task = build_task(cfg, ctx)
    trainer = Trainer(cfg, task, ctx, Path(cfg.run.dir) / cfg.run.name)
    raw = trainer.raw_model
    assert raw.backbone.vision_encoder.num_tokens == 50 and raw.decoder.config.d_model == 768
    result = trainer.fit()
    assert result["step"] == 1 and math.isfinite(result["last_eval"]["dev/bleu4"])
    rows = read_predictions_jsonl(
        Path(cfg.run.dir) / cfg.run.name / "predictions" / "dev_step1.jsonl"
    )
    assert len(rows) == 1 and len(rows[0]["references"]) == 5  # KTVIC: 5 captions per image
    assert_vietnamese(rows[0]["references"][0])
