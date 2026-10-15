"""TeacherStudentTask through the real Trainer on CPU with configs/_smoke/stage1_tiny.yaml."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from tests.conftest import CONFIGS
from vivlmoe.config import load_config
from vivlmoe.config.schema import ConfigError
from vivlmoe.tasks import build_task, import_builtin_tasks
from vivlmoe.tasks.evaluators.retrieval import ClipSpaceRetrievalEvaluator, RetrievalLoaders
from vivlmoe.tasks.evaluators.teacher_mse import TeacherMSEEvaluator, per_sample_teacher_distance
from vivlmoe.tasks.stage1_teacher import TeacherStudentTask
from vivlmoe.training.distributed import DistContext
from vivlmoe.training.state import StepContext
from vivlmoe.training.trainer import Trainer
from vivlmoe.utils.ckpt_meta import read_meta

CTX = DistContext.single_process("cpu")
SMOKE = CONFIGS / "_smoke" / "stage1_tiny.yaml"

import_builtin_tasks()


def _overrides(tmp_path: Path, *extra: str) -> list[str]:
    return [
        f"run.dir={tmp_path / 'runs'}",
        f"data.evals.ktvic_dev.root={tmp_path / 'syn'}",
        *extra,
    ]


def test_stage1_two_steps_and_eval(tmp_path: Path) -> None:
    cfg = load_config(SMOKE, _overrides(tmp_path, "train.max_steps=2"))
    task = build_task(cfg, CTX)
    assert isinstance(task, TeacherStudentTask)
    run_dir = Path(cfg.run.dir) / cfg.run.name
    trainer = Trainer(cfg, task, CTX, run_dir)
    assert trainer.effective_batch == 8 and trainer.steps_per_epoch == 4
    assert isinstance(trainer.evaluators["phomt_dev"], TeacherMSEEvaluator)
    assert isinstance(trainer.evaluators["ktvic_dev"], ClipSpaceRetrievalEvaluator)
    assert isinstance(trainer.loaders.evals["ktvic_dev"], RetrievalLoaders)
    assert trainer.selection_metric == "phomt_dev/mse" and trainer.selection_mode == "min"
    # the teacher is owned by the task: not a submodule, not in the state dict
    assert "teacher" not in {k.split(".")[0] for k in trainer.raw_model.state_dict()}
    assert task._teacher is None  # built lazily on the first micro-batch

    result = trainer.fit()
    assert result["step"] == 2
    last = result["last_eval"]
    for key in ("phomt_dev/mse", "ktvic_dev/rsum", "ktvic_dev/i2t_r1", "ktvic_dev/t2i_r10"):
        assert key in last and torch.isfinite(torch.tensor(last[key]))
    assert last["phomt_dev/num_samples"] == 8
    assert task._teacher is not None and not any(
        p.requires_grad for p in task._teacher.parameters()
    )
    assert list(trainer.evaluators["ktvic_dev"]._image_cache) == ["ktvic_dev"]
    assert trainer.evaluators["ktvic_dev"]._image_cache["ktvic_dev"].shape == (4, 16)

    ck = run_dir / "checkpoints"
    assert (ck / "step_0000002.pt").exists() and (ck / "last.pt").exists()
    assert (ck / "best.pt").exists()
    meta = read_meta(ck / "best.pt")
    assert meta["dev_split_sha1"] == {"phomt_dev": None, "ktvic_dev": None}
    assert meta["transform_spec"]["crop_size"] == 32
    assert meta["metrics"]["phomt_dev/mse"] == last["phomt_dev/mse"]
    assert meta["tower_provenance"] == {"text_encoder": "trained"}
    assert set(meta["config"]["task"]) >= {"name", "eval_clip_retrieval"}
    assert (run_dir / "eval_history.jsonl").exists()
    for split in ("phomt_dev", "ktvic_dev"):  # rank-0 metrics dumps per evaluation
        dumped = json.loads((run_dir / "metrics" / f"{split}_step2.json").read_text("utf-8"))
        assert dumped["split"] == split and dumped["step"] == 2
        assert dumped["metrics"] == {
            k.split("/", 1)[1]: v for k, v in last.items() if k.startswith(split)
        }


def test_stage1_metrics_and_loss_are_finite(tmp_path: Path) -> None:
    cfg = load_config(SMOKE, _overrides(tmp_path, "train.max_steps=1"))
    task = build_task(cfg, CTX)
    run_dir = Path(cfg.run.dir) / cfg.run.name
    trainer = Trainer(cfg, task, CTX, run_dir)
    assert task.selection_metric() == ("phomt_dev/mse", "min")
    it = iter(trainer.train_loader)
    trainer.sampler.set_epoch(0)
    res = trainer.train_optimizer_step(it)
    assert torch.isfinite(res.loss) and res.samples == 8
    assert set(res.metrics) == {"student_norm", "teacher_norm"}
    assert res.tokens > 0


def test_stage1_teacher_target_is_fp32_under_autocast(tmp_path: Path) -> None:
    """The Eq. 12 target z_en must not carry bf16 rounding when the Trainer autocasts."""
    cfg = load_config(SMOKE, _overrides(tmp_path, "train.max_steps=1"))
    task = build_task(cfg, CTX)
    run_dir = Path(cfg.run.dir) / cfg.run.name
    trainer = Trainer(cfg, task, CTX, run_dir)
    seen: list[tuple[torch.dtype, bool]] = []

    def hook(_m: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
        seen.append((out.dtype, torch.is_autocast_cpu_enabled()))

    task.teacher(torch.device("cpu")).register_forward_hook(hook)
    trainer.sampler.set_epoch(0)
    batch = next(iter(trainer.train_loader))
    sc = StepContext(step=0, micro=0, accum=1, device=torch.device("cpu"))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = task.train_step(trainer.model, batch, sc)
    assert seen == [(torch.float32, False)]
    assert torch.isfinite(out.loss)


def test_stage1_build_eval_loaders_is_public_and_needs_no_train_source(tmp_path: Path) -> None:
    cfg = load_config(SMOKE, _overrides(tmp_path, "data.train=null"))
    task = build_task(cfg, CTX)
    evals = task.build_eval_loaders()
    assert set(evals) == {"phomt_dev", "ktvic_dev"}
    assert isinstance(evals["ktvic_dev"], RetrievalLoaders)
    assert task._split_kinds == {"phomt_dev": "parallel", "ktvic_dev": "retrieval"}
    assert isinstance(task.evaluators()["phomt_dev"], TeacherMSEEvaluator)
    assert task.selection_metric() == ("phomt_dev/mse", "min")
    with pytest.raises(ConfigError, match="data.train"):
        task.build_train_loader()


def test_stage1_image_split_requires_eval_clip_retrieval(tmp_path: Path) -> None:
    cfg = load_config(SMOKE, _overrides(tmp_path, "task.eval_clip_retrieval=false"))
    task = build_task(cfg, CTX)
    with pytest.raises(ConfigError, match="eval_clip_retrieval"):
        task.build_eval_loaders()


def test_per_sample_distance_matches_loss() -> None:
    from vivlmoe.losses.teacher import TeacherMSELoss

    g = torch.Generator().manual_seed(0)
    s, t = torch.randn(5, 7, generator=g), torch.randn(5, 7, generator=g)
    for preset in ("eq12", "legacy"):
        per = per_sample_teacher_distance(s, t, preset)
        assert per.shape == (5,)
        assert torch.allclose(per.mean(), TeacherMSELoss(preset)(s, t))


def test_real_stage1_config_loads() -> None:
    cfg = load_config(CONFIGS / "pretrain" / "stage1_teacher.yaml")
    assert cfg.task.name == "stage1_teacher" and cfg.task.eval_clip_retrieval
    assert cfg.data.train is not None and cfg.data.train.kind == "phomt"
    assert cfg.data.train.options["max_train_samples"] == 2_000_000
    assert cfg.data.evals["ktvic_dev"].holdout is not None
    assert cfg.data.evals["ktvic_dev"].holdout.ids_file.endswith("ktvic_dev_image_ids.json")
    assert cfg.data.batch_size * cfg.train.grad_accum_steps * 4 == 2048
    assert cfg.eval.selection_metric == "phomt_dev/mse" and cfg.eval.selection_mode == "min"
    assert cfg.optim.freeze_preset == "stage1" and cfg.train.early_stopping.patience == 2
