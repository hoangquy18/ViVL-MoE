"""ViMACSATask through the Trainer with the tiny smoke config (synthetic reviews, CPU)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tests.unit.tasks.test_captioning_task import run_smoke, smoke_config
from vivlmoe.config.schema import ConfigError
from vivlmoe.models.freezing import FREEZE_PRESETS
from vivlmoe.tasks.evaluators.classification import ClassificationEvaluator
from vivlmoe.tasks.evaluators.dump import read_predictions_jsonl
from vivlmoe.tasks.registry import build_task
from vivlmoe.tasks.vimacsa import ViMACSATask
from vivlmoe.training.distributed import DistContext

ASPECTS = ["Location", "Food", "Room", "Facilities", "Service", "Public_area"]


def test_vimacsa_two_steps_eval_dump_and_checkpoint(tmp_path: Path) -> None:
    cfg, task, result = run_smoke("vimacsa_tiny", tmp_path, "train.max_steps=2")
    assert isinstance(task, ViMACSATask)
    run_dir = Path(cfg.run.dir) / cfg.run.name
    assert result["step"] == 2
    keys = set(result["last_eval"])
    assert {"dev/macro_f1", "dev/macro_p", "dev/macro_r"} <= keys
    assert {f"dev/f1/{a}" for a in ASPECTS} <= keys
    assert all(math.isfinite(v) for v in result["last_eval"].values())
    rows = read_predictions_jsonl(run_dir / "predictions" / "dev_step2.jsonl")
    assert [r["uid"] for r in rows] == list(range(8))
    assert set(rows[0]["pred"]) == set(ASPECTS) and set(rows[0]["gold"]) == set(ASPECTS)
    assert all(v in {"None", "Negative", "Neutral", "Positive"} for v in rows[0]["gold"].values())
    metrics = json.loads((run_dir / "metrics" / "dev_step2.json").read_text(encoding="utf-8"))
    assert metrics["aspects"] == ASPECTS and metrics["exclude_none_in_f1"] is False
    assert (run_dir / "checkpoints" / "best.pt.meta.json").exists()


def test_vimacsa_task_wiring(tmp_path: Path) -> None:
    cfg = smoke_config("vimacsa_tiny", tmp_path)
    task = build_task(cfg, DistContext.single_process("cpu"))
    assert task.always_frozen() == FREEZE_PRESETS["downstream_cls"]
    assert task.selection_metric() == ("dev/macro_f1", "max")
    evs = task.evaluators()
    assert isinstance(evs["dev"], ClassificationEvaluator) and evs["dev"].aspects == ASPECTS
    batch = next(iter(task.build_train_loader()))
    assert batch["pixel_values"].shape[1:] == (3, 3, 32, 32)  # max_images 3
    assert batch["input_ids"].shape[:2] == (2, 6) and batch["labels"].shape == (2, 6)
    assert batch["input_ids"].shape[-1] <= 32  # cls_max_len
    cfg.task.metrics = ["bleu4"]
    with pytest.raises(ConfigError, match="task.metrics must be"):
        build_task(cfg, DistContext.single_process("cpu")).build_evaluators()
