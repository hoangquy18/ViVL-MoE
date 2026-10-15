"""VQATask through the Trainer with the tiny smoke config (synthetic questions, CPU)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tests.unit.tasks.test_captioning_task import assert_vietnamese, run_smoke, smoke_config
from vivlmoe.models.freezing import FREEZE_PRESETS
from vivlmoe.tasks.evaluators.dump import read_predictions_jsonl
from vivlmoe.tasks.registry import build_task
from vivlmoe.tasks.vqa import VQATask
from vivlmoe.training.distributed import DistContext
from vivlmoe.utils.ckpt_meta import read_meta


def test_vqa_two_steps_eval_dump_and_checkpoint(tmp_path: Path) -> None:
    cfg, task, result = run_smoke("vqa_tiny", tmp_path, "train.max_steps=2")
    assert isinstance(task, VQATask)
    run_dir = Path(cfg.run.dir) / cfg.run.name
    assert result["step"] == 2
    assert {"dev/rougeL", "dev/f1", "dev/em"} <= set(result["last_eval"])
    assert all(math.isfinite(v) for v in result["last_eval"].values())
    rows = read_predictions_jsonl(run_dir / "predictions" / "dev_step2.jsonl")
    assert [r["uid"] for r in rows] == list(range(8))
    assert rows[0]["prompt"].endswith("?") and len(rows[0]["references"]) == 2
    assert_vietnamese(rows[0]["prompt"])
    metrics = json.loads((run_dir / "metrics" / "dev_step2.json").read_text(encoding="utf-8"))
    assert metrics["mode"] == "joint_vqa" and metrics["metric_names"] == ["rougeL", "f1", "em"]
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert read_meta(run_dir / "checkpoints" / "best.pt")["transform_spec"]["crop_size"] == 32


def test_vqa_task_wiring_and_prompt_template(tmp_path: Path) -> None:
    cfg = smoke_config("vqa_tiny", tmp_path)
    task = build_task(cfg, DistContext.single_process("cpu"))
    assert task.always_frozen() == FREEZE_PRESETS["downstream_seq2seq"]
    assert task.selection_metric() == ("dev/rougeL", "max")
    train = task.build_train_loader()
    assert len(train.dataset) == 32  # train_on_all_answers: 16 questions x 2 answers
    tb = next(iter(train))
    assert {"pixel_values", "input_ids", "attention_mask", "labels", "uids"} <= set(tb)
    assert tb["input_ids"].shape[0] == 2
    assert (tb["labels"] == -100).any()  # variable-length synthetic answers -> -100 padding
    # the adapter template wins unless task.prompt_template is set explicitly
    assert task.resolve_template("{question}") == "{question}"
    assert task.resolve_template("Câu hỏi: {question}") == "Câu hỏi: {question}"
    cfg.task.prompt_template = "Hỏi: {question} Trả lời:"
    task2 = build_task(cfg, DistContext.single_process("cpu"))
    assert task2.resolve_template("Câu hỏi: {question}") == "Hỏi: {question} Trả lời:"
    batch = next(iter(task2.build_eval_loaders()["dev"]))
    assert batch["prompts"][0].startswith("Hỏi: ") and batch["prompts"][0].endswith(" Trả lời:")
    cfg.task.metrics = ["bleu4"]  # not a VQA metric
    with pytest.raises(ValueError, match="unknown metrics"):
        build_task(cfg, DistContext.single_process("cpu")).build_evaluators()
