"""Captioning / VQA with the v2 LLM head through the Trainer (configs/v2/_smoke, CPU)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from tests.unit.tasks.test_captioning_task import run_smoke, smoke_config
from vivlmoe.config.schema import ConfigError
from vivlmoe.models.heads.causal_lm import CausalLMVLModel
from vivlmoe.tasks.evaluators.dump import read_predictions_jsonl
from vivlmoe.tasks.registry import build_task
from vivlmoe.training.distributed import DistContext

V2 = "../v2/_smoke/"  # smoke_config() resolves names under configs/_smoke
CTX = DistContext.single_process("cpu")


@pytest.mark.parametrize(
    ("name", "metric", "num_samples"), [("caption_tiny", "bleu4", 4), ("vqa_tiny", "rougeL", 8)]
)
def test_llm_head_two_steps_eval_and_checkpoint(
    tmp_path: Path, name: str, metric: str, num_samples: int
) -> None:
    cfg, _task, result = run_smoke(V2 + name, tmp_path, "train.max_steps=2")
    run_dir = Path(cfg.run.dir) / cfg.run.name
    assert result["step"] == 2 and f"dev/{metric}" in result["last_eval"]
    assert all(math.isfinite(v) for v in result["last_eval"].values())
    rows = read_predictions_jsonl(run_dir / "predictions" / "dev_step2.jsonl")
    assert [r["uid"] for r in rows] == list(range(num_samples))
    state = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu")["model"]
    assert {k.split(".")[0] for k in state} >= {"backbone", "llm", "projector"}


def test_llm_head_task_wiring(tmp_path: Path) -> None:
    task = build_task(smoke_config(V2 + "vqa_tiny", tmp_path), CTX)
    assert task.causal_lm and isinstance(task.build_model(), CausalLMVLModel)
    gen_kwargs = task.evaluators()["dev"].gen_kwargs
    assert (gen_kwargs["eos_token_id"], gen_kwargs["pad_token_id"]) == (1, 0)
    tb = next(iter(task.build_train_loader()))
    assert {"input_ids", "attention_mask", "llm_input_ids", "llm_attention_mask", "labels"} <= set(
        tb
    )
    assert all(row[row != -100][-1] == 1 for row in tb["labels"])  # answers end with the LLM EOS
    v1 = build_task(smoke_config("vqa_tiny", tmp_path), CTX)  # ViT5 head: unchanged
    assert not v1.causal_lm and "pad_token_id" not in v1.evaluators()["dev"].gen_kwargs


def test_llm_head_config_rules(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="causal_lm uses projector"):
        smoke_config(V2 + "caption_tiny", tmp_path, "model.generation.bridge_norm=true")
    with pytest.raises(ConfigError, match="captioning has no"):
        smoke_config(V2 + "caption_tiny", tmp_path, "task.llm_prompt_template=hỏi {question}")


def test_llm_head_on_a_moe_backbone_with_the_routing_probe(tmp_path: Path) -> None:
    moe = [
        "model.moe.num_shared=1",
        "model.moe.num_routed=2",
        "model.moe.top_k=1",
        "analysis.routing_probe.split=dev",
        "analysis.routing_probe.every_steps=1",
    ]
    cfg, _task, result = run_smoke(V2 + "vqa_tiny", tmp_path, "train.max_steps=2", *moe)
    assert cfg.model.is_moe and result["step"] == 2
    reports = sorted((Path(cfg.run.dir) / cfg.run.name / "routing").glob("*.json"))
    assert reports, "the routing probe wrote no report"
