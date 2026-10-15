from __future__ import annotations

import pytest
import torch
from torch import nn

from vivlmoe.config.schema import ModelCfg
from vivlmoe.models.backbone import ViVLBackbone
from vivlmoe.models.factory import build_task_model
from vivlmoe.models.freezing import (
    FREEZE_PRESETS,
    apply_freeze,
    apply_frozen_dtype_policy,
    assert_no_unused_trainable_params,
    freeze_paths_for_preset,
)
from vivlmoe.models.towers import find_tower_paths

from .conftest import make_contrastive_batch


def test_presets_validate_against_models(tiny_cfg: ModelCfg) -> None:
    backbone = build_task_model("backbone", tiny_cfg)
    student = build_task_model("stage1", tiny_cfg)
    for preset in ("stage1",):
        assert freeze_paths_for_preset(student, preset) == []
    for preset in ("stage2", "stage3", "sparse"):
        assert freeze_paths_for_preset(backbone, preset) == FREEZE_PRESETS[preset]
    seq2seq = build_task_model("seq2seq", tiny_cfg)
    assert "backbone.logit_scale" in freeze_paths_for_preset(seq2seq, "downstream_seq2seq")
    with pytest.raises(ValueError, match="neither a submodule nor a parameter"):
        freeze_paths_for_preset(
            backbone, "downstream_seq2seq"
        )  # no 'backbone.' prefix on a backbone
    with pytest.raises(ValueError, match="unknown freeze preset"):
        freeze_paths_for_preset(backbone, "nope")


def test_apply_freeze_stage2_uses_tower_policy(tiny_cfg: ModelCfg) -> None:
    backbone = build_task_model("backbone", tiny_cfg)
    frozen = apply_freeze(backbone, freeze_paths_for_preset(backbone, "stage2"))
    assert all(n.startswith(("vision_encoder.", "vision_proj.")) for n in frozen)
    backbone.train()
    assert not backbone.vision_encoder.model.training and backbone.text_encoder.model.training
    assert not backbone.vision_proj.weight.requires_grad
    assert backbone.text_proj.weight.requires_grad and backbone.logit_scale.requires_grad
    assert_no_unused_trainable_params(backbone, make_contrastive_batch(), "contrastive")
    assert all(p.grad is None for p in backbone.parameters())  # cleaned up afterwards


def test_stage3_all_trainable_no_unused(tiny_cfg: ModelCfg) -> None:
    backbone = build_task_model("backbone", tiny_cfg)
    apply_freeze(backbone, freeze_paths_for_preset(backbone, "stage3"))
    assert_no_unused_trainable_params(backbone, make_contrastive_batch(), "contrastive")


def test_unused_parameter_is_detected(tiny_cfg: ModelCfg) -> None:
    backbone = build_task_model("backbone", tiny_cfg)
    backbone.unused_extra = nn.Linear(4, 4)  # intentionally outside the graph
    with pytest.raises(RuntimeError, match="unused_extra.weight"):
        assert_no_unused_trainable_params(backbone, make_contrastive_batch(), "contrastive")
    backbone.unused_extra.requires_grad_(False)
    assert_no_unused_trainable_params(backbone, make_contrastive_batch(), "contrastive")


def test_stage1_student_no_unused(tiny_cfg: ModelCfg) -> None:
    student = build_task_model("stage1", tiny_cfg)
    batch = {
        "vi_input_ids": torch.randint(3, 1000, (2, 5)),
        "vi_attention_mask": torch.ones(2, 5, dtype=torch.long),
    }
    assert_no_unused_trainable_params(student, batch, "stage1")


def test_apply_frozen_dtype_policy_walks_nested_towers(tiny_cfg: ModelCfg) -> None:
    import dataclasses

    cfg = dataclasses.replace(
        tiny_cfg, vision=dataclasses.replace(tiny_cfg.vision, frozen_dtype="bf16")
    )
    model = build_task_model("seq2seq", cfg)
    assert find_tower_paths(model) == {
        "vision_encoder": "backbone.vision_encoder",
        "text_encoder": "backbone.text_encoder",
    }
    for p in model.backbone.vision_encoder.parameters():
        p.requires_grad_(False)
    assert apply_frozen_dtype_policy(model) == {"backbone.vision_encoder": torch.bfloat16}
    assert next(model.backbone.vision_encoder.model.parameters()).dtype == torch.bfloat16
    backbone = ViVLBackbone.from_config(tiny_cfg)
    assert find_tower_paths(backbone) == {
        "vision_encoder": "vision_encoder",
        "text_encoder": "text_encoder",
    }
    assert apply_frozen_dtype_policy(backbone) == {"vision_encoder": torch.float32}  # trainable
