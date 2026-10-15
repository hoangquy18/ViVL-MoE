"""v2 distillation teachers: SigLIP text / image towers, the teacher_kind dispatch, mean pooling."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import SiglipConfig, SiglipModel, SiglipVisionConfig, SiglipVisionModel

from vivlmoe.config.schema import ModelCfg
from vivlmoe.models.factory import (
    build_image_teacher,
    build_task_model,
    build_teacher,
    load_model_cfg,
)
from vivlmoe.models.stage1 import Stage1Student
from vivlmoe.models.towers import (
    CLIPTextTeacher,
    SiglipImageTeacher,
    SiglipTextTeacher,
    teacher_output_dim,
    teacher_vision_cfg,
)

V2_TINY = Path(__file__).resolve().parents[3] / "configs" / "model" / "v2_tiny.yaml"
TEXT = dict(vocab_size=1000, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, max_position_embeddings=64, projection_size=24)  # fmt: skip
VISION = dict(hidden_size=24, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
              image_size=32, patch_size=16)  # fmt: skip


def _with_stage1(cfg: ModelCfg, **changes: Any) -> ModelCfg:
    return dataclasses.replace(cfg, stage1=dataclasses.replace(cfg.stage1, **changes))


def _pretrained(path: Path) -> ModelCfg:
    return _with_stage1(load_model_cfg(V2_TINY), teacher_name=str(path), teacher_tiny_config=None)


def test_tiny_text_teacher_ignores_the_mask_and_stays_frozen() -> None:
    cfg = load_model_cfg(V2_TINY)
    teacher = build_teacher(cfg)
    assert isinstance(teacher, SiglipTextTeacher)
    assert teacher.output_dim == teacher_output_dim(cfg) == 24  # projection_size, not hidden 32
    ids = torch.randint(3, 1000, (2, 64))
    z = teacher(ids, torch.ones(2, 64, dtype=torch.long))
    assert z.shape == (2, 24) and z.dtype == torch.float32 and not z.requires_grad
    assert torch.equal(z, teacher(ids, torch.zeros(2, 64, dtype=torch.long)))  # attends to pads
    teacher.train()
    assert not teacher.training and not teacher.model.training
    assert not any(p.requires_grad for p in teacher.parameters())


def test_pretrained_teachers_load_each_tower_of_the_full_repo_in_fp32(tmp_path: Path) -> None:
    torch.manual_seed(0)
    full = SiglipModel(SiglipConfig(text_config=TEXT, vision_config=VISION)).eval()
    full.to(torch.bfloat16).save_pretrained(tmp_path)
    full.float()  # the bf16 weights, upcast: what the teachers must hold
    cfg = _pretrained(tmp_path)
    assert teacher_output_dim(cfg) == 24
    text, image = build_teacher(cfg), build_image_teacher(cfg)
    assert isinstance(image, SiglipImageTeacher) and text.provenance == f"hf:{tmp_path}"
    for module in (text, image):
        assert {p.dtype for p in module.parameters()} == {torch.float32}
    ids, px = torch.randint(3, 1000, (2, 64)), torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(text(ids), full.text_model(input_ids=ids).pooler_output, atol=1e-5)
        expected = full.vision_model(pixel_values=px).pooler_output  # MAP head kept
        assert torch.allclose(image(px), expected, atol=1e-5)
    vision = teacher_vision_cfg(cfg)
    assert (vision.kind, vision.name, vision.pretrained) == ("siglip", str(tmp_path), True)


def test_teacher_config_errors(tmp_path: Path) -> None:
    cfg = load_model_cfg(V2_TINY)
    with pytest.raises(ValueError, match="projected"):
        build_teacher(_with_stage1(cfg, teacher_target="eos_hidden"))
    with pytest.raises(ValueError, match="vision_config"):
        build_image_teacher(_with_stage1(cfg, teacher_tiny_config={"text_config": TEXT}))
    wide = {"text_config": TEXT, "vision_config": {**VISION, "hidden_size": 32}}
    with pytest.raises(ValueError, match="image width 32"):
        build_image_teacher(_with_stage1(cfg, teacher_tiny_config=wide))
    with pytest.raises(ValueError, match="teacher_name"):
        teacher_output_dim(_with_stage1(cfg, teacher_tiny_config=None))
    SiglipVisionModel(SiglipVisionConfig(**VISION)).save_pretrained(tmp_path)  # vision-only repo
    with pytest.raises(ValueError, match="full SigLIP repo"):
        teacher_output_dim(_pretrained(tmp_path))


def test_clip_teacher_dispatch_is_unchanged(tiny_cfg: ModelCfg) -> None:
    assert isinstance(build_teacher(tiny_cfg), CLIPTextTeacher)
    assert teacher_vision_cfg(tiny_cfg) is tiny_cfg.vision
    assert teacher_output_dim(tiny_cfg) == 16


def test_mean_pooled_student_ignores_padding() -> None:
    student = build_task_model("stage1", load_model_cfg(V2_TINY)).eval()
    assert isinstance(student, Stage1Student) and student.pooling == "mean"

    def embed(ids: torch.Tensor) -> torch.Tensor:
        batch = {"vi_input_ids": ids, "vi_attention_mask": (ids != 1).long()}
        return student(batch, "stage1").student

    with torch.no_grad():
        short = embed(torch.tensor([[0, 5, 6, 7, 2]]))
        padded = embed(torch.tensor([[0, 5, 6, 7, 2, 1, 1]]))
    assert short.shape == (1, 24) and torch.allclose(short, padded, atol=1e-5)
