from __future__ import annotations

import dataclasses

import pytest
import torch

from vivlmoe.config.schema import ModelCfg
from vivlmoe.models.outputs import TeacherStudentOutput
from vivlmoe.models.stage1 import Stage1Student
from vivlmoe.models.towers.clip_teacher import CLIPTextTeacher


def _batch(b: int = 2) -> dict:
    return {
        "en_input_ids": torch.randint(3, 1000, (b, 77)),
        "en_attention_mask": torch.ones(b, 77, dtype=torch.long),
        "vi_input_ids": torch.randint(3, 1000, (b, 9)),
        "vi_attention_mask": torch.ones(b, 9, dtype=torch.long),
        "uids": list(range(b)),
    }


def test_student_matches_teacher_dim(tiny_cfg: ModelCfg) -> None:
    teacher = CLIPTextTeacher(tiny_cfg)
    student = Stage1Student(tiny_cfg, teacher_dim=teacher.output_dim)
    batch = _batch()
    out = student(batch, "stage1")
    assert isinstance(out, TeacherStudentOutput) and out.teacher is None
    assert out.student.shape == (2, 16) and out.student.dtype == torch.float32
    assert out.student.requires_grad
    z_en = teacher(batch["en_input_ids"], batch["en_attention_mask"])
    assert z_en.shape == out.student.shape
    assert {n.split(".")[0] for n, _ in student.named_parameters()} == {"text_encoder", "fc1"}
    with pytest.raises(ValueError, match="unknown mode"):
        student(batch, "contrastive")


def test_student_two_layer_fc1(tiny_cfg: ModelCfg) -> None:
    cfg = dataclasses.replace(tiny_cfg, stage1=dataclasses.replace(tiny_cfg.stage1, fc1_hidden=24))
    student = Stage1Student(cfg, teacher_dim=16)
    assert isinstance(student.fc1, torch.nn.Sequential)
    assert student(_batch(), "stage1").student.shape == (2, 16)
