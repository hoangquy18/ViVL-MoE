"""Stage-1 teachers by ``model.stage1.teacher_kind``: CLIP (v1, paper Sec. 4.1) or SigLIP 2 (v2).

v1 reads the teacher from ``model.vision`` (the CLIP tower is also the student's vision encoder);
v2 keeps them apart (C-RADIO vision, SigLIP 2 teacher), so everything that describes the teacher
(target width, image transform, tokenizer) goes through these helpers instead of ``model.vision``.
"""

from __future__ import annotations

from torch import nn

from vivlmoe.config.schema import ModelCfg, VisionCfg
from vivlmoe.models.towers import clip_teacher, siglip_teacher


def teacher_vision_cfg(cfg: ModelCfg) -> VisionCfg:
    """The image tower of the teacher's joint space."""
    if cfg.stage1.teacher_kind == "siglip":
        return siglip_teacher.siglip_vision_cfg(cfg)
    return cfg.vision


def teacher_output_dim(cfg: ModelCfg) -> int:
    """Width of the Stage-1 target without loading weights (drives ``Stage1Student.fc1``)."""
    if cfg.stage1.teacher_kind == "siglip":
        return siglip_teacher.siglip_teacher_dim(cfg)
    return clip_teacher.teacher_output_dim(cfg)


def build_text_teacher(cfg: ModelCfg) -> nn.Module:
    if cfg.stage1.teacher_kind == "siglip":
        return siglip_teacher.SiglipTextTeacher(cfg)
    return clip_teacher.CLIPTextTeacher(cfg)


def build_image_teacher(cfg: ModelCfg) -> nn.Module:
    if cfg.stage1.teacher_kind == "siglip":
        return siglip_teacher.SiglipImageTeacher(cfg)
    return clip_teacher.CLIPImageTeacher(cfg)


__all__ = ["build_image_teacher", "build_text_teacher", "teacher_output_dim", "teacher_vision_cfg"]
