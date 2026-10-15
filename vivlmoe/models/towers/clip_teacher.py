"""Frozen CLIP teachers for Stage 1 (paper Sec. 4.1, Eq. 12; Appendix A retrieval in CLIP space).

The teachers are owned by the Stage-1 task, never by the student module (plan decision 11), so they
take part in no DDP wrapping or checkpointing. Both are permanently in eval mode and run under
``torch.no_grad``.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
from torch import nn
from transformers import (
    AutoConfig,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from vivlmoe.config.schema import ModelCfg
from vivlmoe.utils.hf import load_fp32, resolve_model_path

log = logging.getLogger(__name__)

TeacherTarget = Literal["projected", "eos_hidden"]


def _tiny_text_config(cfg: ModelCfg) -> CLIPTextConfig:
    if cfg.stage1.teacher_tiny_config is None:
        raise ValueError(
            "model.stage1.teacher_tiny_config is required for a random teacher (vision.pretrained=false)"
        )
    return CLIPTextConfig(**cfg.stage1.teacher_tiny_config)


def teacher_output_dim(cfg: ModelCfg) -> int:
    """Width of the Stage-1 target ``z_en`` without loading weights (drives ``Stage1Student.fc1``)."""
    target = cfg.stage1.teacher_target
    if cfg.vision.pretrained:
        hf = AutoConfig.from_pretrained(resolve_model_path(cfg.vision.name))
        if getattr(hf, "text_config", None) is None:
            raise ValueError(
                f"model.vision.name={cfg.vision.name!r} is a vision-only repo ({type(hf).__name__}); the "
                "Stage-1 teacher needs a full CLIP repo with a text tower (CLIPConfig.text_config)"
            )
        text_hidden = int(hf.text_config.hidden_size)
        proj = int(hf.projection_dim)
    else:
        tc = _tiny_text_config(cfg)
        text_hidden, proj = int(tc.hidden_size), int(tc.projection_dim)
    return proj if target == "projected" else text_hidden


class _FrozenTeacher(nn.Module):
    """Shared plumbing: parameters frozen, permanently eval."""

    model: nn.Module

    def _freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> _FrozenTeacher:
        super().train(False)
        return self


class CLIPTextTeacher(_FrozenTeacher):
    """English CLIP text encoder producing ``z_en`` (Eq. 12 target).

    ``target='projected'`` -> ``text_embeds`` (CLIP joint space, Appendix A);
    ``target='eos_hidden'`` -> the EOS-pooled hidden state before the projection.
    """

    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        self.target: TeacherTarget = cfg.stage1.teacher_target
        if cfg.vision.pretrained:
            self.model = load_fp32(CLIPTextModelWithProjection, resolve_model_path(cfg.vision.name))
            self.provenance = f"hf:{cfg.vision.name}"
        else:
            self.model = CLIPTextModelWithProjection(_tiny_text_config(cfg))
            self.provenance = "random"
        self.output_dim: int = teacher_output_dim(cfg)
        self._freeze()
        log.info(
            "CLIPTextTeacher target=%s dim=%d (%s)", self.target, self.output_dim, self.provenance
        )

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """``-> [B, P] fp32`` (no grad)."""
        if self.target == "projected":
            z = self.model(input_ids=input_ids, attention_mask=attention_mask).text_embeds
        else:
            z = self.model.text_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).pooler_output
        if z.shape[-1] != self.output_dim:
            raise RuntimeError(f"teacher produced dim {z.shape[-1]}, expected {self.output_dim}")
        return z.float()


class CLIPImageTeacher(_FrozenTeacher):
    """CLIP image encoder (projected) used only for Stage-1 dev retrieval in CLIP space."""

    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        if cfg.vision.pretrained:
            self.model = load_fp32(
                CLIPVisionModelWithProjection, resolve_model_path(cfg.vision.name)
            )
            self.provenance = f"hf:{cfg.vision.name}"
        else:
            if cfg.vision.tiny_config is None:
                raise ValueError("model.vision.tiny_config is required when pretrained=false")
            self.model = CLIPVisionModelWithProjection(CLIPVisionConfig(**cfg.vision.tiny_config))
            self.provenance = "random"
        self.output_dim: int = int(self.model.config.projection_dim)
        self._freeze()

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``-> [B, P] fp32`` (no grad)."""
        return self.model(pixel_values=pixel_values).image_embeds.float()
