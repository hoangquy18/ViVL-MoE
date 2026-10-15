"""Frozen SigLIP / SigLIP 2 teachers for the v2 text distillation (``model.stage1.teacher_kind: siglip``).

The SigLIP 2 text tower is multilingual, so it can teach the mmBERT student on Vietnamese captions
directly (caption -> same caption) as well as on PhoMT (English -> Vietnamese). The target is the
joint-space embedding ``pooler_output`` (last token + head). It is computed **without** an attention
mask and from texts padded to ``max_length`` 64, as SigLIP was trained (``data.tokenizer.en_max_len``
must be 64). The text and image towers are loaded separately from the full repo ``teacher_name``
(the full ``SiglipModel`` is never built twice); the image tower serves only dev retrieval in the
teacher's joint space. Like the CLIP teachers, both are owned by the task, frozen and always in eval.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import (
    AutoConfig,
    SiglipTextConfig,
    SiglipTextModel,
    SiglipVisionConfig,
    SiglipVisionModel,
)

from vivlmoe.config.schema import ModelCfg, VisionCfg
from vivlmoe.models.towers.clip_teacher import _FrozenTeacher
from vivlmoe.utils.hf import load_fp32, resolve_model_path

log = logging.getLogger(__name__)


def _tiny_part(cfg: ModelCfg, key: str) -> dict[str, Any]:
    tiny = cfg.stage1.teacher_tiny_config or {}
    if key not in tiny:
        raise ValueError(
            f"model.stage1.teacher_tiny_config needs a {key!r} mapping (random SigLIP)"
        )
    return dict(tiny[key])


def _teacher_name(cfg: ModelCfg) -> str:
    if cfg.stage1.teacher_name is None:
        raise ValueError("model.stage1.teacher_name is required for a pretrained SigLIP teacher")
    return cfg.stage1.teacher_name


def _is_pretrained(cfg: ModelCfg) -> bool:
    return cfg.stage1.teacher_tiny_config is None


def siglip_vision_cfg(cfg: ModelCfg) -> VisionCfg:
    """The teacher's image tower as a ``VisionCfg`` (its transform, its ``image_size``)."""
    if _is_pretrained(cfg):
        return VisionCfg(kind="siglip", name=_teacher_name(cfg))
    return VisionCfg(kind="siglip", pretrained=False, tiny_config=_tiny_part(cfg, "vision_config"))


def siglip_teacher_dim(cfg: ModelCfg) -> int:
    """Width of the joint space (``text_config.projection_size``) without loading weights."""
    if _is_pretrained(cfg):
        name = _teacher_name(cfg)
        text = getattr(AutoConfig.from_pretrained(resolve_model_path(name)), "text_config", None)
        if text is None:
            raise ValueError(
                f"model.stage1.teacher_name={name!r} has no text_config: the teacher needs a full "
                "SigLIP repo (text + vision towers)"
            )
    else:
        text = SiglipTextConfig(**_tiny_part(cfg, "text_config"))
    return int(text.projection_size)


class SiglipTextTeacher(_FrozenTeacher):
    """SigLIP text tower producing the distillation target ``z_t = pooler_output``."""

    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        if cfg.stage1.teacher_target != "projected":
            raise ValueError("a SigLIP teacher only provides teacher_target=projected")
        self.target = "projected"
        if _is_pretrained(cfg):
            name = _teacher_name(cfg)
            self.model = load_fp32(SiglipTextModel, resolve_model_path(name))
            self.provenance = f"hf:{name}"
        else:
            self.model = SiglipTextModel(SiglipTextConfig(**_tiny_part(cfg, "text_config")))
            self.provenance = "random"
        self.output_dim: int = int(self.model.config.projection_size)
        self._freeze()
        log.info("SiglipTextTeacher dim=%d (%s)", self.output_dim, self.provenance)

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``-> [B, P] fp32`` (no grad). ``attention_mask`` is ignored: SigLIP attends to padding."""
        del attention_mask
        return self.model(input_ids=input_ids).pooler_output.float()


class SiglipImageTeacher(_FrozenTeacher):
    """SigLIP image tower (MAP-head ``pooler_output``) for dev retrieval in the joint space."""

    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        vision = siglip_vision_cfg(cfg)
        if vision.pretrained:
            self.model = load_fp32(SiglipVisionModel, resolve_model_path(vision.name))
            self.provenance = f"hf:{vision.name}"
        else:
            self.model = SiglipVisionModel(SiglipVisionConfig(**(vision.tiny_config or {})))
            self.provenance = "random"
        self.output_dim: int = int(self.model.config.hidden_size)
        text_dim = siglip_teacher_dim(cfg)
        if self.output_dim != text_dim:
            raise ValueError(
                f"SigLIP teacher image width {self.output_dim} != text projection {text_dim}"
            )
        self._freeze()

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``-> [B, P] fp32`` (no grad)."""
        return self.model(pixel_values=pixel_values).pooler_output.float()


__all__ = [
    "SiglipImageTeacher",
    "SiglipTextTeacher",
    "siglip_teacher_dim",
    "siglip_vision_cfg",
]
