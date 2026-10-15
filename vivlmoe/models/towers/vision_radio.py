"""NVIDIA C-RADIO vision tower (v2 default ``nvidia/C-RADIOv4-SO400M``), frozen-only.

C-RADIOv4 is distilled from SigLIP2-g, DINOv3-7B and SAM3: one backbone with language-aligned and dense
spatial features, any resolution in multiples of 16 (preferred 512). It ships as Hugging Face remote code
(``trust_remote_code``; pin ``vision.revision``, which also enters the provenance) and takes pixels in
[0, 1], normalising them internally (the ``tower`` transform produces that range).

``model(x)`` returns ``(summary, spatial)``; the spatial features ``[B, (S/16)^2, 1152]`` feed the
resampler. The summary and the teacher adaptors are not used for training (``scripts/probe_vision.py``
uses the ``siglip2-g`` adaptor). The tower is frozen when built and refuses to train: its remote code
offers no gradient checkpointing, and its unused summary / adaptor parameters would break DDP's
unused-parameter check.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from vivlmoe.config.schema import VisionCfg
from vivlmoe.models.towers.radio_stub import RadioStub
from vivlmoe.models.towers.vision import BaseVisionTower
from vivlmoe.utils.hf import resolve_model_path

log = logging.getLogger(__name__)

RADIO_DEFAULT_RESOLUTION = 512  # C-RADIOv4's preferred resolution


def _spatial(out: Any) -> torch.Tensor:
    """Spatial features of a RADIO output: ``(summary, spatial)`` or, with adaptors, a dict."""
    if isinstance(out, dict):
        out = out["backbone"]
    return out[1]


def _preferred(model: nn.Module) -> int | None:
    res = getattr(model, "preferred_resolution", None)
    if res is None:
        return None
    return int(res[0]) if isinstance(res, tuple | list) else int(res)


def _feature_dim(model: nn.Module, patch_size: int) -> int:
    """Feature width from one tiny forward (the remote config does not record it)."""
    res = model.get_nearest_supported_resolution(4 * patch_size, 4 * patch_size)
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        return int(_spatial(model(torch.zeros(1, 3, *res, dtype=dtype))).shape[-1])


class RadioVisionTower(BaseVisionTower):
    def __init__(self, cfg: VisionCfg) -> None:
        super().__init__(cfg)
        if cfg.pretrained:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(
                resolve_model_path(cfg.name), trust_remote_code=True, revision=cfg.revision
            )
            self.provenance = f"hf:{cfg.name}" + (f"@{cfg.revision}" if cfg.revision else "")
        else:
            if cfg.tiny_config is None:
                raise ValueError("model.vision.tiny_config is required when pretrained=false")
            self.model = RadioStub(**cfg.tiny_config)
            self.provenance = "random"
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self.patch_size = int(self.model.patch_size)
        wanted = int(cfg.image_size or _preferred(self.model) or RADIO_DEFAULT_RESOLUTION)
        self.image_size = int(self.model.get_nearest_supported_resolution(wanted, wanted)[0])
        if self.image_size != wanted:
            log.warning("C-RADIO: resolution %d -> nearest supported %d", wanted, self.image_size)
        self.num_tokens = (self.image_size // self.patch_size) ** 2
        self.hidden_size = _feature_dim(self.model, self.patch_size)
        log.info(
            "RadioVisionTower %s: hidden=%d, image=%d, patch=%d, tokens=%d (frozen)",
            cfg.name if cfg.pretrained else "random-tiny",
            self.hidden_size,
            self.image_size,
            self.patch_size,
            self.num_tokens,
        )

    def set_trainable(self, flag: bool) -> None:
        if flag:
            raise ValueError("C-RADIO vision towers are frozen-only: freeze model.vision_encoder")
        super().set_trainable(False)

    def _features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return _spatial(self.model(pixel_values))
