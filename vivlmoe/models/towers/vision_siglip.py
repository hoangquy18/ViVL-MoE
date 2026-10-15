"""SigLIP / SigLIP 2 vision tower (v2): ``[B, (S/p)^2, d_v]`` patch tokens.

Fixed-resolution SigLIP 2 checkpoints (e.g. ``google/siglip2-giant-opt-patch16-384``) load as
``SiglipVisionModel``. There is no CLS token, ``last_hidden_state`` already includes the post-LayerNorm,
and the MAP pooling head is not built (``vision_use_head=False``): only the patch tokens feed the
resampler. ``vision.image_size`` different from the checkpoint's interpolates the position embeddings.
"""

from __future__ import annotations

import logging

import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

from vivlmoe.config.schema import VisionCfg
from vivlmoe.models.towers.vision import BaseVisionTower
from vivlmoe.utils.hf import resolve_model_path

log = logging.getLogger(__name__)


class SiglipVisionTower(BaseVisionTower):
    def __init__(self, cfg: VisionCfg) -> None:
        super().__init__(cfg)
        if cfg.apply_post_layernorm:
            raise ValueError("SigLIP's last_hidden_state already includes the post-LayerNorm")
        if cfg.pretrained:
            path = resolve_model_path(cfg.name)
            # an optional config attribute (read with hasattr), so it is set on the config, not passed
            hf_cfg = SiglipVisionConfig.from_pretrained(path)
            hf_cfg.vision_use_head = False
            self.model = SiglipVisionModel.from_pretrained(path, config=hf_cfg)
            self.provenance = f"hf:{cfg.name}"
        else:
            if cfg.tiny_config is None:
                raise ValueError("model.vision.tiny_config is required when pretrained=false")
            tiny = {**cfg.tiny_config, "vision_use_head": False}
            self.model = SiglipVisionModel(SiglipVisionConfig(**tiny))
            self.provenance = "random"
        hf = self.model.config
        self.hidden_size = int(hf.hidden_size)
        self.patch_size = int(hf.patch_size)
        self.image_size = int(cfg.image_size or hf.image_size)
        self._interpolate = self.image_size != int(hf.image_size)
        self.num_tokens = (self.image_size // self.patch_size) ** 2
        if cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        log.info(
            "SiglipVisionTower %s: hidden=%d, image=%d, patch=%d, tokens=%d, grad_ckpt=%s",
            cfg.name if cfg.pretrained else "random-tiny",
            self.hidden_size,
            self.image_size,
            self.patch_size,
            self.num_tokens,
            cfg.gradient_checkpointing,
        )

    def _features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=pixel_values, interpolate_pos_encoding=self._interpolate)
        return out.last_hidden_state
