"""Frozen / trainable vision towers ``f_v`` (paper Sec. 4.2, ``V = f_v(I)``).

:class:`BaseVisionTower` holds the policy every tower shares: frozen towers stay in eval mode
(``train()`` override), build no autograd graph (``set_grad_enabled``) and, on CUDA, hold
reduced-precision weights (``frozen_dtype: auto`` -> the AMP dtype, bf16 by default). The weight dtype is
changed **only** by :meth:`BaseVisionTower.apply_dtype_policy` (called from :meth:`set_trainable` and by
the Trainer once the model sits on its device); ``forward`` never casts, so the state-dict layout
(checkpoint fingerprint) is identical before and after the first step.

:class:`VisionTower` is the v1 CLIP tower (design decision 12 of the plan): a plain ``CLIPVisionModel``
returning the raw last hidden state (``post_layernorm`` only when ``apply_post_layernorm``); a trainable
tower (Stage 3) uses non-reentrant gradient checkpointing. The v2 towers (SigLIP 2, C-RADIOv4) live in
:mod:`vision_siglip` / :mod:`vision_radio` and are built by ``vivlmoe.models.towers.build_vision_tower``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import torch
from torch import nn
from transformers import CLIPVisionConfig, CLIPVisionModel

from vivlmoe.config.schema import VisionCfg
from vivlmoe.utils.hf import resolve_model_path

log = logging.getLogger(__name__)


class BaseVisionTower(nn.Module):
    """``forward(pixel_values [B,3,H,W]) -> [B, num_tokens, hidden_size]``.

    Subclasses build ``self.model``, set ``hidden_size`` / ``image_size`` / ``patch_size`` /
    ``num_tokens`` and implement :meth:`_features` (called with inputs already in the weight dtype).
    """

    model: nn.Module
    hidden_size: int
    image_size: int
    patch_size: int
    num_tokens: int

    def __init__(self, cfg: VisionCfg) -> None:
        super().__init__()
        self.cfg = cfg
        # parameters that never train (e.g. CLIP's post_layernorm outside the graph)
        self._always_frozen: set[int] = set()

    def _features(self, pixel_values: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def _freeze_forever(self, module: nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad_(False)
            self._always_frozen.add(id(p))

    # ---------------------------------------------------------------- trainability / dtype
    def _switchable_params(self) -> Iterator[nn.Parameter]:
        return (p for p in self.model.parameters() if id(p) not in self._always_frozen)

    @property
    def trainable(self) -> bool:
        """True when any switchable tower parameter requires grad (robust to external ``requires_grad_``)."""
        return any(p.requires_grad for p in self._switchable_params())

    def set_trainable(self, flag: bool) -> None:
        """Toggle requires_grad on every switchable tower parameter and apply the frozen-dtype policy."""
        for p in self._switchable_params():
            p.requires_grad_(flag)
        self.apply_dtype_policy()
        self.train(self.training)

    def target_dtype(self, amp_dtype: torch.dtype | None = None) -> torch.dtype:
        """Weight dtype prescribed by ``frozen_dtype`` for the current trainability / device.

        ``amp_dtype`` is the Trainer's autocast dtype (None = AMP disabled): ``auto`` on CUDA resolves to it
        (bf16 when unknown) so autocast never re-casts the frozen weights; explicit ``bf16`` + fp16 AMP is
        rejected (plan decision 12).
        """
        if self.trainable:
            return torch.float32
        policy = self.cfg.frozen_dtype
        if policy == "fp32":
            return torch.float32
        if policy == "bf16":
            if amp_dtype is torch.float16:
                raise ValueError(
                    "model.vision.frozen_dtype=bf16 cannot be combined with fp16 AMP (set fp32 or auto)"
                )
            return torch.bfloat16
        device = next(self.model.parameters()).device
        if device.type != "cuda":
            return torch.float32
        return amp_dtype if amp_dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

    def apply_dtype_policy(self, amp_dtype: torch.dtype | None = None) -> torch.dtype:
        """Cast the tower to :meth:`target_dtype` (idempotent) and return the resulting weight dtype.

        This is the only place the weights change dtype: call it after ``model.to(device)`` (the Trainer
        does, before DDP wrapping and before any checkpoint fingerprint is taken).
        """
        target = self.target_dtype(amp_dtype)
        current = next(self.model.parameters()).dtype
        if current != target:
            log.info(
                "%s: casting weights %s -> %s (trainable=%s)",
                type(self).__name__,
                current,
                target,
                self.trainable,
            )
            self.model.to(target)
        return target

    def train(self, mode: bool = True) -> BaseVisionTower:
        super().train(mode)
        if not self.trainable:
            self.model.eval()
        return self

    # ---------------------------------------------------------------- checks
    def assert_dims(self, expected_tokens: int, expected_hidden: int) -> None:
        if (self.num_tokens, self.hidden_size) != (expected_tokens, expected_hidden):
            raise ValueError(
                f"{type(self).__name__} dims mismatch: got tokens={self.num_tokens}, "
                f"hidden={self.hidden_size}; expected tokens={expected_tokens}, hidden={expected_hidden}"
            )

    # ---------------------------------------------------------------- forward
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError(f"pixel_values must be [B,3,H,W], got {tuple(pixel_values.shape)}")
        trainable = self.trainable
        # never cast the weights here (fingerprint stability)
        weight_dtype = next(self.model.parameters()).dtype
        with torch.set_grad_enabled(trainable and torch.is_grad_enabled()):
            hidden = self._features(pixel_values.to(weight_dtype))
        if hidden.shape[1] != self.num_tokens or hidden.shape[2] != self.hidden_size:
            raise RuntimeError(
                f"{type(self).__name__} produced {tuple(hidden.shape)}, expected "
                f"[B,{self.num_tokens},{self.hidden_size}]"
            )
        return hidden if trainable else hidden.float()


class VisionTower(BaseVisionTower):
    """CLIP vision tower: ``[B, n_raw, d_v]`` patch tokens (CLS first)."""

    def __init__(self, cfg: VisionCfg) -> None:
        super().__init__(cfg)
        if cfg.pretrained:
            path = resolve_model_path(cfg.name)
            self.model = CLIPVisionModel.from_pretrained(path)
            self.provenance = f"hf:{cfg.name}"
        else:
            if cfg.tiny_config is None:
                raise ValueError("model.vision.tiny_config is required when pretrained=false")
            self.model = CLIPVisionModel(CLIPVisionConfig(**cfg.tiny_config))
            self.provenance = "random"
        hf = self.model.config
        self.hidden_size = int(hf.hidden_size)
        self.image_size = int(hf.image_size)
        self.patch_size = int(hf.patch_size)
        self.num_tokens = (self.image_size // self.patch_size) ** 2 + 1
        # post_layernorm only feeds the (unused) pooler unless apply_post_layernorm: keep it frozen so a
        # trainable tower (Stage 3) has no parameter outside the graph (DDP find_unused_parameters=False).
        if not cfg.apply_post_layernorm:
            self._freeze_forever(self.model.vision_model.post_layernorm)
        if cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        log.info(
            "VisionTower %s: hidden=%d, image=%d, patch=%d, tokens=%d, grad_ckpt=%s",
            cfg.name if cfg.pretrained else "random-tiny",
            self.hidden_size,
            self.image_size,
            self.patch_size,
            self.num_tokens,
            cfg.gradient_checkpointing,
        )

    def _features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = self.model(pixel_values=pixel_values).last_hidden_state
        if self.cfg.apply_post_layernorm:
            hidden = self.model.vision_model.post_layernorm(hidden)
        return hidden
