"""Symmetric InfoNCE (paper Eq. 13) with global-batch negatives across DDP ranks.

Features are gathered with the grad-flowing ``torch.distributed.nn.functional.all_gather`` and the
labels are the local rows ``arange(B) + rank * B`` (open_clip style). Per-rank losses are means over the
local rows, so DDP's gradient averaging reproduces the single-process gradient of the global batch
exactly (verified by ``tests/unit/losses/test_contrastive.py::test_gloo_gather_matches_single_process``).
The loss object is a plain callable (no ``nn.Module``): collectives never live inside modules.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F

SymmetricReduction = Literal["mean", "sum"]


def _gather_features(x: torch.Tensor) -> torch.Tensor:
    from torch.distributed.nn.functional import all_gather  # grad-flowing variant

    return torch.cat(all_gather(x), dim=0)


class ContrastiveLoss:
    """``(loss, {"acc_i2t", "acc_t2i"}) = loss(image_embeds, text_embeds, logit_scale)``.

    ``symmetric_reduction='mean'`` = ``0.5 * (CE_i2t + CE_t2i)`` (calibration of the paper's lambdas,
    plan decision 9); ``'sum'`` = ``CE_i2t + CE_t2i``.
    """

    def __init__(
        self,
        gather: bool = True,
        symmetric_reduction: SymmetricReduction = "mean",
        debug_check_shapes: bool = False,
    ) -> None:
        if symmetric_reduction not in ("mean", "sum"):
            raise ValueError(f"symmetric_reduction must be mean|sum, got {symmetric_reduction!r}")
        self.gather = gather
        self.symmetric_reduction = symmetric_reduction
        self.debug_check_shapes = debug_check_shapes

    @staticmethod
    def _distributed() -> bool:
        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    def _check_shapes(self, image_embeds: torch.Tensor) -> None:
        shapes: list[object] = [None] * dist.get_world_size()
        dist.all_gather_object(shapes, tuple(image_embeds.shape))
        if any(s != shapes[0] for s in shapes):
            raise RuntimeError(f"contrastive loss: per-rank feature shapes differ: {shapes}")

    def __call__(
        self, image_embeds: torch.Tensor, text_embeds: torch.Tensor, logit_scale: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if image_embeds.shape != text_embeds.shape or image_embeds.ndim != 2:
            raise ValueError(
                f"image/text embeds must be [B,d] with equal shapes, got "
                f"{tuple(image_embeds.shape)} / {tuple(text_embeds.shape)}"
            )
        img, txt = image_embeds.float(), text_embeds.float()
        scale = logit_scale.float()
        b = img.shape[0]
        if self.gather and self._distributed():
            if self.debug_check_shapes:
                self._check_shapes(img)
            all_img, all_txt = _gather_features(img), _gather_features(txt)
            offset = dist.get_rank() * b
        else:
            all_img, all_txt, offset = img, txt, 0
        labels = torch.arange(b, device=img.device) + offset
        logits_i2t = scale * img @ all_txt.t()  # [B, W*B]
        logits_t2i = scale * txt @ all_img.t()
        ce_i2t = F.cross_entropy(logits_i2t, labels)
        ce_t2i = F.cross_entropy(logits_t2i, labels)
        loss = ce_i2t + ce_t2i
        if self.symmetric_reduction == "mean":
            loss = 0.5 * loss
        with torch.no_grad():
            stats = {
                "acc_i2t": (logits_i2t.argmax(dim=1) == labels).float().mean().item(),
                "acc_t2i": (logits_t2i.argmax(dim=1) == labels).float().mean().item(),
            }
        return loss, stats
