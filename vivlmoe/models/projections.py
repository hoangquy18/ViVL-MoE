"""Random-init glue modules of the backbone (paper Sec. 4.2): ``W_v``, ``W_t`` and modality embeddings."""

from __future__ import annotations

import torch
from torch import nn


class VisionProjection(nn.Linear):
    """``W_v: d_v -> d`` applied to every raw patch token before the resampler."""

    def __init__(self, d_v: int, dim: int) -> None:
        super().__init__(d_v, dim)


class TextProjection(nn.Linear):
    """``W_t: d_t -> d`` applied to every XLM-R token."""

    def __init__(self, d_t: int, dim: int) -> None:
        super().__init__(d_t, dim)


class ModalityEmbedding(nn.Embedding):
    """``e_v`` (row 0) and ``e_t`` (row 1) added to the projected tokens (Sec. 4.2, joint mode)."""

    def __init__(self, dim: int) -> None:
        super().__init__(2, dim)


@torch.no_grad()
def init_module_(module: nn.Module, std: float = 0.02, skip: tuple[str, ...] = ()) -> None:
    """Normal(0, std) for Linear / Embedding weights, zero biases, unit LayerNorm (recursively).

    Sub-trees whose attribute name (any dotted-path component) is in ``skip`` are left untouched, e.g.
    ``skip=("router",)`` keeps a MoE router's own ``router_init_std`` initialisation.
    """
    for name, m in module.named_modules():
        if skip and any(part in skip for part in name.split(".")):
            continue
        if isinstance(m, nn.Linear):
            m.weight.normal_(0.0, std)
            if m.bias is not None:
                m.bias.zero_()
        elif isinstance(m, nn.Embedding):
            m.weight.normal_(0.0, std)
        elif isinstance(m, nn.LayerNorm):
            m.weight.fill_(1.0)
            m.bias.zero_()
