"""Auxiliary-loss-free load balancing step (paper Sec. 5.3, Eq. 22; plan decision 7).

Called by the Task on every rank after ``optimizer.step()`` (``moe.bias_update_every: step``) or at
the end of every micro-batch (``micro``). The per-layer dispatch counts are packed into one
``[L, N_r + 1]`` tensor and summed across ranks with a single collective, so the sign rule is applied
to identical fractions everywhere and ``expert_bias`` never diverges between ranks
(``broadcast_buffers=False``).
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from vivlmoe.training.distributed import DistContext, all_reduce_sum_

log = logging.getLogger(__name__)


def expert_fractions(stats: torch.Tensor) -> torch.Tensor:
    """``[L, N_r + 1]`` (counts | tokens) -> ``f [L, N_r] = counts / max(tokens, 1)``."""
    counts = stats[:, :-1].to(torch.float32)
    tokens = stats[:, -1:].to(torch.float32).clamp(min=1.0)
    return counts / tokens


@torch.no_grad()
def aux_free_bias_step(
    encoder: nn.Module, ctx: DistContext, eta: float, per_layer_logs: bool = False
) -> dict[str, float]:
    """All-reduce the routed-token counts, apply Eq. 22 on every layer, reset the counters.

    ``encoder`` is a :class:`vivlmoe.models.encoder.MultimodalEncoder` with ``is_moe``. When no
    token was routed since the last reset the biases are left untouched (logged). Returns
    ``{'moe/expert_frac_max', 'moe/expert_frac_min', 'moe/bias_range', 'moe/tokens'}`` and,
    optionally, ``'moe/layer{l}/expert_frac_max'`` / ``'moe/layer{l}/expert_frac_min'``.
    """
    if not getattr(encoder, "is_moe", False):
        raise RuntimeError("aux_free_bias_step called on a dense encoder")
    stats = encoder.dispatch_stats()
    if stats.numel() == 0:
        raise RuntimeError("aux_free_bias_step: encoder reports no MoE layers")
    stats = stats.to(ctx.device)
    all_reduce_sum_(stats, ctx)  # the single collective of the update
    f = expert_fractions(stats)  # [L, N_r]
    if int(stats[:, -1].sum()) == 0:
        # f == 0 would shift every bias by +eta (sign(k/N_r - 0) = +1): not a load signal, skip it
        log.warning(
            "aux_free_bias_step: no tokens were routed since the last reset, bias unchanged"
        )
    else:
        encoder.apply_router_bias_update(f, eta)
    encoder.reset_dispatch_stats()
    biases = torch.stack(encoder.router_bias_state())  # [L, N_r]
    logs = {
        "moe/expert_frac_max": float(f.max()),
        "moe/expert_frac_min": float(f.min()),
        "moe/bias_range": float((biases.max(dim=1).values - biases.min(dim=1).values).max()),
        "moe/tokens": float(stats[0, -1]),
    }
    if per_layer_logs:
        for layer, f_l in enumerate(f):
            logs[f"moe/layer{layer}/expert_frac_max"] = float(f_l.max())
            logs[f"moe/layer{layer}/expert_frac_min"] = float(f_l.min())
    return logs


__all__ = ["aux_free_bias_step", "expert_fractions"]
