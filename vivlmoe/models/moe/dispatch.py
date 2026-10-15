"""Token -> expert dispatch without capacity limits or token dropping (plan decision 6).

One ``argsort`` over the flattened ``[T * k]`` assignments groups the tokens of every expert into a
contiguous slice; each expert is applied to its slice and the gate-weighted outputs are summed back
with ``index_add_``. **Idle experts are still called on an empty ``[0, d]`` slice** so every expert
parameter stays in the autograd graph (DDP with ``find_unused_parameters=False`` never sees an
unused parameter). The combine runs in fp32. The only device->host sync per call is the
``counts.tolist()`` that sizes the per-expert slices.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def dispatch(
    x: torch.Tensor,
    experts: Sequence[nn.Module],
    topk_idx: torch.Tensor,
    gates: torch.Tensor,
) -> torch.Tensor:
    """``sum_{i in S(x)} p~_i E_i(x)`` for every token: ``x [T, d]``, ``topk_idx/gates [T, k]`` -> fp32 ``[T, d]``."""
    if x.ndim != 2:
        raise ValueError(f"dispatch expects x [T, d], got {tuple(x.shape)}")
    if topk_idx.shape != gates.shape or topk_idx.ndim != 2 or topk_idx.shape[0] != x.shape[0]:
        raise ValueError(
            f"topk_idx {tuple(topk_idx.shape)} and gates {tuple(gates.shape)} must be [T={x.shape[0]}, k]"
        )
    num_experts = len(experts)
    t, k = topk_idx.shape
    flat_expert = topk_idx.reshape(-1)  # [T * k]; in range by construction (``topk`` over [T, N_r])
    order = torch.argsort(flat_expert, stable=True)  # slots grouped by expert
    token_of_slot = torch.div(order, k, rounding_mode="floor")  # [T * k]
    # the single device->host sync of the dispatch (``one_hot`` raises on out-of-range indices)
    counts = F.one_hot(flat_expert, num_experts).sum(dim=0).tolist() if t > 0 else [0] * num_experts
    out_dtype = torch.promote_types(x.dtype, torch.float32)  # fp32 (double stays double: gradcheck)
    flat_gates = gates.reshape(-1).to(out_dtype)
    out = torch.zeros(t, x.shape[1], dtype=out_dtype, device=x.device)
    start = 0
    for e, expert in enumerate(experts):
        n = int(counts[e])
        slots = order[start : start + n]
        tokens = token_of_slot[start : start + n]
        y = expert(x.index_select(0, tokens))  # [n, d]; n == 0 keeps the expert in the graph
        out.index_add_(0, tokens, y.to(out_dtype) * flat_gates.index_select(0, slots).unsqueeze(1))
        start += n
    return out


def dispatch_naive(
    x: torch.Tensor,
    experts: Sequence[nn.Module],
    topk_idx: torch.Tensor,
    gates: torch.Tensor,
) -> torch.Tensor:
    """Per-token reference loop (tests only)."""
    out_dtype = torch.promote_types(x.dtype, torch.float32)
    rows = []
    for t in range(x.shape[0]):
        acc = torch.zeros(x.shape[1], dtype=out_dtype, device=x.device)
        for j in range(topk_idx.shape[1]):
            e = experts[int(topk_idx[t, j])]
            acc = acc + gates[t, j].to(out_dtype) * e(x[t : t + 1]).to(out_dtype)[0]
        rows.append(acc)
    if not rows:
        return torch.zeros(0, x.shape[1], dtype=out_dtype, device=x.device)
    return torch.stack(rows)


__all__ = ["dispatch", "dispatch_naive"]
