"""Evaluator protocol and ordered gathering across ranks (plan Sec. 9, "Eval gather").

Evaluation loaders use an ``OrderedShardSampler`` (indices ``rank::world``, no padding), so each
rank produces ``(uids, payload)`` for a disjoint subset; :func:`gather_ordered` reassembles them in
uid order and asserts the ids are exactly ``0..N-1``. Both helpers are collectives: call them on
every rank, even when the local shard is empty.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from vivlmoe.training.distributed import DistContext, all_gather_object, all_gather_varlen
from vivlmoe.training.state import EvalContext


class Evaluator(ABC):
    """Runs on the unwrapped model in eval mode under ``no_grad``; returns metrics on rank 0."""

    name: str = "evaluator"

    @abstractmethod
    def run(self, model: nn.Module, loader: DataLoader, ec: EvalContext) -> dict[str, float] | None:
        """Return ``{metric: value}`` on rank 0 and ``None`` on the other ranks."""

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return None


def _check_ids(ids: Sequence[int]) -> None:
    n = len(ids)
    if list(ids) != list(range(n)):
        seen = set(ids)
        missing = sorted(set(range(n)) - seen)[:10]
        dupes = sorted({i for i in ids if list(ids).count(i) > 1})[:10] if len(seen) != n else []
        raise RuntimeError(
            f"gathered uids are not 0..{n - 1}: {len(seen)} unique of {n}; missing(head)={missing}; "
            f"duplicates(head)={dupes}. Eval loaders must use an ordered, unpadded shard sampler."
        )


def gather_ordered(ids: Sequence[int], payload: Sequence[Any], ctx: DistContext) -> list[Any]:
    """Concatenate per-rank ``payload`` items in ``ids`` order (returned on every rank)."""
    if len(ids) != len(payload):
        raise ValueError(f"ids ({len(ids)}) and payload ({len(payload)}) lengths differ")
    parts = all_gather_object((list(ids), list(payload)), ctx)
    all_ids: list[int] = []
    all_payload: list[Any] = []
    for part_ids, part_payload in parts:
        all_ids += [int(i) for i in part_ids]
        all_payload += list(part_payload)
    order = sorted(range(len(all_ids)), key=all_ids.__getitem__)
    _check_ids([all_ids[j] for j in order])
    return [all_payload[j] for j in order]


def gather_ordered_tensors(
    ids: torch.Tensor, values: torch.Tensor, ctx: DistContext
) -> torch.Tensor:
    """Tensor variant: ``values[i]`` belongs to uid ``ids[i]``; returns ``[N, ...]`` in uid order.

    Tensors are moved to the process-group device (``ctx.device``: CUDA under NCCL, which cannot
    gather CPU tensors) for the collective and returned on ``values.device``.
    """
    if ids.ndim != 1 or ids.shape[0] != values.shape[0]:
        raise ValueError(
            f"ids must be [n] and values [n, ...]; got {tuple(ids.shape)} / {tuple(values.shape)}"
        )
    out_device = values.device
    all_ids = torch.cat(all_gather_varlen(ids.to(ctx.device, torch.int64), ctx))
    all_values = torch.cat(all_gather_varlen(values.to(ctx.device), ctx))
    order = torch.argsort(all_ids)
    _check_ids(all_ids[order].tolist())
    return all_values[order].to(out_device)


__all__ = ["Evaluator", "gather_ordered", "gather_ordered_tensors"]
