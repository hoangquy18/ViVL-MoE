"""Process-group setup and the few collectives the trainer and evaluators use.

Plan decision 7: no collective is ever called from inside an ``nn.Module``; tasks and evaluators
call the helpers below with an explicit :class:`DistContext`. torchrun provides ``RANK``,
``LOCAL_RANK`` and ``WORLD_SIZE``; a plain ``python`` launch runs single-process. MPS is never used.
"""

from __future__ import annotations

import datetime
import logging
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from vivlmoe.config.schema import ConfigError, DistCfg

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str  # "nccl" | "gloo" | "none" (single process)

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @classmethod
    def single_process(cls, device: torch.device | str | None = None) -> DistContext:
        dev = torch.device(device) if device is not None else _default_device(0)
        return cls(rank=0, local_rank=0, world_size=1, device=dev, backend="none")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"environment variable {name}={raw!r} is not an integer") from None


def _default_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def init_distributed(cfg: DistCfg, init_method: str | None = None) -> DistContext:
    """Initialise the process group from torchrun's environment (or run single-process).

    ``init_method`` defaults to ``env://``; tests pass ``file://...`` to spawn gloo groups.
    """
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", 0)
    device = _default_device(local_rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world_size == 1 and init_method is None:
        log.info("single-process run on %s", device)
        return DistContext(0, local_rank, 1, device, "none")
    backend = cfg.backend
    if backend == "auto":
        backend = "nccl" if device.type == "cuda" else "gloo"
    if backend == "nccl" and device.type != "cuda":
        raise ConfigError("dist.backend=nccl requires CUDA devices")
    if dist.is_initialized():
        raise RuntimeError("torch.distributed is already initialised")
    dist.init_process_group(
        backend=backend,
        init_method=init_method or "env://",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=cfg.timeout_minutes),
    )
    ctx = DistContext(rank, local_rank, world_size, device, backend)
    log.info("distributed: rank %d/%d backend=%s device=%s", rank, world_size, backend, device)
    return ctx


def destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# --------------------------------------------------------------------------------------------
# collectives (no-ops in single-process mode)
# --------------------------------------------------------------------------------------------
def barrier(ctx: DistContext) -> None:
    if ctx.is_distributed:
        dist.barrier()


def all_reduce_sum_(t: torch.Tensor, ctx: DistContext) -> torch.Tensor:
    """In-place sum across ranks; returns ``t``."""
    if ctx.is_distributed:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def all_reduce_mean(t: torch.Tensor, ctx: DistContext) -> torch.Tensor:
    """Mean across ranks of a (float) tensor, returned as a new tensor."""
    out = t.detach().clone().float()
    if ctx.is_distributed:
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        out /= ctx.world_size
    return out


def all_gather_varlen(t: torch.Tensor, ctx: DistContext) -> list[torch.Tensor]:
    """Gather tensors whose first dimension differs per rank (zero-length shards allowed)."""
    if not ctx.is_distributed:
        return [t]
    n_local = torch.tensor([t.shape[0]], dtype=torch.int64, device=t.device)
    sizes = [torch.zeros_like(n_local) for _ in range(ctx.world_size)]
    dist.all_gather(sizes, n_local)
    lengths = [int(s.item()) for s in sizes]
    max_len = max(lengths)
    if max_len == 0:
        return [t[:0].clone() for _ in range(ctx.world_size)]
    pad_shape = (max_len, *t.shape[1:])
    padded = t.new_zeros(pad_shape)
    padded[: t.shape[0]] = t
    chunks = [t.new_zeros(pad_shape) for _ in range(ctx.world_size)]
    dist.all_gather(chunks, padded)
    return [c[:n] for c, n in zip(chunks, lengths, strict=True)]


def all_gather_object(obj: Any, ctx: DistContext) -> list[Any]:
    if not ctx.is_distributed:
        return [obj]
    out: list[Any] = [None] * ctx.world_size
    dist.all_gather_object(out, obj)
    return out


def broadcast_object(obj: Any, ctx: DistContext, src: int = 0) -> Any:
    if not ctx.is_distributed:
        return obj
    box = [obj if ctx.rank == src else None]
    device = ctx.device if ctx.backend == "nccl" else None
    dist.broadcast_object_list(box, src=src, device=device)
    return box[0]


# --------------------------------------------------------------------------------------------
# seeding
# --------------------------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    """Identical seed on every rank: model init must match before DDP broadcasts rank-0 weights."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def seed_rank_streams(seed: int, ctx: DistContext) -> int:
    """Offset the RNG streams per rank (dropout, sampling) after the model is built and wrapped."""
    rank_seed = seed + 1_000_003 * ctx.rank
    random.seed(rank_seed)
    np.random.seed(rank_seed % (2**32))
    torch.manual_seed(rank_seed)
    log.debug("rank %d RNG streams seeded with %d", ctx.rank, rank_seed)
    return rank_seed


__all__ = [
    "DistContext",
    "all_gather_object",
    "all_gather_varlen",
    "all_reduce_mean",
    "all_reduce_sum_",
    "barrier",
    "broadcast_object",
    "destroy_distributed",
    "init_distributed",
    "seed_everything",
    "seed_rank_streams",
]
