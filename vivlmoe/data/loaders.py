"""The single place that builds ``DataLoader``s and moves batches to a device."""

from __future__ import annotations

import logging
import os
import random
import sys
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from vivlmoe.config.schema import DataCfg
from vivlmoe.data.samplers import OrderedShardSampler, build_train_sampler

log = logging.getLogger(__name__)


def seed_worker(worker_id: int) -> None:
    """Seed ``random``/``numpy`` from the torch worker seed (base seed + worker id, set by torch)."""
    seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)


def _loader_kwargs(data_cfg: DataCfg) -> dict[str, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    workers = data_cfg.num_workers
    kw: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": data_cfg.pin_memory and torch.cuda.is_available(),
        "worker_init_fn": seed_worker if workers > 0 else None,
        "persistent_workers": data_cfg.persistent_workers and workers > 0,
    }
    if workers > 0:
        kw["prefetch_factor"] = data_cfg.prefetch_factor
        if sys.platform == "darwin":
            kw["multiprocessing_context"] = "spawn"  # fork + Arrow/OpenMP deadlocks on macOS
    return kw


def build_train_loader(
    dataset: Dataset[Any],
    collate_fn: Callable[[Sequence[Any]], Any],
    data_cfg: DataCfg,
    world_size: int = 1,
    rank: int = 0,
    seed: int = 0,
    groups: Sequence[Sequence[int]] | None = None,
) -> DataLoader[Any]:
    """Train loader with the one-caption-per-image sampler (``drop_last``, equal shards)."""
    n = len(dataset)  # type: ignore[arg-type]
    sampler = build_train_sampler(
        n,
        world_size,
        rank,
        seed,
        groups=groups,
        one_caption_per_image=data_cfg.one_caption_per_image,
    )
    g = torch.Generator()
    g.manual_seed(seed * 7919 + rank)  # worker base seeds; sampler order is seeded separately
    loader = DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=True,
        generator=g,
        **_loader_kwargs(data_cfg),
    )
    log.info(
        "train loader: %d samples/rank (%d images, %d records), batch %d, %d batches/epoch, workers %d",
        len(sampler),
        len(sampler.groups),
        n,
        data_cfg.batch_size,
        len(loader),
        data_cfg.num_workers,
    )
    return loader


def build_eval_loader(
    dataset: Dataset[Any],
    collate_fn: Callable[[Sequence[Any]], Any],
    data_cfg: DataCfg,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader[Any]:
    """Ordered ``rank::world`` sharding without padding; evaluators gather by ``uid``."""
    n = len(dataset)  # type: ignore[arg-type]
    sampler = OrderedShardSampler(n, world_size, rank)
    return DataLoader(
        dataset,
        batch_size=data_cfg.eval_batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=False,
        **_loader_kwargs(data_cfg),
    )


def move_to_device(batch: Any, device: torch.device | str, non_blocking: bool = False) -> Any:
    """Recursively move tensors inside dict/list/tuple; strings and other objects are untouched."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in batch.items()}
    if isinstance(batch, tuple):
        return tuple(move_to_device(v, device, non_blocking) for v in batch)
    if isinstance(batch, list):
        return [move_to_device(v, device, non_blocking) for v in batch]
    return batch


__all__ = ["build_eval_loader", "build_train_loader", "move_to_device", "seed_worker"]
