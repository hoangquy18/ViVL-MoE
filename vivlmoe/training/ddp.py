"""DistributedDataParallel wrapping and gradient-sync helpers (plan decisions 6, 19, 22)."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from vivlmoe.config.schema import ConfigError, DistCfg
from vivlmoe.training.distributed import DistContext

log = logging.getLogger(__name__)


def wrap_ddp(model: nn.Module, ctx: DistContext, cfg: DistCfg) -> nn.Module:
    """Wrap in DDP when distributed; a single-process run returns ``model`` unchanged.

    ``broadcast_buffers=False``: router biases are kept identical by all-reduced updates instead.
    ``find_unused_parameters`` defaults to False because idle experts are always called (decision 6).
    """
    if not ctx.is_distributed:
        return model
    device_ids = [ctx.local_rank] if ctx.device.type == "cuda" else None
    ddp = DistributedDataParallel(
        model,
        device_ids=device_ids,
        broadcast_buffers=cfg.broadcast_buffers,
        find_unused_parameters=cfg.find_unused_parameters,
        gradient_as_bucket_view=True,
        bucket_cap_mb=cfg.bucket_cap_mb,
        static_graph=cfg.static_graph,
    )
    if cfg.grad_compression_bf16:
        if ctx.backend != "nccl":
            raise ConfigError("dist.grad_compression_bf16 requires the nccl backend")
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

        ddp.register_comm_hook(state=None, hook=default_hooks.bf16_compress_hook)
        log.info("DDP: registered bf16 gradient-compression hook")
    log.info(
        "DDP: world=%d find_unused=%s broadcast_buffers=%s static_graph=%s bucket=%dMB",
        ctx.world_size,
        cfg.find_unused_parameters,
        cfg.broadcast_buffers,
        cfg.static_graph,
        cfg.bucket_cap_mb,
    )
    return ddp


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


@contextlib.contextmanager
def sync_context(model: nn.Module, sync: bool) -> Iterator[None]:
    """``model.no_sync()`` for accumulation micro-batches (``sync=False``); no-op otherwise."""
    if isinstance(model, DistributedDataParallel) and not sync:
        with model.no_sync():
            yield
    else:
        yield


def check_all_trainable_params_received_grad(model: nn.Module) -> None:
    """Hard error listing trainable parameters without ``.grad`` after the first backward.

    Plan decision 19: an unused trainable parameter would deadlock DDP (``find_unused=False``)
    or silently receive no update, so the check runs once at startup on every rank.
    """
    missing = [n for n, p in unwrap(model).named_parameters() if p.requires_grad and p.grad is None]
    if missing:
        shown = ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else "")
        raise RuntimeError(
            f"{len(missing)} trainable parameter(s) received no gradient in the first backward: "
            f"{shown}. Freeze them (optim.freeze) or make the task use them."
        )


def grad_norm_of(params: list[torch.Tensor]) -> torch.Tensor:
    """L2 norm over all ``.grad`` tensors (0 when no grads)."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.zeros(())
    return torch.norm(torch.stack([g.detach().float().norm() for g in grads]))


__all__ = [
    "check_all_trainable_params_received_grad",
    "grad_norm_of",
    "sync_context",
    "unwrap",
    "wrap_ddp",
]
