"""Step-based learning-rate schedules (paper Sec. 6.1: cosine with 10% linear warmup).

Built as a ``LambdaLR`` so every optimizer group keeps its own base lr (Stage 3 uses two lrs);
the Trainer steps it exactly once per optimizer step (never per micro-batch).
"""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR

from vivlmoe.config.schema import ConfigError, ScheduleCfg


def resolve_warmup_steps(cfg: ScheduleCfg, total_steps: int) -> int:
    if total_steps < 1:
        raise ConfigError(f"total_steps must be >= 1, got {total_steps}")
    if cfg.warmup_steps is not None:
        warmup = cfg.warmup_steps
    else:
        warmup = int(round((cfg.warmup_ratio or 0.0) * total_steps))
    if warmup < 0:
        raise ConfigError("schedule.warmup_steps must be >= 0")
    if warmup >= total_steps and total_steps > 1:
        raise ConfigError(
            f"warmup ({warmup} steps) must be shorter than the total budget ({total_steps} steps)"
        )
    return min(warmup, max(total_steps - 1, 0))


def lr_factor(step: int, kind: str, warmup: int, total: int, min_lr_ratio: float) -> float:
    """Multiplier of the base lr after ``step`` completed optimizer steps."""
    if step < warmup:
        return (step + 1) / (warmup + 1)
    if kind == "constant":
        return 1.0
    decay_steps = max(total - warmup, 1)
    progress = min((step - warmup) / decay_steps, 1.0)
    if kind == "linear":
        shape = 1.0 - progress
    elif kind == "cosine":
        shape = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ConfigError(f"unknown schedule kind '{kind}'")
    return min_lr_ratio + (1.0 - min_lr_ratio) * shape


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: ScheduleCfg, total_steps: int
) -> LambdaLR:
    warmup = resolve_warmup_steps(cfg, total_steps)
    if not 0.0 <= cfg.min_lr_ratio <= 1.0:
        raise ConfigError("schedule.min_lr_ratio must be in [0, 1]")

    def fn(step: int) -> float:
        return lr_factor(step, cfg.kind, warmup, total_steps, cfg.min_lr_ratio)

    return LambdaLR(optimizer, lr_lambda=fn)


__all__ = ["build_scheduler", "lr_factor", "resolve_warmup_steps"]
