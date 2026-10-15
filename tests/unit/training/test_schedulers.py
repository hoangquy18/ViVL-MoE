from __future__ import annotations

import math

import pytest
import torch

from vivlmoe.config.schema import ConfigError, ScheduleCfg
from vivlmoe.training.schedulers import build_scheduler, lr_factor, resolve_warmup_steps


def _opt(lrs: list[float]) -> torch.optim.Optimizer:
    groups = [{"params": [torch.nn.Parameter(torch.zeros(2))], "lr": lr} for lr in lrs]
    return torch.optim.AdamW(groups, lr=1.0)


def test_warmup_then_cosine_to_min() -> None:
    cfg = ScheduleCfg(kind="cosine", warmup_ratio=None, warmup_steps=10, min_lr_ratio=0.1)
    total = 100
    assert lr_factor(0, "cosine", 10, total, 0.1) == pytest.approx(1 / 11)
    assert lr_factor(10, "cosine", 10, total, 0.1) == pytest.approx(1.0)
    assert lr_factor(55, "cosine", 10, total, 0.1) == pytest.approx(0.1 + 0.9 * 0.5)
    assert lr_factor(100, "cosine", 10, total, 0.1) == pytest.approx(0.1)
    assert lr_factor(500, "cosine", 10, total, 0.1) == pytest.approx(0.1)
    assert resolve_warmup_steps(cfg, total) == 10


def test_linear_and_constant() -> None:
    assert lr_factor(50, "linear", 0, 100, 0.0) == pytest.approx(0.5)
    assert lr_factor(100, "linear", 0, 100, 0.0) == pytest.approx(0.0)
    assert lr_factor(999, "constant", 0, 100, 0.0) == 1.0
    with pytest.raises(ConfigError):
        lr_factor(1, "bogus", 0, 10, 0.0)


def test_scheduler_preserves_per_group_base_lr_and_steps_once() -> None:
    opt = _opt([1e-4, 5e-5])
    cfg = ScheduleCfg(kind="cosine", warmup_ratio=0.1, warmup_steps=None)
    sched = build_scheduler(opt, cfg, total_steps=20)  # warmup = 2
    assert [g["lr"] for g in opt.param_groups] == pytest.approx([1e-4 / 3, 5e-5 / 3])
    for _ in range(5):
        opt.step()
        sched.step()
    f = lr_factor(5, "cosine", 2, 20, 0.0)
    assert sched.last_epoch == 5
    assert [g["lr"] for g in opt.param_groups] == pytest.approx([1e-4 * f, 5e-5 * f])
    # resume = optimizer state (carries the group lrs) + scheduler state (carries the position)
    opt2 = _opt([1e-4, 5e-5])
    sched2 = build_scheduler(opt2, cfg, 20)
    opt2.load_state_dict(opt.state_dict())
    sched2.load_state_dict(sched.state_dict())
    assert sched2.last_epoch == 5
    assert [g["lr"] for g in opt2.param_groups] == pytest.approx([1e-4 * f, 5e-5 * f])
    opt2.step()
    sched2.step()
    f6 = lr_factor(6, "cosine", 2, 20, 0.0)
    assert [g["lr"] for g in opt2.param_groups] == pytest.approx([1e-4 * f6, 5e-5 * f6])


def test_warmup_validation() -> None:
    with pytest.raises(ConfigError, match="shorter than the total"):
        resolve_warmup_steps(ScheduleCfg(warmup_ratio=None, warmup_steps=10), total_steps=10)
    with pytest.raises(ConfigError):
        resolve_warmup_steps(ScheduleCfg(warmup_ratio=0.1), total_steps=0)
    assert resolve_warmup_steps(ScheduleCfg(warmup_ratio=0.1), total_steps=1) == 0
    assert math.isclose(lr_factor(0, "cosine", 0, 1, 0.0), 1.0)
