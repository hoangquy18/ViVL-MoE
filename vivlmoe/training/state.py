"""Typed containers exchanged between the Trainer, Tasks and Evaluators."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:  # pragma: no cover
    from vivlmoe.training.distributed import DistContext
    from vivlmoe.training.logging import MetricsWriter


@dataclass
class TrainState:
    step: int = 0  # optimizer steps completed
    epoch: int = 0
    micro_in_epoch: int = (
        0  # micro-batches consumed in the current epoch (multiple of accum at save time)
    )
    samples_seen: int = 0
    best_metric: float | None = None
    best_step: int | None = None
    evals_without_improvement: int = 0
    skipped_steps: int = 0  # fp16 overflow skips
    nonfinite_micro_batches: int = 0
    last_eval_step: int = -1  # persisted so a resume never repeats an eval / checkpoint
    last_ckpt_step: int = -1

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, d: dict[str, Any]) -> None:
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)


@dataclass
class StepContext:
    step: int
    micro: int
    accum: int
    device: torch.device
    is_train: bool = True


@dataclass
class StepOutput:
    """What a Task returns from ``train_step``. ``loss`` is the full (un-divided) scalar loss."""

    loss: torch.Tensor
    batch_size: int
    num_tokens: int = 0
    metrics: dict[str, float] = field(
        default_factory=dict
    )  # detached scalars, averaged across ranks at log time


@dataclass
class EvalContext:
    ctx: DistContext
    step: int
    run_dir: Path
    split_name: str
    writer: MetricsWriter | None = None
    save_predictions: bool = True
