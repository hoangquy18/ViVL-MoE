"""Training infrastructure: distributed helpers, DDP, optimisation, checkpoints and the Trainer.

``Trainer`` is exported lazily: it imports :mod:`vivlmoe.tasks`, which in turn imports the
distributed helpers from this package.
"""

from __future__ import annotations

from typing import Any

from vivlmoe.training.distributed import DistContext, init_distributed
from vivlmoe.training.state import EvalContext, StepContext, StepOutput, TrainState

__all__ = [
    "DistContext",
    "EvalContext",
    "StepContext",
    "StepOutput",
    "TrainState",
    "Trainer",
    "init_distributed",
]


def __getattr__(name: str) -> Any:
    if name == "Trainer":
        from vivlmoe.training.trainer import Trainer

        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
