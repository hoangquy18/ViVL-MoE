"""Early stopping on the broadcast eval dict (identical decision on every rank)."""

from __future__ import annotations

import logging
import math

from vivlmoe.config.schema import EarlyStoppingCfg
from vivlmoe.training.state import TrainState

log = logging.getLogger(__name__)


def is_improvement(value: float, best: float | None, mode: str, min_delta: float = 0.0) -> bool:
    if best is None or math.isnan(best):
        return not math.isnan(value)
    if math.isnan(value):
        return False
    return value > best + min_delta if mode == "max" else value < best - min_delta


class EarlyStopping:
    """Tracks the selection metric in :class:`TrainState` (so it survives checkpoints)."""

    def __init__(self, cfg: EarlyStoppingCfg, mode: str) -> None:
        self.enabled = cfg.enabled
        self.patience = cfg.patience
        self.min_delta = cfg.min_delta
        self.mode = mode

    def record(self, value: float, state: TrainState) -> bool:
        """Update ``best_metric`` / ``evals_without_improvement``; return True on improvement."""
        improved = is_improvement(value, state.best_metric, self.mode, self.min_delta)
        if improved:
            state.best_metric = float(value)
            state.best_step = state.step
            state.evals_without_improvement = 0
        else:
            state.evals_without_improvement += 1
        return improved

    def should_stop(self, state: TrainState) -> bool:
        stop = self.enabled and state.evals_without_improvement >= self.patience
        if stop:
            log.info(
                "early stopping: %d evaluations without improvement (best %.6g at step %s)",
                state.evals_without_improvement,
                state.best_metric,
                state.best_step,
            )
        return stop


__all__ = ["EarlyStopping", "is_improvement"]
