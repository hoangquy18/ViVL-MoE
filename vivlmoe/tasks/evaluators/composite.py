"""Run several evaluators on the same split and merge their metric dicts."""

from __future__ import annotations

from typing import Any

from torch import nn
from torch.utils.data import DataLoader

from vivlmoe.tasks.evaluators.base import Evaluator
from vivlmoe.training.state import EvalContext


class CompositeEvaluator(Evaluator):
    name = "composite"

    def __init__(self, evaluators: list[Evaluator]) -> None:
        if not evaluators:
            raise ValueError("CompositeEvaluator needs at least one evaluator")
        self.evaluators = list(evaluators)
        names = [e.name for e in self.evaluators]
        if len(set(names)) != len(names):
            raise ValueError(f"evaluator names must be unique within a composite: {names}")

    def run(self, model: nn.Module, loader: DataLoader, ec: EvalContext) -> dict[str, float] | None:
        merged: dict[str, float] = {}
        for ev in self.evaluators:  # every sub-evaluator runs on every rank (collectives inside)
            res = ev.run(model, loader, ec)
            if ec.ctx.is_main:
                if res is None:
                    raise RuntimeError(f"evaluator '{ev.name}' returned None on rank 0")
                clash = sorted(set(res) & set(merged))
                if clash:
                    raise KeyError(f"evaluator '{ev.name}' re-defines metrics {clash}")
                merged.update(res)
        return merged if ec.ctx.is_main else None

    def state_dict(self) -> dict[str, Any]:
        return {ev.name: ev.state_dict() for ev in self.evaluators}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for ev in self.evaluators:
            if ev.name in state:
                ev.load_state_dict(state[ev.name])


__all__ = ["CompositeEvaluator"]
