"""The Task protocol: everything objective-specific the Trainer delegates (plan 8.7).

One Trainer, one Task per *objective* (``stage1_teacher``, ``contrastive``, ``captioning``, ``vqa``,
``vimacsa``). The Task builds the model and loaders, computes the loss per micro-batch, owns its
evaluators and runs the MoE aux-free bias update in :meth:`post_optimizer_step` (decision 7).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from torch import nn
from torch.utils.data import DataLoader

from vivlmoe.config.schema import RunConfig
from vivlmoe.tasks.evaluators.base import Evaluator
from vivlmoe.training.distributed import DistContext
from vivlmoe.training.state import StepContext, StepOutput, TrainState

if TYPE_CHECKING:  # pragma: no cover
    from vivlmoe.training.trainer import Trainer


@dataclass
class LoaderBundle:
    train: DataLoader | None
    evals: dict[str, DataLoader] = field(default_factory=dict)


class Task(ABC):
    name: str = "task"

    def __init__(self, cfg: RunConfig, ctx: DistContext) -> None:
        self.cfg = cfg
        self.ctx = ctx

    # -- model -------------------------------------------------------------------------------
    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build the task model on CPU (moved / wrapped by the Trainer)."""

    def always_frozen(self) -> list[str]:
        """Module paths frozen regardless of ``optim.freeze`` (e.g. removed T5 encoder stack)."""
        return []

    # -- data --------------------------------------------------------------------------------
    @abstractmethod
    def build_loaders(self) -> LoaderBundle:
        """Train loader (sampler with ``set_epoch`` / ``skip_first``) and eval loaders by split."""

    # -- training ----------------------------------------------------------------------------
    @abstractmethod
    def train_step(self, model: nn.Module, batch: dict[str, Any], sc: StepContext) -> StepOutput:
        """One forward through the (DDP-wrapped) model; returns the un-divided loss."""

    def post_optimizer_step(self, trainer: Trainer, state: TrainState) -> None:
        """Called on every rank after ``optimizer.step()`` (MoE bias update lives here)."""
        return None

    def after_first_backward(self, model: nn.Module) -> None:
        """Called once with the unwrapped model after the first backward (e.g. reset counters)."""
        return None

    def on_train_mode(self, raw_model: nn.Module) -> None:
        """Called after every ``model.train()`` so frozen towers can stay in eval mode."""
        return None

    # -- evaluation --------------------------------------------------------------------------
    def evaluators(self) -> dict[str, Evaluator]:
        """Evaluator per eval split name (keys must match ``LoaderBundle.evals``)."""
        return {}

    def selection_metric(self) -> tuple[str | None, str]:
        """Default ``('<split>/<metric>', 'max'|'min')`` when ``eval.selection_metric`` is unset."""
        return None, "max"

    # -- state -------------------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return None

    def checkpoint_metadata(self) -> dict[str, Any]:
        """Optional ``dev_split_sha1`` / ``transform_spec`` entries stored in every checkpoint."""
        return {}


__all__ = ["Evaluator", "LoaderBundle", "Task"]
