"""Shared machinery of the downstream Tasks (captioning, VQA, ViMACSA; paper Sec. 6.1, plan 8.7).

:class:`DownstreamTask` owns everything the three objectives have in common:

* the model comes from :func:`vivlmoe.models.factory.build_task_model` (``seq2seq`` or
  ``classifier``); the freeze preset (``downstream_caption`` / ``downstream_seq2seq`` /
  ``downstream_cls``) is applied through :meth:`always_frozen`;
* the model is called exactly once per micro-batch (``model(batch, forward_mode)``); when the
  backbone is MoE the router stabilisers (z-loss Eq. 20, modality-importance loss Eq. 24, optional
  Switch-LB) are added by ``RouterLossComputer`` (milestone M6, imported lazily) and the aux-free
  bias update (Eq. 22) runs in :meth:`post_optimizer_step` (or per micro-batch) on the dispatch
  counts of every micro-batch of the step (the Trainer has no startup forward, so nothing is reset
  in ``after_first_backward``);
* eval loaders use the ordered shard sampler; the routing probe (Sec. 6.5) is attached by
  ``maybe_add_routing_probe`` (M6) at the end of :meth:`evaluators` when configured (the Trainer
  runs it on its own ``routing_probe.every_steps`` cadence, independent of the eval cadence);
* checkpoints record ``dev_split_sha1`` (holdout id lists) and the image ``transform_spec``.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from vivlmoe.config.schema import ConfigError, DataSourceCfg, RunConfig
from vivlmoe.data.builders import build_transform_for, draft_size_for
from vivlmoe.data.loaders import build_eval_loader, build_train_loader
from vivlmoe.data.splits import ids_sha1, load_ids
from vivlmoe.data.tokenization import TokenizerBundle, build_tokenizers
from vivlmoe.data.transforms import ImageTransformSpec
from vivlmoe.models.freezing import FREEZE_PRESETS
from vivlmoe.models.outputs import RouterOutput
from vivlmoe.tasks.base import Evaluator, LoaderBundle, Task
from vivlmoe.tasks.evaluators.dump import metrics_path, predictions_path
from vivlmoe.training.distributed import DistContext
from vivlmoe.training.state import StepOutput, TrainState

log = logging.getLogger(__name__)


def encoder_of(model: nn.Module) -> nn.Module:
    """The shared encoder ``f_enc`` of any task model (``model`` for a backbone, ``model.backbone`` otherwise)."""
    return getattr(model, "backbone", model).encoder


def _import_m6(what: str) -> Any:
    """Lazy import of the MoE helpers provided by milestone M6 with a clear error when absent."""
    try:
        if what == "RouterLossComputer":
            from vivlmoe.losses.router_losses import RouterLossComputer

            return RouterLossComputer
        if what == "aux_free_bias_step":
            from vivlmoe.models.moe.balancing import aux_free_bias_step

            return aux_free_bias_step
        if what == "maybe_add_routing_probe":
            from vivlmoe.tasks.evaluators.routing import maybe_add_routing_probe

            return maybe_add_routing_probe
    except ImportError as exc:
        raise ImportError(
            f"{what} is provided by milestone M6 (MoE) and is not importable: {exc}. "
            "A MoE backbone / routing probe cannot be trained downstream without it."
        ) from exc
    raise ValueError(f"unknown M6 symbol {what!r}")


class DownstreamTask(Task):
    """Base class for the downstream objectives; subclasses set the class attributes below."""

    name = "downstream"
    model_kind: str = "seq2seq"  # build_task_model kind
    forward_mode: str = "joint_caption"  # model(batch, mode)
    freeze_preset: str = "downstream_seq2seq"  # vivlmoe.models.freezing.FREEZE_PRESETS key
    needs_decoder_tokenizer: bool = True

    def __init__(self, cfg: RunConfig, ctx: DistContext) -> None:
        super().__init__(cfg, ctx)
        if cfg.data.train is None:
            log.warning("task %s: data.train is not set (evaluation-only configuration)", self.name)
        self.run_dir = Path(cfg.run.dir) / cfg.run.name
        self._router_loss: Callable[..., tuple[torch.Tensor, dict[str, float]]] | None = None
        self._evaluators: dict[str, Evaluator] | None = None

    # ------------------------------------------------------------------ model
    def build_model(self) -> nn.Module:
        from vivlmoe.models.factory import build_task_model

        return build_task_model(self.model_kind, self.cfg.model)

    def always_frozen(self) -> list[str]:
        """Freeze preset of the objective (ViT + ``logit_scale``; captioning also freezes the text tower)."""
        return list(FREEZE_PRESETS[self.freeze_preset])

    @property
    def is_moe(self) -> bool:
        return self.cfg.model.is_moe

    @property
    def amp_dtype(self) -> torch.dtype | None:
        amp = self.cfg.train.amp
        if not amp.enabled:
            return None
        return torch.bfloat16 if amp.dtype == "bf16" else torch.float16

    # ------------------------------------------------------------------ data helpers
    @cached_property
    def tokenizers(self) -> TokenizerBundle:
        return build_tokenizers(
            self.cfg.model, self.cfg.data, need_decoder=self.needs_decoder_tokenizer
        )

    @cached_property
    def transforms(self) -> tuple[ImageTransformSpec, Any, Any]:
        """``(spec, train_transform, eval_transform)`` derived from the vision tower."""
        return build_transform_for(self.cfg)

    @property
    def transform_spec(self) -> ImageTransformSpec:
        return self.transforms[0]

    @property
    def draft_size(self) -> int | None:
        return draft_size_for(self.cfg, self.transform_spec)

    def holdout_sha1s(self) -> dict[str, str]:
        """sha1 of every committed holdout id list referenced by ``data.train`` / ``data.evals``."""
        out: dict[str, str] = {}
        sources = {"train": self.cfg.data.train, **self.cfg.data.evals}
        for name, src in sources.items():
            if src is None or src.holdout is None:
                continue
            if src.holdout.ids_file is None:
                log.warning("%s: holdout computed on the fly, no dev_split_sha1 recorded", name)
                continue
            out[name] = ids_sha1(load_ids(src.holdout.ids_file))
        return out

    def train_source(self) -> DataSourceCfg:
        if self.cfg.data.train is None:
            raise ConfigError(f"task {self.name}: data.train is required for training")
        return self.cfg.data.train

    def make_train_loader(
        self, dataset: Any, collate: Callable[..., dict[str, Any]], groups: Any = None
    ) -> DataLoader:
        return build_train_loader(
            dataset,
            collate,
            self.cfg.data,
            world_size=self.ctx.world_size,
            rank=self.ctx.rank,
            seed=self.cfg.run.seed,
            groups=groups,
        )

    def make_eval_loader(self, dataset: Any, collate: Callable[..., dict[str, Any]]) -> DataLoader:
        return build_eval_loader(
            dataset, collate, self.cfg.data, world_size=self.ctx.world_size, rank=self.ctx.rank
        )

    @abstractmethod
    def build_train_loader(self) -> DataLoader:
        """Training loader (sampler with ``set_epoch`` / ``skip_first``)."""

    @abstractmethod
    def build_eval_loaders(self) -> dict[str, DataLoader]:
        """One ordered-shard loader per ``data.evals`` entry (also used by ``scripts/evaluate.py``)."""

    def build_loaders(self) -> LoaderBundle:
        return LoaderBundle(train=self.build_train_loader(), evals=self.build_eval_loaders())

    # ------------------------------------------------------------------ evaluation
    @abstractmethod
    def build_evaluators(self) -> dict[str, Evaluator]:
        """Task-specific evaluators keyed by split (before the routing probe is attached)."""

    def evaluators(self) -> dict[str, Evaluator]:
        if self._evaluators is None:
            evs = self.build_evaluators()
            if self.cfg.analysis.routing_probe is not None:
                if self.is_moe:
                    _import_m6("maybe_add_routing_probe")(
                        evs, self.cfg, self.ctx, self.forward_mode
                    )
                else:  # the dense baseline shares the downstream defaults (probe configured)
                    log.warning(
                        "analysis.routing_probe is set but the model is dense (model.moe: null): "
                        "no routing probe will run"
                    )
            self._evaluators = evs
        return self._evaluators

    def selection_metric(self) -> tuple[str | None, str]:
        """``<first eval split>/<first task metric>`` (max) when ``eval.selection_metric`` is unset."""
        if not self.cfg.data.evals or not self.cfg.task.metrics:
            return None, "max"
        return f"{next(iter(self.cfg.data.evals))}/{self.cfg.task.metrics[0]}", "max"

    # ------------------------------------------------------------------ MoE stabilisers
    def router_aux(
        self, router_outputs: list[RouterOutput], device: torch.device
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """``(aux, logs)``: z / modality-importance / LB losses (Eq. 20, 24, 19) or a 0-tensor for dense."""
        if not self.is_moe:
            return torch.zeros((), device=device), {}
        if self._router_loss is None:
            self._router_loss = _import_m6("RouterLossComputer")(self.cfg.loss.router)
        aux, logs = self._router_loss(router_outputs)
        return aux, {k: float(v) for k, v in logs.items()}

    def finish_train_step(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        router_outputs: list[RouterOutput],
        batch: dict[str, Any],
        num_tokens: int = 0,
        metrics: dict[str, float] | None = None,
    ) -> StepOutput:
        """Add the router aux loss, run the per-micro-batch bias update when configured, build the output."""
        aux, logs = self.router_aux(router_outputs, loss.device)
        total = loss + aux
        out_metrics = dict(metrics or {})
        out_metrics["task_loss"] = float(loss.detach())
        if self.is_moe:
            out_metrics["aux"] = float(aux.detach())
            out_metrics.update(logs)
            moe = self.cfg.model.moe
            assert moe is not None
            if moe.aux_free_bias and moe.bias_update_every == "micro":
                raw = getattr(model, "module", model)
                _import_m6("aux_free_bias_step")(encoder_of(raw), self.ctx, moe.bias_update_rate)
        return StepOutput(
            loss=total, batch_size=len(batch["uids"]), num_tokens=num_tokens, metrics=out_metrics
        )

    def post_optimizer_step(self, trainer: Any, state: TrainState) -> None:
        """Aux-free bias update (Eq. 22) once per optimizer step, all-reduced across ranks (decision 7)."""
        moe = self.cfg.model.moe
        if moe is None or not moe.aux_free_bias or moe.bias_update_every != "step":
            return
        logs = _import_m6("aux_free_bias_step")(
            encoder_of(trainer.raw_model), self.ctx, moe.bias_update_rate
        )
        if state.step % self.cfg.train.log_every == 0:
            trainer.writer.log_scalars({str(k): float(v) for k, v in logs.items()}, state.step)

    # ------------------------------------------------------------------ bookkeeping
    def checkpoint_metadata(self) -> dict[str, Any]:
        sha1: Any = None
        holdouts = self.holdout_sha1s()
        if holdouts:
            unique = sorted(set(holdouts.values()))
            sha1 = unique[0] if len(unique) == 1 else holdouts
        return {"dev_split_sha1": sha1, "transform_spec": self.transform_spec.to_dict()}

    def predictions_path(self, split: str, step: int) -> Path:
        return predictions_path(self.run_dir, split, step)

    def metrics_path(self, split: str, step: int) -> Path:
        return metrics_path(self.run_dir, split, step)

    @staticmethod
    def count_label_tokens(batch: dict[str, Any]) -> int:
        labels = batch.get("labels")
        if labels is None or labels.ndim != 2:
            return 0
        return int((labels != -100).sum())


__all__ = ["DownstreamTask", "encoder_of"]
