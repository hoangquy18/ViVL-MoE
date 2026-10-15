"""Contrastive pretraining task (paper Sec. 4.2-4.3, Eq. 13; Sec. 5.4 sparse training).

Used unchanged for Stage 2 (ViT + ``W_v`` frozen), Stage 3 (end-to-end) and MoE sparse training:
the differences live in the config (freeze preset, lr groups, ``model.moe``). One
``model(batch, 'contrastive')`` per micro-batch (plan decision 2) returns both single-modality
embeddings; the loss is the symmetric InfoNCE with global-batch negatives plus, for a MoE encoder,
the router auxiliary losses (``RouterLossComputer``, plan decision 8). The aux-free bias update
(Eq. 22) runs in :meth:`post_optimizer_step` (plan decision 7) on the dispatch counts of every
micro-batch of the step (no counter reset in ``after_first_backward``: the Trainer has no startup
forward). Negatives are the per-step global batch ``B x world`` (decision 15), not the optimizer batch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from torch import nn
from torch.utils.data import DataLoader

from vivlmoe.config.schema import ConfigError, RunConfig
from vivlmoe.data.builders import (
    build_image_text_train,
    build_records,
    build_transform_for,
    draft_size_for,
)
from vivlmoe.data.collators import ContrastiveCollator
from vivlmoe.data.datasets import RetrievalEvalDataset
from vivlmoe.data.loaders import build_train_loader
from vivlmoe.data.records import ImageTextRecord
from vivlmoe.data.tokenization import TokenizerBundle, build_tokenizers
from vivlmoe.losses.contrastive import ContrastiveLoss
from vivlmoe.tasks.base import LoaderBundle, Task
from vivlmoe.tasks.evaluators.base import Evaluator
from vivlmoe.tasks.evaluators.retrieval import (
    RetrievalEvaluator,
    RetrievalLoaders,
    build_retrieval_loaders,
    text_collator,
)
from vivlmoe.tasks.registry import TASKS
from vivlmoe.training.distributed import DistContext
from vivlmoe.training.state import StepContext, StepOutput, TrainState

if TYPE_CHECKING:  # pragma: no cover
    from vivlmoe.training.trainer import Trainer

log = logging.getLogger(__name__)


def encoder_of(model: nn.Module) -> nn.Module:
    """``f_enc`` of any task model (``model`` for a backbone, ``model.backbone`` for heads)."""
    return getattr(model, "backbone", model).encoder


def _router_loss_computer(cfg: RunConfig) -> Any:
    try:
        from vivlmoe.losses.router_losses import RouterLossComputer
    except ImportError as exc:
        raise ImportError(
            "model.moe is set but vivlmoe.losses.router_losses.RouterLossComputer is not available "
            "(milestone M6)"
        ) from exc
    return RouterLossComputer(cfg.loss.router)


def _aux_free_bias_step(encoder: nn.Module, ctx: DistContext, eta: float) -> dict[str, float]:
    try:
        from vivlmoe.models.moe.balancing import aux_free_bias_step
    except ImportError as exc:
        raise ImportError(
            "model.moe.aux_free_bias requires vivlmoe.models.moe.balancing.aux_free_bias_step "
            "(milestone M6)"
        ) from exc
    return aux_free_bias_step(encoder, ctx, eta)


@TASKS.register("contrastive")
class ContrastiveTask(Task):
    name = "contrastive"

    def __init__(self, cfg: RunConfig, ctx: DistContext) -> None:
        super().__init__(cfg, ctx)
        lc = cfg.loss.contrastive
        self.loss = ContrastiveLoss(lc.gather, lc.symmetric_reduction)
        self._router_loss: Any = None
        self._tokenizers: TokenizerBundle | None = None
        self._eval_datasets: dict[str, RetrievalEvalDataset] = {}
        self._dev_split_sha1: dict[str, str | None] = {}
        self._transform_spec: dict[str, Any] | None = None
        self._evaluators: dict[str, Evaluator] | None = None

    # ------------------------------------------------------------------ model
    def build_model(self) -> nn.Module:
        from vivlmoe.models.factory import build_task_model

        return build_task_model("backbone", self.cfg.model)

    # ------------------------------------------------------------------ data
    def tokenizers(self) -> TokenizerBundle:
        if self._tokenizers is None:
            self._tokenizers = build_tokenizers(self.cfg.model, self.cfg.data)
        return self._tokenizers

    def _run_dir(self) -> Path | None:
        return Path(self.cfg.run.dir) / self.cfg.run.name if self.ctx.is_main else None

    def build_train_loader(self) -> DataLoader:
        """Image-text train loader (``build_image_text_train`` + ``ContrastiveCollator``)."""
        cfg, d = self.cfg, self.cfg.data
        if d.train is None:
            raise ConfigError("contrastive needs data.train (an image-text source)")
        tok = self.tokenizers()
        spec, train_tf, _eval_tf = build_transform_for(cfg)
        self._transform_spec = spec.to_dict()
        train_ds, groups = build_image_text_train(
            d.train, train_tf, draft_size_for(cfg, spec), self._run_dir(), self.ctx.is_main
        )
        negatives = int(d.batch_size) * self.ctx.world_size
        log.info(
            "contrastive negatives per step = batch %d x world %d = %d pairs (paper: 2048); "
            "grad_accum_steps %d enlarges the optimizer batch, not the Eq. 13 negatives",
            d.batch_size,
            self.ctx.world_size,
            negatives,
            cfg.train.grad_accum_steps,
        )
        return build_train_loader(
            train_ds,
            ContrastiveCollator(tok.vi, tok.vi_max_len),
            d,
            self.ctx.world_size,
            self.ctx.rank,
            cfg.run.seed,
            groups=groups,
        )

    def build_eval_loaders(self) -> dict[str, RetrievalLoaders]:
        """One :class:`RetrievalLoaders` pair per ``data.evals`` entry (public for ``scripts/evaluate.py``)."""
        cfg, d = self.cfg, self.cfg.data
        tok = self.tokenizers()
        spec, _train_tf, eval_tf = build_transform_for(cfg)
        self._transform_spec = spec.to_dict()
        draft = draft_size_for(cfg, spec)
        run_dir = self._run_dir()
        evals: dict[str, RetrievalLoaders] = {}
        self._eval_datasets.clear()
        for split, src in d.evals.items():
            records, adapter = build_records(src, None, run_dir, self.ctx.is_main)
            if records.record_type is not ImageTextRecord:
                raise ConfigError(
                    f"data.evals.{split}: contrastive evaluates image-text (retrieval) splits only, "
                    f"got {records.record_type.__name__} records"
                )
            info = adapter.holdout_info.get(src.split)
            self._dev_split_sha1[split] = info.sha1 if info is not None else None
            dataset = RetrievalEvalDataset(records, adapter.image_store(src.split), eval_tf, draft)
            self._eval_datasets[split] = dataset
            evals[split] = build_retrieval_loaders(
                dataset, text_collator(tok.vi, tok.vi_max_len), d, self.ctx
            )
            log.info(
                "retrieval split %s: %d images / %d captions",
                split,
                len(dataset.image_ids),
                len(dataset.captions),
            )
        return evals

    def build_loaders(self) -> LoaderBundle:
        return LoaderBundle(train=self.build_train_loader(), evals=self.build_eval_loaders())

    # ------------------------------------------------------------------ training
    def train_step(self, model: nn.Module, batch: dict[str, Any], sc: StepContext) -> StepOutput:
        out = model(batch, "contrastive")
        loss_c, stats = self.loss(out.image_embeds, out.text_embeds, out.logit_scale)
        metrics: dict[str, float] = {
            "loss_contrastive": float(loss_c.detach()),
            "acc_i2t": stats["acc_i2t"],
            "acc_t2i": stats["acc_t2i"],
            "logit_scale": float(out.logit_scale.detach()),
        }
        loss = loss_c
        raw = getattr(model, "module", model)
        encoder = encoder_of(raw)
        if encoder.is_moe:
            if self._router_loss is None:
                self._router_loss = _router_loss_computer(self.cfg)
            aux, logs = self._router_loss(out.router_outputs)
            loss = loss_c + aux
            metrics["loss_aux"] = float(aux.detach())
            metrics.update({k: float(v) for k, v in logs.items()})
            moe = self.cfg.model.moe
            if moe is not None and moe.aux_free_bias and moe.bias_update_every == "micro":
                metrics.update(_aux_free_bias_step(encoder, self.ctx, moe.bias_update_rate))
        return StepOutput(
            loss=loss,
            batch_size=len(batch["uids"]),
            num_tokens=int(batch["attention_mask"].sum().item()),
            metrics=metrics,
        )

    def post_optimizer_step(self, trainer: Trainer, state: TrainState) -> None:
        """Eq. 22 aux-free bias update once per optimizer step (all ranks, one all-reduce)."""
        moe = self.cfg.model.moe
        if moe is None or not moe.aux_free_bias or moe.bias_update_every != "step":
            return
        encoder = encoder_of(trainer.raw_model)
        if not encoder.is_moe:
            raise RuntimeError("model.moe is set but the encoder is dense")
        logs = _aux_free_bias_step(encoder, self.ctx, moe.bias_update_rate)
        if self.ctx.is_main and state.step % self.cfg.train.log_every == 0:
            trainer.writer.log_scalars(logs, state.step)

    # ------------------------------------------------------------------ evaluation
    def evaluators(self) -> dict[str, Evaluator]:
        if not self._eval_datasets and self.cfg.data.evals:
            raise RuntimeError("evaluators() must be called after build_eval_loaders()")
        if self._evaluators is not None:
            return self._evaluators
        out: dict[str, Evaluator] = {s: RetrievalEvaluator(self.ctx) for s in self._eval_datasets}
        if self.cfg.analysis.routing_probe is not None:
            try:
                from vivlmoe.tasks.evaluators.routing import maybe_add_routing_probe
            except ImportError as exc:
                raise ImportError(
                    "analysis.routing_probe requires vivlmoe.tasks.evaluators.routing (milestone M6)"
                ) from exc
            maybe_add_routing_probe(out, self.cfg, self.ctx, "contrastive")
        self._evaluators = out
        return out

    def selection_metric(self) -> tuple[str | None, str]:
        for split in self.cfg.data.evals:
            return f"{split}/rsum", "max"
        return None, "max"

    # ------------------------------------------------------------------ state
    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "dev_split_sha1": dict(self._dev_split_sha1),
            "transform_spec": self._transform_spec,
        }


__all__ = ["ContrastiveTask", "encoder_of"]
