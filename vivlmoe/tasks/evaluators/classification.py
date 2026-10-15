"""ViMACSA aspect-sentiment evaluator (paper Sec. 6.1; legacy ``train_macsa.py`` protocol).

Per batch the unwrapped classifier returns ``logits [B, A, C]``; the argmax per (sample, aspect) and
the gold labels are gathered in ``uid`` order with :func:`gather_ordered_tensors` (a collective,
called on every rank even for empty shards). Rank 0 reports
:func:`vivlmoe.metrics.aspect_macro_f1` (macro-P/R/F1 over the 4 labels per aspect, then the mean
over aspects; ``exclude_none`` drops label 0) as ``macro_f1``, ``macro_p``, ``macro_r`` and
``f1/<aspect>``, and dumps per-aspect label names to ``predictions/<split>_step<N>.jsonl``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from vivlmoe.data.loaders import move_to_device
from vivlmoe.metrics import aspect_macro_f1
from vivlmoe.tasks.evaluators.base import Evaluator, gather_ordered_tensors
from vivlmoe.tasks.evaluators.dump import (
    metrics_path,
    predictions_path,
    write_metrics_json,
    write_predictions_jsonl,
)
from vivlmoe.training.distributed import DistContext
from vivlmoe.training.state import EvalContext

log = logging.getLogger(__name__)

LABEL_NAMES: tuple[str, ...] = ("None", "Negative", "Neutral", "Positive")


class ClassificationEvaluator(Evaluator):
    name = "classification"

    def __init__(
        self,
        ctx: DistContext,
        aspects: Sequence[str],
        exclude_none: bool = False,
        mode: str = "joint_cls",
        num_labels: int = len(LABEL_NAMES),
        label_names: Sequence[str] = LABEL_NAMES,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        if len(label_names) != num_labels:
            raise ValueError(f"{len(label_names)} label names for num_labels={num_labels}")
        self.ctx = ctx
        self.aspects = list(aspects)
        self.exclude_none = exclude_none
        self.mode = mode
        self.num_labels = num_labels
        self.label_names = list(label_names)
        self.amp_dtype = amp_dtype

    def _forward(self, model: nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
        if device.type == "cuda" and self.amp_dtype is not None:
            with torch.autocast("cuda", dtype=self.amp_dtype):
                out = model(batch, self.mode)
        else:
            out = model(batch, self.mode)
        logits = out.logits
        if logits.ndim != 3 or logits.shape[1] != len(self.aspects):
            raise RuntimeError(
                f"expected logits [B, A={len(self.aspects)}, C], got {tuple(logits.shape)}"
            )
        return logits

    def run(self, model: nn.Module, loader: DataLoader, ec: EvalContext) -> dict[str, float] | None:
        device = ec.ctx.device
        a = len(self.aspects)
        uids: list[int] = []
        preds: list[torch.Tensor] = []
        golds: list[torch.Tensor] = []
        for batch in loader:
            batch = move_to_device(batch, device)
            logits = self._forward(model, batch, device)
            preds.append(logits.argmax(-1).to(torch.int64).cpu())
            golds.append(batch["labels"].to(torch.int64).cpu())
            uids += [int(u) for u in batch["uids"]]
        ids = torch.tensor(uids, dtype=torch.int64)
        pred = torch.cat(preds) if preds else torch.zeros(0, a, dtype=torch.int64)
        gold = torch.cat(golds) if golds else torch.zeros(0, a, dtype=torch.int64)
        pred_all = gather_ordered_tensors(ids, pred, ec.ctx)  # collectives on every rank
        gold_all = gather_ordered_tensors(ids, gold, ec.ctx)
        if not ec.ctx.is_main:
            return None
        n = int(pred_all.shape[0])
        if n == 0:
            raise RuntimeError(f"split {ec.split_name!r} produced no samples")
        res = aspect_macro_f1(
            gold_all.numpy(),
            pred_all.numpy(),
            self.aspects,
            exclude_none=self.exclude_none,
            num_labels=self.num_labels,
        )
        per_aspect = res["per_aspect"]
        assert isinstance(per_aspect, dict)
        metrics: dict[str, float] = {
            "macro_f1": float(res["macro_f1"]),
            "macro_p": float(res["macro_p"]),
            "macro_r": float(res["macro_r"]),
            "accuracy": float((pred_all == gold_all).float().mean()),
            "num_samples": float(n),
        }
        for asp, d in per_aspect.items():
            metrics[f"f1/{asp}"] = float(d["f1"])
        if ec.save_predictions:
            self._dump(ec, pred_all, gold_all, metrics)
        return metrics

    def _dump(
        self, ec: EvalContext, pred: torch.Tensor, gold: torch.Tensor, metrics: dict[str, float]
    ) -> None:
        names = self.label_names

        def row(uid: int) -> dict:
            return {
                "uid": uid,
                "pred": {asp: names[int(pred[uid, j])] for j, asp in enumerate(self.aspects)},
                "gold": {asp: names[int(gold[uid, j])] for j, asp in enumerate(self.aspects)},
                "pred_ids": pred[uid].tolist(),
                "gold_ids": gold[uid].tolist(),
            }

        write_predictions_jsonl(
            predictions_path(ec.run_dir, ec.split_name, ec.step),
            (row(i) for i in range(int(pred.shape[0]))),
        )
        write_metrics_json(
            metrics_path(ec.run_dir, ec.split_name, ec.step),
            {
                "split": ec.split_name,
                "step": ec.step,
                "mode": self.mode,
                "num_samples": int(pred.shape[0]),
                "metrics": metrics,
                "aspects": self.aspects,
                "labels": names,
                "exclude_none_in_f1": self.exclude_none,
                "scales": {"macro_f1/macro_p/macro_r/accuracy": "fraction in [0, 1]"},
            },
        )


__all__ = ["LABEL_NAMES", "ClassificationEvaluator"]
