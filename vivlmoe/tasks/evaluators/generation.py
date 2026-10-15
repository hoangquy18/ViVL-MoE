"""Beam-search generation evaluator for captioning / VQA (paper Sec. 6.1, 6.3; plan 8.7).

Every rank decodes its ordered shard with ``model.generate`` (unwrapped model, ``eval()``,
``no_grad``, autocast on CUDA), decodes the ids with the decoder tokenizer and gathers
``(prediction, references, prompt, image_id)`` by ``uid`` (:func:`gather_ordered` is a collective
and is called even on empty shards). Rank 0 computes the requested metrics with
:func:`vivlmoe.metrics.compute_generation_metrics` (BLEU-4 multi-ref, ROUGE-L, CIDEr, token-F1,
EM; fractions in [0, 1] except CIDEr on its 0-10 scale) and dumps predictions + metrics.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from vivlmoe.config.schema import GenerationCfg, TaskCfg
from vivlmoe.data.loaders import move_to_device
from vivlmoe.metrics import METRICS, compute_generation_metrics, tokenizer_signatures
from vivlmoe.metrics.tokenize import MetricTokenizer
from vivlmoe.tasks.evaluators.base import Evaluator, gather_ordered
from vivlmoe.tasks.evaluators.dump import (
    metrics_path,
    predictions_path,
    write_metrics_json,
    write_predictions_jsonl,
)
from vivlmoe.training.distributed import DistContext
from vivlmoe.training.state import EvalContext

log = logging.getLogger(__name__)

GENERATION_METRICS = ("bleu4", "rougeL", "cider", "f1", "em")


def generation_kwargs(gen: GenerationCfg) -> dict[str, Any]:
    """``GenerationCfg`` -> HF ``generate`` keyword arguments (beam 5 in the paper)."""
    return {
        "num_beams": gen.num_beams,
        "max_new_tokens": gen.max_new_tokens,
        "min_new_tokens": gen.min_new_tokens,
        "length_penalty": gen.length_penalty,
        "early_stopping": gen.early_stopping,
        "no_repeat_ngram_size": gen.no_repeat_ngram_size,
        "repetition_penalty": gen.repetition_penalty,
        "do_sample": False,
    }


def special_token_kwargs(tokenizer: PreTrainedTokenizerBase) -> dict[str, int]:
    """``eos_token_id`` / ``pad_token_id`` of ``tokenizer`` for ``generate`` (pad falls back to EOS)."""
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError(f"{type(tokenizer).__name__} has no eos token")
    pad = tokenizer.pad_token_id
    return {"eos_token_id": int(eos), "pad_token_id": int(eos if pad is None else pad)}


def validate_metric_names(names: list[str], allowed: tuple[str, ...], where: str) -> list[str]:
    if not names:
        raise ValueError(f"{where}: task.metrics must list at least one of {list(allowed)}")
    bad = [n for n in names if n not in allowed or n not in METRICS]
    if bad:
        raise ValueError(f"{where}: unknown metrics {bad}; allowed: {list(allowed)}")
    return list(dict.fromkeys(names))


def _autocast(device: torch.device, dtype: torch.dtype | None) -> contextlib.AbstractContextManager:
    if device.type == "cuda" and dtype is not None:
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


class GenerationEvaluator(Evaluator):
    """``model.generate(batch, mode=..., **gen)`` -> decoded strings -> corpus metrics on rank 0."""

    name = "generation"

    def __init__(
        self,
        ctx: DistContext,
        task_cfg: TaskCfg,
        tokenizer: PreTrainedTokenizerBase,
        mode: str | None = None,
        amp_dtype: torch.dtype | None = None,
        allowed_metrics: tuple[str, ...] = GENERATION_METRICS,
        pass_special_ids: bool = False,
    ) -> None:
        self.ctx = ctx
        self.task_cfg = task_cfg
        self.tokenizer = tokenizer
        self.mode = mode
        self.amp_dtype = amp_dtype
        self.metric_names = validate_metric_names(
            task_cfg.metrics, allowed_metrics, f"GenerationEvaluator(mode={mode})"
        )
        self.gen_kwargs = generation_kwargs(task_cfg.generation)
        if pass_special_ids:  # causal-LM heads: stop on the EOS the answers were trained with
            self.gen_kwargs.update(special_token_kwargs(tokenizer))
        self.metric_tokenizer = MetricTokenizer(task_cfg.metric_tokenizer)
        self.bleu_tokenizer = MetricTokenizer(task_cfg.bleu_tokenizer)

    # ------------------------------------------------------------------ decoding
    def _decode(self, ids: torch.Tensor) -> list[str]:
        texts = self.tokenizer.batch_decode(ids.detach().cpu(), skip_special_tokens=True)
        return [t.strip() for t in texts]

    def _generate(self, model: nn.Module, batch: dict[str, Any], device: torch.device) -> list[str]:
        with _autocast(device, self.amp_dtype):
            ids = model.generate(batch, mode=self.mode, **self.gen_kwargs)
        return self._decode(ids)

    # ------------------------------------------------------------------ Evaluator API
    def run(self, model: nn.Module, loader: DataLoader, ec: EvalContext) -> dict[str, float] | None:
        device = ec.ctx.device
        uids: list[int] = []
        payload: list[tuple[str, list[str], str, Any]] = []
        for batch in loader:
            batch = move_to_device(batch, device)
            preds = self._generate(model, batch, device)
            refs = batch["refs"]
            prompts = batch.get("prompts") or [""] * len(preds)
            image_ids = batch.get("image_ids") or [None] * len(preds)
            if not (len(preds) == len(refs) == len(batch["uids"])):
                raise RuntimeError(
                    f"generate returned {len(preds)} sequences for a batch of {len(batch['uids'])}"
                )
            uids += [int(u) for u in batch["uids"]]
            payload += [
                (p, list(r), str(q), i)
                for p, r, q, i in zip(preds, refs, prompts, image_ids, strict=True)
            ]
        rows = gather_ordered(uids, payload, ec.ctx)  # collective: every rank, even empty shards
        if not ec.ctx.is_main:
            return None
        if not rows:
            raise RuntimeError(f"split {ec.split_name!r} produced no samples")
        preds = [r[0] for r in rows]
        refs = [r[1] for r in rows]
        metrics = compute_generation_metrics(
            self.metric_names,
            preds,
            refs,
            self.metric_tokenizer,
            bleu_tokenizer=self.bleu_tokenizer,
            report_paper_compat_rouge=self.task_cfg.report_paper_compat_rouge,
        )
        metrics["num_samples"] = float(len(rows))
        metrics["pred_len_mean"] = float(
            sum(len(self.metric_tokenizer.tokenize(p)) for p in preds) / len(preds)
        )
        if ec.save_predictions:
            self._dump(ec, rows, metrics)
        return metrics

    def _dump(
        self,
        ec: EvalContext,
        rows: list[tuple[str, list[str], str, Any]],
        metrics: dict[str, float],
    ) -> None:
        write_predictions_jsonl(
            predictions_path(ec.run_dir, ec.split_name, ec.step),
            (
                {
                    "uid": uid,
                    "image_id": image_id,
                    "prompt": prompt,
                    "prediction": pred,
                    "references": refs,
                }
                for uid, (pred, refs, prompt, image_id) in enumerate(rows)
            ),
        )
        write_metrics_json(
            metrics_path(ec.run_dir, ec.split_name, ec.step),
            {
                "split": ec.split_name,
                "step": ec.step,
                "mode": self.mode,
                "num_samples": len(rows),
                "metrics": metrics,
                "metric_names": self.metric_names,
                "scales": {
                    "bleu4/rougeL/f1/em": "fraction in [0, 1] (multiply by 100 for paper tables)",
                    "cider": "CIDEr-D on the 0-10 scale",
                },
                "tokenizer_signatures": tokenizer_signatures(
                    self.metric_tokenizer, self.bleu_tokenizer
                ),
                "generation": dataclasses.asdict(self.task_cfg.generation),
                "decoder_tokenizer": type(self.tokenizer).__name__,
            },
        )


__all__ = [
    "GENERATION_METRICS",
    "GenerationEvaluator",
    "generation_kwargs",
    "special_token_kwargs",
    "validate_metric_names",
]
