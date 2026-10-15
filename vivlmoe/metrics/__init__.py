"""Offline evaluation metrics (plan §6 decision 17, §8.8).

``METRICS`` maps a metric name to ``callable(predictions, references, tokenizer) -> dict``;
:func:`compute_generation_metrics` evaluates a list of names into one flat dict, using a
separate tokenizer for BLEU (paper-compat ``13a`` by default) and optionally adding the
paper-compatible ``rougeL_paper_compat`` computed the way HF ``evaluate``'s ``rouge`` does:
``rouge_score``'s default tokenizer and the bootstrap-aggregator mid estimate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from vivlmoe.config.registry import Registry
from vivlmoe.metrics.bleu import bleu4
from vivlmoe.metrics.cider import cider
from vivlmoe.metrics.classification import aspect_macro_f1, macro_prf
from vivlmoe.metrics.qa import f1_em
from vivlmoe.metrics.retrieval import recall_at_k
from vivlmoe.metrics.rouge import rouge_l
from vivlmoe.metrics.tokenize import MetricTokenizer

MetricFn = Callable[[Sequence[str], Sequence[Sequence[str]], MetricTokenizer], dict[str, float]]

METRICS: Registry[MetricFn] = Registry("metrics")

PAPER_COMPAT_ROUGE_SEED = 0
"""numpy seed for the bootstrap aggregator behind ``rougeL_paper_compat``."""


@METRICS.register("bleu4")
def _bleu4(
    preds: Sequence[str], refs: Sequence[Sequence[str]], tok: MetricTokenizer
) -> dict[str, float]:
    out = bleu4(preds, refs, tok)
    return {
        "bleu4": float(out["bleu4"]),
        "bleu4_bp": float(out["brevity_penalty"]),
        "bleu4_len_ratio": float(out["length_ratio"]),
    }


@METRICS.register("rougeL")
def _rouge_l(
    preds: Sequence[str], refs: Sequence[Sequence[str]], tok: MetricTokenizer
) -> dict[str, float]:
    return rouge_l(preds, refs, tok)


@METRICS.register("cider")
def _cider(
    preds: Sequence[str], refs: Sequence[Sequence[str]], tok: MetricTokenizer
) -> dict[str, float]:
    return cider(preds, refs, tok)


@METRICS.register("f1")
def _f1(
    preds: Sequence[str], refs: Sequence[Sequence[str]], tok: MetricTokenizer
) -> dict[str, float]:
    return {"f1": f1_em(preds, refs, tok)["f1"]}


@METRICS.register("em")
def _em(
    preds: Sequence[str], refs: Sequence[Sequence[str]], tok: MetricTokenizer
) -> dict[str, float]:
    return {"em": f1_em(preds, refs, tok)["em"]}


def compute_generation_metrics(
    names: Sequence[str],
    preds: Sequence[str],
    refs: Sequence[Sequence[str]],
    tokenizer: MetricTokenizer,
    bleu_tokenizer: MetricTokenizer | None = None,
    report_paper_compat_rouge: bool = True,
) -> dict[str, float]:
    """Evaluate ``names`` (subset of ``METRICS``) and return one flat ``{metric: float}`` dict.

    BLEU uses ``bleu_tokenizer`` when given (``TaskCfg.bleu_tokenizer``, paper-compat ``13a``),
    every other metric uses ``tokenizer``. When ``rougeL`` is requested and
    ``report_paper_compat_rouge`` is set, ``rougeL_paper_compat`` is added for comparison
    with the paper's numbers: it mirrors HF ``evaluate``'s ``rouge`` defaults, i.e.
    rouge_score's default (diacritic-stripping) tokenizer *and* the bootstrap-aggregator
    mid estimate (``use_aggregator=True``, seeded with ``PAPER_COMPAT_ROUGE_SEED``), whereas
    ``rougeL`` itself is the plain per-sample mean.
    """
    out: dict[str, float] = {}
    for name in names:
        fn = METRICS.get(name)
        tok = bleu_tokenizer if (name == "bleu4" and bleu_tokenizer is not None) else tokenizer
        out.update(fn(preds, refs, tok))
        if name == "rougeL" and report_paper_compat_rouge:
            compat = rouge_l(
                preds,
                refs,
                MetricTokenizer("rouge_default"),
                use_aggregator=True,
                seed=PAPER_COMPAT_ROUGE_SEED,
            )
            out["rougeL_paper_compat"] = compat["rougeL"]
    return out


def tokenizer_signatures(
    tokenizer: MetricTokenizer, bleu_tokenizer: MetricTokenizer | None = None
) -> dict[str, Any]:
    """Description of the tokenizers used, for ``metrics.json`` (plan decision 17)."""
    return {
        "tokenizer": tokenizer.signature,
        "bleu_tokenizer": (bleu_tokenizer or tokenizer).signature,
        "rougeL_paper_compat_tokenizer": "rouge_default",
        "rougeL_paper_compat_aggregator": f"bootstrap(seed={PAPER_COMPAT_ROUGE_SEED})",
        "rougeL_aggregator": "mean",
    }


__all__ = [
    "METRICS",
    "PAPER_COMPAT_ROUGE_SEED",
    "MetricFn",
    "MetricTokenizer",
    "aspect_macro_f1",
    "bleu4",
    "cider",
    "compute_generation_metrics",
    "f1_em",
    "macro_prf",
    "recall_at_k",
    "rouge_l",
    "tokenizer_signatures",
]
