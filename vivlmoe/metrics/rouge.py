"""ROUGE-L via ``rouge_score`` with an explicit, Vietnamese-safe tokenizer (plan decision 17).

The legacy code used ``rouge_score``'s default tokenizer, whose ``[^a-z0-9]+`` rule deletes
every accented Vietnamese letter; here the tokenizer is always passed explicitly. Multiple
references are handled with ``RougeScorer.score_multi`` (max F-measure over references).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class _Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[str]: ...


def rouge_l(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    tokenizer: _Tokenizer,
    use_aggregator: bool = False,
    seed: int = 0,
) -> dict[str, float]:
    """Corpus ROUGE-L (sentence-level LCS, no stemming).

    ``use_aggregator=False`` (default) averages per-sample scores; ``True`` reports the
    bootstrap mid estimate from ``rouge_score.scoring.BootstrapAggregator`` with a fixed
    numpy seed (the global RNG state is restored afterwards).
    Returns ``rougeL`` (F), ``rougeL_p`` and ``rougeL_r`` in [0, 1].
    """
    from rouge_score import rouge_scorer, scoring

    if len(predictions) != len(references):
        raise ValueError(f"{len(predictions)} predictions vs {len(references)} reference lists")
    if len(predictions) == 0:
        raise ValueError("rouge_l needs at least one sample")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=tokenizer)
    scores = []
    for i, (pred, refs) in enumerate(zip(predictions, references, strict=True)):
        if isinstance(refs, str) or len(refs) == 0:
            raise ValueError(f"sample {i}: references must be a non-empty list of strings")
        scores.append(scorer.score_multi(list(refs), pred)["rougeL"])

    if use_aggregator:
        aggregator = scoring.BootstrapAggregator()
        for s in scores:
            aggregator.add_scores({"rougeL": s})
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            mid = aggregator.aggregate()["rougeL"].mid
        finally:
            np.random.set_state(state)
        return {
            "rougeL": float(mid.fmeasure),
            "rougeL_p": float(mid.precision),
            "rougeL_r": float(mid.recall),
        }
    n = len(scores)
    return {
        "rougeL": float(sum(s.fmeasure for s in scores) / n),
        "rougeL_p": float(sum(s.precision for s in scores) / n),
        "rougeL_r": float(sum(s.recall for s in scores) / n),
    }
