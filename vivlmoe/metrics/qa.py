"""SQuAD-style token F1 and exact match for short-answer VQA (ViTextVQA / ViOCR-VQA).

Follows the official SQuAD v1.1 evaluation: token-level F1 from the multiset intersection
of prediction and gold tokens, maximised over gold answers, with normalisation delegated
to the :class:`~vivlmoe.metrics.tokenize.MetricTokenizer` (``vi_safe`` keeps diacritics).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Protocol


class _Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[str]: ...


def token_f1(pred_tokens: Sequence[str], gold_tokens: Sequence[str]) -> float:
    """F1 between two token sequences (SQuAD ``f1_score`` on pre-tokenised input)."""
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        # SQuAD convention: F1 is 1 only if both are empty.
        return float(len(pred_tokens) == len(gold_tokens))
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_em(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    tokenizer: _Tokenizer,
    strict_em: bool = False,
) -> dict[str, float]:
    """Mean token F1 and exact match over the corpus, each maximised over the gold answers.

    EM compares normalised token sequences; ``strict_em=True`` compares the raw strings.
    Returns ``{'f1': x, 'em': y}`` in [0, 1].
    """
    if len(predictions) != len(references):
        raise ValueError(f"{len(predictions)} predictions vs {len(references)} reference lists")
    if len(predictions) == 0:
        raise ValueError("f1_em needs at least one sample")
    f1_sum = 0.0
    em_sum = 0.0
    for i, (pred, refs) in enumerate(zip(predictions, references, strict=True)):
        if isinstance(refs, str) or len(refs) == 0:
            raise ValueError(f"sample {i}: references must be a non-empty list of strings")
        pred_tokens = tokenizer.tokenize(pred)
        gold_token_lists = [tokenizer.tokenize(r) for r in refs]
        f1_sum += max(token_f1(pred_tokens, g) for g in gold_token_lists)
        if strict_em:
            em_sum += float(any(pred == r for r in refs))
        else:
            em_sum += float(any(pred_tokens == g for g in gold_token_lists))
    n = len(predictions)
    return {"f1": f1_sum / n, "em": em_sum / n}
