"""CIDEr-D via ``pycocoevalcap`` on pre-tokenised text (no Java PTB tokenizer needed).

Used for OpenViVQA (paper §5). ``pycocoevalcap.cider.cider.Cider`` splits its inputs on
whitespace, so we tokenize with :class:`~vivlmoe.metrics.tokenize.MetricTokenizer` and
join with single spaces before handing the strings over.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class _Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[str]: ...


def _load_cider() -> type:
    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError as exc:
        raise ImportError(
            "cider() needs 'pycocoevalcap' (pip install pycocoevalcap); "
            "the Java tokenizer is not required"
        ) from exc
    return Cider


def cider(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    tokenizer: _Tokenizer,
) -> dict[str, float]:
    """Corpus CIDEr-D (n=4, sigma=6) on the conventional 0-10 scale; returns ``{'cider': x}``.

    Note that the tf-idf weights are computed on the evaluated corpus itself, so a single
    sample (or n-grams present in every reference set) scores 0 by construction.
    """
    if len(predictions) != len(references):
        raise ValueError(f"{len(predictions)} predictions vs {len(references)} reference lists")
    if len(predictions) == 0:
        raise ValueError("cider needs at least one sample")
    cider_cls = _load_cider()
    gts: dict[int, list[str]] = {}
    res: dict[int, list[str]] = {}
    for i, (pred, refs) in enumerate(zip(predictions, references, strict=True)):
        if isinstance(refs, str) or len(refs) == 0:
            raise ValueError(f"sample {i}: references must be a non-empty list of strings")
        gts[i] = [" ".join(tokenizer.tokenize(r)) for r in refs]
        res[i] = [" ".join(tokenizer.tokenize(pred))]
    score, _ = cider_cls().compute_score(gts, res)
    return {"cider": float(score)}
