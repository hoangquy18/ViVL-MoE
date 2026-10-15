"""Corpus BLEU-4 with multiple references (paper §5 downstream metrics).

``compute_bleu`` is vendored verbatim from HF ``evaluate`` (``metrics/bleu/nmt_bleu.py``),
which in turn is the tensor2tensor ``bleu_hook`` implementation, so the numbers match
``evaluate.load("bleu")`` given the same tokenizer (``MetricTokenizer("13a")`` = paper-compat).
"""

from __future__ import annotations

import collections
import math
from collections.abc import Sequence
from typing import Protocol


class _Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[str]: ...


# --------------------------------------------------------------------------------------
# Vendored from https://github.com/huggingface/evaluate (metrics/bleu/nmt_bleu.py), which
# is https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py.
# Copyright 2017 Google Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# --------------------------------------------------------------------------------------


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.

    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.

    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i : i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    """Computes BLEU score of translated segments against one or more references.

    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.

    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for references, translation in zip(reference_corpus, translation_corpus, strict=False):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = (matches_by_order[i] + 1.0) / (possible_matches_by_order[i] + 1.0)
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1.0 / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.0
    else:
        bp = math.exp(1 - 1.0 / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


# ---------------------------------- end of vendored code --------------------------------


def bleu4(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    tokenizer: _Tokenizer,
    smooth: bool = False,
    max_order: int = 4,
) -> dict[str, float | list[float] | int]:
    """Corpus BLEU (default BLEU-4) with a variable number of references per sample.

    Returns ``bleu4`` in [0, 1] plus ``precisions``, ``brevity_penalty``, ``length_ratio``,
    ``translation_length`` and ``reference_length`` (same fields as HF ``evaluate``).
    An all-empty hypothesis corpus scores 0 instead of raising.
    """
    if len(predictions) != len(references):
        raise ValueError(f"{len(predictions)} predictions vs {len(references)} reference lists")
    ref_tok: list[list[list[str]]] = []
    for i, refs in enumerate(references):
        if isinstance(refs, str) or len(refs) == 0:
            raise ValueError(f"sample {i}: references must be a non-empty list of strings")
        ref_tok.append([tokenizer.tokenize(r) for r in refs])
    hyp_tok = [tokenizer.tokenize(p) for p in predictions]

    reference_length = sum(min(len(r) for r in refs) for refs in ref_tok)
    if reference_length == 0:
        raise ValueError("BLEU is undefined: every sample has an empty shortest reference")
    if sum(len(h) for h in hyp_tok) == 0:
        return {
            "bleu4": 0.0,
            "precisions": [0.0] * max_order,
            "brevity_penalty": 0.0,
            "length_ratio": 0.0,
            "translation_length": 0,
            "reference_length": reference_length,
        }
    bleu, precisions, bp, ratio, translation_length, reference_length = compute_bleu(
        ref_tok, hyp_tok, max_order=max_order, smooth=smooth
    )
    return {
        "bleu4": float(bleu),
        "precisions": [float(p) for p in precisions],
        "brevity_penalty": float(bp),
        "length_ratio": float(ratio),
        "translation_length": int(translation_length),
        "reference_length": int(reference_length),
    }
