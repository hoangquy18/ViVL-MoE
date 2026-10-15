"""Macro precision/recall/F1 for ViMACSA (paper §5; legacy ``train_macsa.py:771-775``).

``macro_prf`` reproduces ``sklearn.metrics.precision_recall_fscore_support(average='macro',
labels=labels, zero_division=0)`` in plain numpy. ViMACSA reports, per aspect, macro-F1
over the four labels ``{0: None, 1: Negative, 2: Neutral, 3: Positive}`` and then the mean
over aspects; ``exclude_none`` drops label 0 from the macro average (plan §14).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def macro_prf(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    labels: Sequence[int],
) -> tuple[float, float, float]:
    """Macro-averaged (precision, recall, F1) over ``labels``; undefined ratios count as 0."""
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    if yt.shape != yp.shape:
        raise ValueError(f"y_true {yt.shape} and y_pred {yp.shape} differ in shape")
    if len(labels) == 0:
        raise ValueError("labels must not be empty")
    precisions, recalls, f1s = [], [], []
    for lab in labels:
        tp = int(np.sum((yt == lab) & (yp == lab)))
        fp = int(np.sum((yt != lab) & (yp == lab)))
        fn = int(np.sum((yt == lab) & (yp != lab)))
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    n = len(labels)
    return float(sum(precisions) / n), float(sum(recalls) / n), float(sum(f1s) / n)


def aspect_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    aspects: Sequence[str],
    exclude_none: bool = False,
    num_labels: int = 4,
    none_label: int = 0,
) -> dict[str, float | dict[str, dict[str, float]]]:
    """ViMACSA protocol: macro-P/R/F1 per aspect over all labels, then mean over aspects.

    ``y_true`` / ``y_pred`` are ``[N, A]`` integer arrays with ``A == len(aspects)``.
    Returns ``{'macro_f1', 'macro_p', 'macro_r', 'per_aspect': {aspect: {'f1','p','r'}}}``.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim != 2 or yt.shape != yp.shape:
        raise ValueError(f"expected [N, A] arrays of equal shape, got {yt.shape} and {yp.shape}")
    if yt.shape[1] != len(aspects):
        raise ValueError(
            f"y has {yt.shape[1]} aspect columns but {len(aspects)} aspects were given"
        )
    labels = [lab for lab in range(num_labels) if not (exclude_none and lab == none_label)]
    per_aspect: dict[str, dict[str, float]] = {}
    for a, name in enumerate(aspects):
        p, r, f = macro_prf(yt[:, a], yp[:, a], labels)
        per_aspect[name] = {"f1": f, "p": p, "r": r}
    n = len(aspects)
    return {
        "macro_f1": float(sum(v["f1"] for v in per_aspect.values()) / n),
        "macro_p": float(sum(v["p"] for v in per_aspect.values()) / n),
        "macro_r": float(sum(v["r"] for v in per_aspect.values()) / n),
        "per_aspect": per_aspect,
    }
