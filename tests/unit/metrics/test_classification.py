from __future__ import annotations

import numpy as np
import pytest

from vivlmoe.metrics.classification import aspect_macro_f1, macro_prf

ASPECTS = ["Location", "Food", "Room", "Facilities", "Service", "Public_area"]


def test_macro_prf_hand_case() -> None:
    y_true = [0, 0, 1, 1, 2]
    y_pred = [0, 1, 1, 1, 3]
    # label 0: tp1 fp0 fn1 -> p1 r.5 f2/3 ; label 1: tp2 fp1 fn0 -> p2/3 r1 f.8
    # label 2: tp0 fp0 fn1 -> 0 ; label 3: tp0 fp1 fn0 -> 0
    p, r, f = macro_prf(y_true, y_pred, labels=[0, 1, 2, 3])
    assert p == pytest.approx((1 + 2 / 3) / 4)
    assert r == pytest.approx((0.5 + 1) / 4)
    assert f == pytest.approx((2 / 3 + 0.8) / 4)


def test_macro_prf_matches_sklearn_on_random_data() -> None:
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, size=(200, 6))
    y_pred = rng.integers(0, 4, size=(200, 6))
    for a in range(6):
        ours = macro_prf(y_true[:, a], y_pred[:, a], labels=[0, 1, 2, 3])
        p, r, f, _ = skm.precision_recall_fscore_support(
            y_true[:, a], y_pred[:, a], average="macro", labels=[0, 1, 2, 3], zero_division=0
        )
        assert ours == pytest.approx((p, r, f))
    # Sparse predictions so some labels are never predicted (zero_division path).
    y_pred2 = np.where(rng.random((200, 6)) < 0.9, 0, y_pred)
    for a in range(6):
        ours = macro_prf(y_true[:, a], y_pred2[:, a], labels=[0, 1, 2, 3])
        p, r, f, _ = skm.precision_recall_fscore_support(
            y_true[:, a], y_pred2[:, a], average="macro", labels=[0, 1, 2, 3], zero_division=0
        )
        assert ours == pytest.approx((p, r, f))


def test_aspect_macro_f1_is_mean_over_aspects_and_matches_sklearn() -> None:
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 4, size=(200, 6))
    y_pred = rng.integers(0, 4, size=(200, 6))
    out = aspect_macro_f1(y_true, y_pred, ASPECTS)
    per = out["per_aspect"]
    assert isinstance(per, dict) and list(per) == ASPECTS
    expected = []
    for a, name in enumerate(ASPECTS):
        _, _, f, _ = skm.precision_recall_fscore_support(
            y_true[:, a], y_pred[:, a], average="macro", labels=[0, 1, 2, 3], zero_division=0
        )
        assert per[name]["f1"] == pytest.approx(f)
        expected.append(f)
    assert out["macro_f1"] == pytest.approx(np.mean(expected))
    assert out["macro_p"] == pytest.approx(np.mean([v["p"] for v in per.values()]))

    excl = aspect_macro_f1(y_true, y_pred, ASPECTS, exclude_none=True)
    _, _, f_excl, _ = skm.precision_recall_fscore_support(
        y_true[:, 0], y_pred[:, 0], average="macro", labels=[1, 2, 3], zero_division=0
    )
    assert excl["per_aspect"]["Location"]["f1"] == pytest.approx(f_excl)
    assert excl["macro_f1"] != pytest.approx(out["macro_f1"])


def test_perfect_prediction_and_shape_errors() -> None:
    y = np.array([[0, 1, 2, 3, 0, 1], [3, 2, 1, 0, 3, 2], [1, 3, 0, 2, 1, 3], [2, 0, 3, 1, 2, 0]])
    assert aspect_macro_f1(y, y, ASPECTS)["macro_f1"] == pytest.approx(1.0)
    # A label absent from both y_true and y_pred scores 0 (sklearn zero_division=0 semantics).
    assert aspect_macro_f1(y[:2], y[:2], ASPECTS)["macro_f1"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        aspect_macro_f1(y, y[:, :5], ASPECTS)
    with pytest.raises(ValueError):
        aspect_macro_f1(y, y, ASPECTS[:3])
    with pytest.raises(ValueError):
        macro_prf([0, 1], [0], labels=[0, 1])
