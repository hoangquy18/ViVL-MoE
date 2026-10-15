from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rouge_score")

from vivlmoe.metrics.rouge import rouge_l  # noqa: E402
from vivlmoe.metrics.tokenize import MetricTokenizer  # noqa: E402

VI = MetricTokenizer("vi_safe")
DEFAULT = MetricTokenizer("rouge_default")


def test_vietnamese_pair_vi_safe_vs_rouge_default() -> None:
    preds, refs = ["một con mèo đen"], [["một con mèo trắng"]]
    good = rouge_l(preds, refs, VI)
    assert good == {
        "rougeL": pytest.approx(0.75),
        "rougeL_p": pytest.approx(0.75),
        "rougeL_r": pytest.approx(0.75),
    }
    # rouge_score's default tokenizer strips the diacritics: 'm t con m o en' vs 'm t con m o tr ng'.
    bad = rouge_l(preds, refs, DEFAULT)
    assert bad["rougeL"] == pytest.approx(10 / 13)
    assert bad["rougeL"] != pytest.approx(good["rougeL"])


def test_multi_reference_takes_max() -> None:
    preds = ["một con mèo đen"]
    refs = [["hoàn toàn khác biệt", "một con mèo đen"]]
    assert rouge_l(preds, refs, VI)["rougeL"] == pytest.approx(1.0)
    assert rouge_l(preds, [["một con mèo đen", "hoàn toàn khác biệt"]], VI)[
        "rougeL"
    ] == pytest.approx(1.0)


def test_corpus_mean_and_empty_prediction() -> None:
    out = rouge_l(["một con mèo đen", ""], [["một con mèo trắng"], ["x y z"]], VI)
    assert out["rougeL"] == pytest.approx(0.75 / 2)


def test_bootstrap_aggregator_is_deterministic_and_restores_rng() -> None:
    preds = ["một con mèo đen", "x y", "a b c d"]
    refs = [["một con mèo trắng"], ["x y z"], ["a b"]]
    np.random.seed(123)
    before = np.random.get_state()
    a = rouge_l(preds, refs, VI, use_aggregator=True)
    after = np.random.get_state()
    assert all(
        np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
        for x, y in zip(before, after, strict=True)
    )
    b = rouge_l(preds, refs, VI, use_aggregator=True)
    assert a == b
    assert 0.0 <= a["rougeL"] <= 1.0


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        rouge_l([], [], VI)
    with pytest.raises(ValueError):
        rouge_l(["a"], [[]], VI)
