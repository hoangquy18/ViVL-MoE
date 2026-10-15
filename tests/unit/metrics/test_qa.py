from __future__ import annotations

import pytest

from vivlmoe.metrics.qa import f1_em, token_f1
from vivlmoe.metrics.tokenize import MetricTokenizer

VI = MetricTokenizer("vi_safe")


def test_normalised_exact_match() -> None:
    assert f1_em(["Hà Nội."], [["hà nội"]], VI) == {"f1": 1.0, "em": 1.0}


def test_partial_overlap() -> None:
    assert f1_em(["a b c"], [["a b"]], VI) == {"f1": pytest.approx(0.8), "em": 0.0}
    assert f1_em(["màu xanh lá"], [["màu xanh"]], VI)["f1"] == pytest.approx(0.8)


def test_multi_answer_max() -> None:
    out = f1_em(["màu xanh"], [["màu đỏ", "màu xanh lá", "màu xanh"]], VI)
    assert out == {"f1": 1.0, "em": 1.0}
    out = f1_em(["màu xanh"], [["màu đỏ", "màu xanh lá"]], VI)
    assert out["f1"] == pytest.approx(0.8) and out["em"] == 0.0


def test_strict_em_compares_raw_strings() -> None:
    preds, refs = ["Hà Nội."], [["hà nội"]]
    assert f1_em(preds, refs, VI, strict_em=True) == {"f1": 1.0, "em": 0.0}
    assert f1_em(["hà nội"], refs, VI, strict_em=True)["em"] == 1.0


def test_corpus_mean_and_no_overlap() -> None:
    out = f1_em(["hà nội", "sài gòn"], [["hà nội"], ["huế"]], VI)
    assert out == {"f1": 0.5, "em": 0.5}


def test_empty_edge_cases() -> None:
    assert token_f1([], []) == 1.0
    assert token_f1(["a"], []) == 0.0
    assert token_f1([], ["a"]) == 0.0
    assert f1_em([""], [["hà nội"]], VI) == {"f1": 0.0, "em": 0.0}


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        f1_em([], [], VI)
    with pytest.raises(ValueError):
        f1_em(["a"], [[]], VI)
