from __future__ import annotations

import sys

import pytest

from vivlmoe.metrics.tokenize import MetricTokenizer

VI = MetricTokenizer("vi_safe")
REFS = [
    ["con mèo ngồi trên thảm"],
    ["con chó chạy trong công viên"],
    ["cô gái đọc sách ở thư viện"],
]
REFS_MULTI = [REFS[0], REFS[1] + ["chú chó đang chạy ngoài công viên"], REFS[2]]


def test_perfect_match_scores_ten() -> None:
    pytest.importorskip("pycocoevalcap")
    from vivlmoe.metrics.cider import cider

    preds = [r[0] for r in REFS]
    assert cider(preds, REFS, VI)["cider"] == pytest.approx(10.0)


def test_tokenizer_is_applied() -> None:
    pytest.importorskip("pycocoevalcap")
    from vivlmoe.metrics.cider import cider

    preds = [
        "Con mèo ngồi trên thảm.",
        "Con chó chạy trong công viên!",
        "Cô gái đọc sách ở thư viện",
    ]
    assert cider(preds, REFS, VI)["cider"] == pytest.approx(10.0)
    # Without normalisation the capitalised / punctuated hypotheses no longer match.
    raw = MetricTokenizer("13a")
    assert cider(preds, REFS, raw)["cider"] < 10.0


def test_multi_reference_averages_over_references() -> None:
    pytest.importorskip("pycocoevalcap")
    from vivlmoe.metrics.cider import cider

    preds = [r[0] for r in REFS]
    multi = cider(preds, REFS_MULTI, VI)["cider"]
    assert 0.0 < multi < 10.0
    wrong = [REFS[0][0], "hoàn toàn sai", REFS[2][0]]
    assert cider(wrong, REFS_MULTI, VI)["cider"] < multi


def test_wrong_and_empty_predictions() -> None:
    pytest.importorskip("pycocoevalcap")
    from vivlmoe.metrics.cider import cider

    out = cider(["", "hoàn toàn sai", "cô gái đọc sách ở thư viện"], REFS, VI)
    assert 0.0 < out["cider"] < 10.0


def test_missing_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from vivlmoe.metrics.cider import cider

    monkeypatch.setitem(sys.modules, "pycocoevalcap.cider.cider", None)
    with pytest.raises(ImportError, match="pycocoevalcap"):
        cider(["a"], [["a"]], VI)
