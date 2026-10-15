from __future__ import annotations

import pytest

from vivlmoe.metrics.bleu import bleu4
from vivlmoe.metrics.tokenize import MetricTokenizer

T13 = MetricTokenizer("13a")

# Hand-computed corpus: precisions 1, 1, 6/8, 3/6 and BP 1 (12 vs 12 tokens).
PREDS = ["the cat sat on the mat", "a dog runs in the park"]
REFS = [["the cat sat on the mat"], ["a dog is running in the park", "the dog runs in a park"]]
EXPECTED = 0.375**0.25  # geometric mean of (1, 1, 0.75, 0.5)


def test_identical_hypothesis_scores_one() -> None:
    out = bleu4(["một con mèo đen ngồi trên thảm ."], [["một con mèo đen ngồi trên thảm ."]], T13)
    assert out["bleu4"] == pytest.approx(1.0)
    assert out["precisions"] == [1.0, 1.0, 1.0, 1.0]
    assert out["brevity_penalty"] == 1.0


def test_toy_corpus_hard_coded_value_and_variable_ref_counts() -> None:
    out = bleu4(PREDS, REFS, T13)
    assert out["bleu4"] == pytest.approx(EXPECTED, abs=1e-9)
    assert out["precisions"] == pytest.approx([1.0, 1.0, 0.75, 0.5])
    assert out["translation_length"] == 12 and out["reference_length"] == 12
    assert out["length_ratio"] == 1.0


def test_parity_with_reference_implementations() -> None:
    """Cross-check the vendored function against evaluate / sacrebleu when installed."""
    checked = False
    try:
        import evaluate  # type: ignore[import-not-found]
    except ImportError:
        evaluate = None
    if evaluate is not None:
        ref = evaluate.load("bleu").compute(predictions=PREDS, references=REFS)
        assert bleu4(PREDS, REFS, T13)["bleu4"] == pytest.approx(ref["bleu"], abs=1e-9)
        checked = True
    try:
        import sacrebleu  # type: ignore[import-not-found]
    except ImportError:
        sacrebleu = None
    if sacrebleu is not None:
        # sacrebleu wants ref streams of equal length: duplicate the single reference.
        streams = [[REFS[0][0], REFS[1][0]], [REFS[0][0], REFS[1][1]]]
        ref = sacrebleu.corpus_bleu(PREDS, streams, tokenize="13a", smooth_method="none")
        assert bleu4(PREDS, REFS, T13)["bleu4"] * 100 == pytest.approx(ref.score, abs=1e-6)
        checked = True
    if not checked:
        pytest.skip("neither evaluate nor sacrebleu installed; parity assertion skipped")


def test_empty_hypothesis_is_zero_without_exception() -> None:
    out = bleu4(["", "the cat"], [["the cat sat"], ["the cat"]], T13)
    assert 0.0 <= out["bleu4"] < 1.0
    out = bleu4([""], [["the cat sat"]], T13)
    assert out["bleu4"] == 0.0 and out["brevity_penalty"] == 0.0
    assert out["translation_length"] == 0 and out["reference_length"] == 3


def test_brevity_penalty_applies_to_short_hypotheses() -> None:
    out = bleu4(["the cat"], [["the cat sat on the mat"]], T13, smooth=True)
    assert out["brevity_penalty"] < 1.0
    assert out["length_ratio"] == pytest.approx(2 / 6)


def test_tokenizer_matters() -> None:
    hyp, refs = ["Hà Nội đẹp."], [["hà nội đẹp"]]
    assert bleu4(hyp, refs, MetricTokenizer("vi_safe"), smooth=True)["bleu4"] == pytest.approx(1.0)
    assert bleu4(hyp, refs, T13, smooth=True)["bleu4"] < 1.0


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="predictions vs"):
        bleu4(["a"], [], T13)
    with pytest.raises(ValueError, match="non-empty list"):
        bleu4(["a"], [[]], T13)
    with pytest.raises(ValueError, match="non-empty list"):
        bleu4(["a"], ["a"], T13)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="undefined"):
        bleu4(["a"], [[""]], T13)
