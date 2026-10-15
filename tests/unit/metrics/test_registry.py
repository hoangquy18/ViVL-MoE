from __future__ import annotations

import pytest

from vivlmoe.metrics import (
    METRICS,
    PAPER_COMPAT_ROUGE_SEED,
    MetricTokenizer,
    compute_generation_metrics,
    rouge_l,
    tokenizer_signatures,
)

PREDS = ["Một con mèo đen.", "Hà Nội"]
REFS = [["một con mèo trắng", "một con mèo đen"], ["hà nội", "thủ đô hà nội"]]


def test_registry_keys() -> None:
    assert METRICS.keys() == ["bleu4", "cider", "em", "f1", "rougeL"]
    with pytest.raises(KeyError, match="unknown key"):
        METRICS.get("meteor")


def test_compute_generation_metrics_flat_dict() -> None:
    pytest.importorskip("rouge_score")
    vi = MetricTokenizer("vi_safe")
    out = compute_generation_metrics(["bleu4", "rougeL", "f1", "em"], PREDS, REFS, vi)
    assert {
        "bleu4",
        "bleu4_bp",
        "bleu4_len_ratio",
        "rougeL",
        "rougeL_p",
        "rougeL_r",
        "rougeL_paper_compat",
        "f1",
        "em",
    } == set(out)
    assert all(isinstance(v, float) for v in out.values())
    assert out["bleu4"] == pytest.approx(1.0)
    assert out["rougeL"] == pytest.approx(1.0) and out["f1"] == 1.0 and out["em"] == 1.0
    # paper-compat ROUGE strips diacritics; still 1.0 here because both sides are stripped alike
    assert out["rougeL_paper_compat"] == pytest.approx(1.0)

    no_compat = compute_generation_metrics(
        ["rougeL"], PREDS, REFS, vi, report_paper_compat_rouge=False
    )
    assert "rougeL_paper_compat" not in no_compat


def test_bleu_tokenizer_override() -> None:
    vi = MetricTokenizer("vi_safe")
    t13 = MetricTokenizer("13a")
    with_vi = compute_generation_metrics(["bleu4"], PREDS, REFS, vi)
    with_13a = compute_generation_metrics(["bleu4"], PREDS, REFS, vi, bleu_tokenizer=t13)
    assert with_vi["bleu4"] == pytest.approx(1.0)
    assert with_13a["bleu4"] < 1.0  # case-sensitive + trailing '.' token


def test_paper_compat_rouge_differs_on_vietnamese() -> None:
    pytest.importorskip("rouge_score")
    out = compute_generation_metrics(
        ["rougeL"], ["một con mèo đen"], [["một con mèo trắng"]], MetricTokenizer("vi_safe")
    )
    assert out["rougeL"] == pytest.approx(0.75)
    assert out["rougeL_paper_compat"] == pytest.approx(10 / 13)


def test_paper_compat_rouge_uses_bootstrap_aggregator() -> None:
    """``rougeL_paper_compat`` mirrors HF evaluate's rouge (bootstrap mid), not the plain mean."""
    pytest.importorskip("rouge_score")
    preds = ["một con mèo đen", "x y", "a b c d", "p q r", "k"]
    refs = [["một con mèo trắng"], ["x y z"], ["a b"], ["p r"], ["k l m n"]]
    compat_tok = MetricTokenizer("rouge_default")
    out = compute_generation_metrics(["rougeL"], preds, refs, MetricTokenizer("vi_safe"))
    mean = rouge_l(preds, refs, compat_tok)["rougeL"]
    boot = rouge_l(preds, refs, compat_tok, use_aggregator=True, seed=PAPER_COMPAT_ROUGE_SEED)
    assert out["rougeL_paper_compat"] == pytest.approx(boot["rougeL"])
    assert out["rougeL_paper_compat"] != pytest.approx(mean)
    # deterministic across calls
    again = compute_generation_metrics(["rougeL"], preds, refs, MetricTokenizer("vi_safe"))
    assert again["rougeL_paper_compat"] == out["rougeL_paper_compat"]


def test_tokenizer_signatures() -> None:
    sig = tokenizer_signatures(MetricTokenizer("vi_safe"), MetricTokenizer("13a"))
    assert sig["tokenizer"].startswith("vi_safe(") and sig["bleu_tokenizer"].startswith("13a(")
    assert sig["rougeL_paper_compat_tokenizer"] == "rouge_default"
    assert sig["rougeL_paper_compat_aggregator"] == "bootstrap(seed=0)"
    assert sig["rougeL_aggregator"] == "mean"
    assert tokenizer_signatures(MetricTokenizer("vi_safe"))["bleu_tokenizer"].startswith("vi_safe(")
