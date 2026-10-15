from __future__ import annotations

import sys
import unicodedata

import pytest

from vivlmoe.config.schema import MetricTokenizerCfg
from vivlmoe.metrics.tokenize import MetricTokenizer

TEXT = "Hà Nội, Việt-Nam! 3.5 km (đẹp)."


def test_13a_is_case_sensitive_and_splits_punctuation() -> None:
    tok = MetricTokenizer("13a")
    assert tok.tokenize(TEXT) == [
        "Hà",
        "Nội",
        ",",
        "Việt-Nam",
        "!",
        "3.5",
        "km",
        "(",
        "đẹp",
        ")",
        ".",
    ]
    # Same behaviour as sacrebleu/evaluate: "word." -> "word ." but "3.5" stays intact.
    assert tok.tokenize("hello world.") == ["hello", "world", "."]
    assert tok.tokenize("a&quot;b &amp; c") == ["a", '"', "b", "&", "c"]
    assert tok.tokenize("Cat") != tok.tokenize("cat")
    assert tok.normalize("hello  world.") == "hello world ."


def test_13a_does_not_apply_nfc() -> None:
    nfd = unicodedata.normalize("NFD", "mèo")
    assert MetricTokenizer("13a").tokenize(nfd) == [nfd]
    assert MetricTokenizer("vi_safe").tokenize(nfd) == [unicodedata.normalize("NFC", "mèo")]


def test_vi_safe_lowercases_and_strips_punctuation_keeping_letters() -> None:
    tok = MetricTokenizer("vi_safe")
    assert tok.tokenize(TEXT) == ["hà", "nội", "việt", "nam", "3", "5", "km", "đẹp"]
    assert tok.tokenize("Cat") == tok.tokenize("cat")
    assert tok.tokenize("") == []
    assert tok.tokenize("  ...  ") == []
    assert tok.normalize("Hà Nội.") == "hà nội"


def test_vi_safe_keeps_every_unicode_letter_unlike_rouge_default() -> None:
    vi = "một con mèo đen"
    assert MetricTokenizer("vi_safe").tokenize(vi) == ["một", "con", "mèo", "đen"]
    # rouge_score's default tokenizer deletes accented letters (the legacy bug).
    assert MetricTokenizer("rouge_default").tokenize(vi) == ["m", "t", "con", "m", "o", "en"]


def test_rouge_default_matches_rouge_score_library() -> None:
    rouge_tokenize = pytest.importorskip("rouge_score.tokenize")
    for text in [TEXT, "Hello, World! 42 apples", "một con mèo đen"]:
        assert MetricTokenizer("rouge_default").tokenize(text) == rouge_tokenize.tokenize(
            text, None
        )


def test_cfg_flags_are_honoured() -> None:
    tok = MetricTokenizer(MetricTokenizerCfg(mode="vi_safe", lowercase=False, strip_punct=False))
    assert tok.tokenize("Hà Nội.") == ["Hà", "Nội."]
    tok13 = MetricTokenizer(MetricTokenizerCfg(mode="13a", lowercase=True, strip_punct=True))
    assert tok13.tokenize("Hà Nội.") == ["hà", "nội"]
    assert "lowercase=True" in tok13.signature
    assert MetricTokenizer("rouge_default").signature == "rouge_default"


def test_rouge_score_compatible_interface() -> None:
    rouge_scorer = pytest.importorskip("rouge_score.rouge_scorer")
    scorer = rouge_scorer.RougeScorer(
        ["rougeL"], use_stemmer=False, tokenizer=MetricTokenizer("vi_safe")
    )
    assert scorer.score("một con mèo", "một con mèo")["rougeL"].fmeasure == 1.0


def test_unknown_mode_and_non_string_raise() -> None:
    with pytest.raises(ValueError, match="unknown metric tokenizer mode"):
        MetricTokenizer("nope")
    with pytest.raises(TypeError):
        MetricTokenizer("vi_safe").tokenize(None)  # type: ignore[arg-type]


def test_underthesea_missing_gives_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "underthesea", None)
    with pytest.raises(ImportError, match="underthesea"):
        MetricTokenizer("underthesea")


def test_underthesea_joins_multiword_tokens() -> None:
    pytest.importorskip("underthesea")
    tok = MetricTokenizer("underthesea")
    tokens = tok.tokenize("Hà Nội là thủ đô của Việt Nam.")
    assert "hà_nội" in tokens
    assert all(" " not in t and "." not in t for t in tokens)
