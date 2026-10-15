"""Metric tokenizers (plan §6 decision 17, §8.8).

Four modes, selected by ``MetricTokenizerCfg.mode`` or a bare mode string:

* ``13a`` – the WMT/sacreBLEU tokenizer as vendored in HF ``evaluate``'s BLEU metric:
  no Unicode normalisation, case-sensitive, punctuation split into separate tokens.
  This is the paper-compatible BLEU tokenizer.
* ``vi_safe`` – NFC -> lowercase -> punctuation/symbols to spaces -> whitespace split.
  Keeps every Unicode letter, so Vietnamese diacritics survive (unlike ``rouge_default``).
* ``underthesea`` – ``vi_safe`` normalisation followed by ``underthesea.word_tokenize``;
  multi-syllable words are joined with ``_`` (optional dependency).
* ``rouge_default`` – bit-for-bit reproduction of ``rouge_score``'s default tokenizer
  (lowercase, ``[^a-z0-9]+`` -> space). It *deletes* Vietnamese letters and is kept only
  to report the paper-compatible ROUGE-L number next to the correct one.

The object exposes ``tokenize`` (so it can be passed as ``rouge_score``'s ``tokenizer=``)
and ``normalize`` (the space-joined token sequence, used for exact match).
"""

from __future__ import annotations

import importlib
import logging
import re
import unicodedata
from collections.abc import Callable
from typing import Literal

from vivlmoe.config.schema import MetricTokenizerCfg

logger = logging.getLogger(__name__)

TokenizerMode = Literal["13a", "vi_safe", "underthesea", "rouge_default"]

# mode -> (nfc, lowercase, strip_punct) used when the tokenizer is built from a bare string.
_PRESETS: dict[str, tuple[bool, bool, bool]] = {
    "13a": (False, False, False),
    "vi_safe": (True, True, True),
    "underthesea": (True, True, True),
    "rouge_default": (False, True, True),
}

# --------------------------------------------------------------------------------------
# Vendored from https://github.com/huggingface/evaluate (metrics/bleu/tokenizer_13a.py),
# itself derived from sacreBLEU (https://github.com/mjpost/sacrebleu).
# Copyright 2020 SacreBLEU Authors. Licensed under the Apache License, Version 2.0.
# Only change: the per-line ``lru_cache`` decorators were removed (identical output).
# --------------------------------------------------------------------------------------


class _TokenizerRegexp:
    """Language-dependent part of the ``13a`` tokenizer (Western punctuation rules)."""

    def __init__(self) -> None:
        self._re: list[tuple[re.Pattern[str], str]] = [
            # language-dependent part (assuming Western languages)
            (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),
            # tokenize period and comma unless preceded by a digit
            (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),
            # tokenize period and comma unless followed by a digit
            (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),
            # tokenize dash when preceded by a digit
            (re.compile(r"([0-9])(-)"), r"\1 \2 "),
        ]

    def __call__(self, line: str) -> list[str]:
        for regex, repl in self._re:
            line = regex.sub(repl, line)
        # no leading or trailing spaces, single space within words
        return line.split()


class Tokenizer13a:
    """WMT ``13a`` tokenizer: language-independent clean-up then :class:`_TokenizerRegexp`."""

    def __init__(self) -> None:
        self._post_tokenizer = _TokenizerRegexp()

    def signature(self) -> str:
        return "13a"

    def __call__(self, line: str) -> list[str]:
        # language-independent part:
        line = line.replace("<skipped>", "")
        line = line.replace("-\n", "")
        line = line.replace("\n", " ")
        if "&" in line:
            line = line.replace("&quot;", '"')
            line = line.replace("&amp;", "&")
            line = line.replace("&lt;", "<")
            line = line.replace("&gt;", ">")
        return self._post_tokenizer(f" {line} ")


# ---------------------------------- end of vendored code --------------------------------

# rouge_score.tokenize defaults (reproduced, not imported, so the mode is stable across versions).
_ROUGE_NON_ALPHANUM_RE = re.compile(r"[^a-z0-9]+")
_ROUGE_SPACES_RE = re.compile(r"\s+")
_ROUGE_VALID_TOKEN_RE = re.compile(r"^[a-z0-9]+$")


def _is_punct(ch: str) -> bool:
    """True for Unicode punctuation (P*) and symbols (S*); letters, marks and digits are kept."""
    return unicodedata.category(ch)[0] in "PS"


def _punct_to_space(text: str) -> str:
    return "".join(" " if _is_punct(ch) else ch for ch in text)


def _load_underthesea() -> Callable[[str], list[str]]:
    try:
        module = importlib.import_module("underthesea")
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "MetricTokenizer(mode='underthesea') needs the optional dependency "
            "'underthesea' (pip install 'vivlmoe[viet]')"
        ) from exc
    return module.word_tokenize


class MetricTokenizer:
    """Tokenizer for text metrics; usable as ``rouge_score``'s ``tokenizer=`` argument.

    Built either from a :class:`MetricTokenizerCfg` (its flags are honoured) or from a bare
    mode string (the mode's canonical preset is used, e.g. ``'13a'`` is case-sensitive).
    For ``rouge_default`` the flags are meaningless and ignored: the mode reproduces
    ``rouge_score`` exactly.
    """

    def __init__(self, cfg: MetricTokenizerCfg | str) -> None:
        if isinstance(cfg, str):
            if cfg not in _PRESETS:
                raise ValueError(
                    f"unknown metric tokenizer mode {cfg!r}; expected {sorted(_PRESETS)}"
                )
            mode = cfg
            self.nfc, self.lowercase, self.strip_punct = _PRESETS[mode]
        else:
            mode = cfg.mode
            if mode not in _PRESETS:
                raise ValueError(
                    f"unknown metric tokenizer mode {mode!r}; expected {sorted(_PRESETS)}"
                )
            self.nfc, self.lowercase, self.strip_punct = cfg.nfc, cfg.lowercase, cfg.strip_punct
        self.mode: str = mode
        self._t13a = Tokenizer13a() if mode == "13a" else None
        self._segment = _load_underthesea() if mode == "underthesea" else None

    # -- description (recorded in metrics.json) ----------------------------------------
    @property
    def signature(self) -> str:
        if self.mode == "rouge_default":
            return "rouge_default"
        return (
            f"{self.mode}(nfc={self.nfc},lowercase={self.lowercase},strip_punct={self.strip_punct})"
        )

    def __repr__(self) -> str:
        return f"MetricTokenizer({self.signature})"

    # -- API ---------------------------------------------------------------------------
    def tokenize(self, text: str) -> list[str]:
        """Tokenize one string according to the configured mode."""
        if not isinstance(text, str):
            raise TypeError(f"MetricTokenizer expects str, got {type(text).__name__}")
        if self.mode == "rouge_default":
            return self._tokenize_rouge_default(text)
        text = self._pre_normalize(text)
        if self.mode == "13a":
            assert self._t13a is not None
            tokens = self._t13a(text)
            if self.strip_punct:
                tokens = [t for t in tokens if not all(_is_punct(ch) for ch in t)]
            return tokens
        if self.mode == "vi_safe":
            if self.strip_punct:
                text = _punct_to_space(text)
            return text.split()
        assert self._segment is not None
        words = self._segment(text)
        if self.strip_punct:
            words = [w for w in words if not all(_is_punct(ch) or ch.isspace() for ch in w)]
        return [w.replace(" ", "_") for w in words if w.strip()]

    __call__ = tokenize

    def normalize(self, text: str) -> str:
        """Canonical form used for exact match: the space-joined token sequence."""
        return " ".join(self.tokenize(text))

    # -- helpers -----------------------------------------------------------------------
    def _pre_normalize(self, text: str) -> str:
        if self.nfc:
            text = unicodedata.normalize("NFC", text)
        if self.lowercase:
            text = text.lower()
        return text

    @staticmethod
    def _tokenize_rouge_default(text: str) -> list[str]:
        text = _ROUGE_NON_ALPHANUM_RE.sub(" ", text.lower())
        return [t for t in _ROUGE_SPACES_RE.split(text) if _ROUGE_VALID_TOKEN_RE.match(t)]
