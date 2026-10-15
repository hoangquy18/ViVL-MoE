"""Vietnamese text normalisation (ported from the legacy ``text_preprocess.py``, made optional).

Differences from the legacy code, all deliberate:

* Unicode is normalised with NFC instead of a hand-written Windows-1252 table.
* The ``(\\w)\\1+`` "collapse repeated characters" rule is **off** by default: it also collapses
  legitimate doubled letters and digits (``xoong`` -> ``xong``, ``2000`` -> ``20``); words such as
  ``tất`` must survive normalisation unchanged (``collapse_repeated_chars`` keeps the legacy rule
  available for parity experiments only).
* Tone-mark repositioning (``hoà`` -> ``hòa``) is applied to letter runs only, so trailing
  punctuation no longer blocks it, and letter case is preserved.
* Lower-casing, emoji / mention / punctuation stripping are individual switches.
* ``underthesea.text_normalize`` is optional (``pip install vivlmoe[viet]``); asking for it without
  the package raises instead of silently skipping.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Protocol

_VOWELS_TABLE = [
    ["a", "à", "á", "ả", "ã", "ạ"],
    ["ă", "ằ", "ắ", "ẳ", "ẵ", "ặ"],
    ["â", "ầ", "ấ", "ẩ", "ẫ", "ậ"],
    ["e", "è", "é", "ẻ", "ẽ", "ẹ"],
    ["ê", "ề", "ế", "ể", "ễ", "ệ"],
    ["i", "ì", "í", "ỉ", "ĩ", "ị"],
    ["o", "ò", "ó", "ỏ", "õ", "ọ"],
    ["ô", "ồ", "ố", "ổ", "ỗ", "ộ"],
    ["ơ", "ờ", "ớ", "ở", "ỡ", "ợ"],
    ["u", "ù", "ú", "ủ", "ũ", "ụ"],
    ["ư", "ừ", "ứ", "ử", "ữ", "ự"],
    ["y", "ỳ", "ý", "ỷ", "ỹ", "ỵ"],
]
_VOWEL_TO_ID: dict[str, tuple[int, int]] = {
    ch: (row, col) for row, chars in enumerate(_VOWELS_TABLE) for col, ch in enumerate(chars)
}
_ROW_E_HAT, _ROW_O_HAT, _ROW_O_HORN, _ROW_U, _ROW_I = 4, 7, 8, 9, 5

_LETTER_RUN = re.compile(r"[^\W\d_]+")
_EMOJI = re.compile(
    "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f1e0-\U0001f1ff]+"
)
_MENTION = re.compile(r"(@[A-Za-z0-9_]+)|(#[0-9A-Za-z_]+)")
_LEGACY_PUNCT = re.compile(r"[:)^@!`~%;?(\+\-\'\"“”]+")
_REPEATED = re.compile(r"(\w)\1+")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizeOptions:
    nfc: bool = True
    tone_marks: bool = True  # move the tone mark onto the canonical vowel (hoà -> hòa)
    lowercase: bool = False
    strip_emoji: bool = False
    strip_mentions: bool = False  # @user / #hashtag
    strip_punct: bool = False  # legacy ViMACSA character class
    collapse_repeated_chars: bool = False  # legacy (\w)\1+ rule; destroys words, keep off
    collapse_spaces: bool = True
    use_underthesea: bool = False  # underthesea.text_normalize before the rules above


PRESETS: dict[str, NormalizeOptions | None] = {
    "none": None,
    "pretrain": NormalizeOptions(),
    # legacy train_macsa.py: lowercase + emoji + mentions + punctuation class; the repeated-char
    # collapse and underthesea are NOT enabled (see module docstring).
    "vimacsa_legacy": NormalizeOptions(
        lowercase=True, strip_emoji=True, strip_mentions=True, strip_punct=True
    ),
}


class TextNormalizer(Protocol):
    def __call__(self, text: str) -> str: ...


def _valid_vietnamese_word(chars: list[str]) -> bool:
    """Vowels must form one contiguous run (else the word is foreign; leave it alone)."""
    last = -1
    for i, ch in enumerate(chars):
        if ch in _VOWEL_TO_ID:
            if last != -1 and i - last != 1:
                return False
            last = i
    return True


def _put_tone(chars: list[str], idx: int, tone: int) -> None:
    row = _VOWEL_TO_ID[chars[idx]][0]
    chars[idx] = _VOWELS_TABLE[row][tone]


def standardize_word(word: str) -> str:
    """Port of the legacy ``WordStandardized`` (lower-case letters only, case preserved)."""
    lower = word.lower()
    chars = list(lower)
    if not _valid_vietnamese_word(chars):
        return word
    vowels: list[int] = []
    qu_or_gi = False
    tone = 0
    for i, ch in enumerate(chars):
        if ch not in _VOWEL_TO_ID:
            continue
        row, col = _VOWEL_TO_ID[ch]
        if row == _ROW_U and i != 0 and chars[i - 1] == "q":
            qu_or_gi = True
        elif row == _ROW_I and i != 0 and chars[i - 1] == "g":
            qu_or_gi = True
        if col != 0:
            tone = col
            chars[i] = _VOWELS_TABLE[row][0]
        vowels.append(i)
    if not vowels:
        return word
    if len(vowels) == 1:
        _put_tone(chars, vowels[0], tone)
    else:
        done = False
        for vi in vowels:
            row = _VOWEL_TO_ID[chars[vi]][0]
            if row in (_ROW_E_HAT, _ROW_O_HAT, _ROW_O_HORN):  # ê, ô, ơ always carry the tone
                _put_tone(chars, vi, tone)
                done = True
                break
        if not done:
            if qu_or_gi and len(vowels) in (2, 3):
                _put_tone(chars, vowels[1], tone)
            elif len(vowels) == 2:
                target = vowels[0] if vowels[-1] == len(chars) - 1 else vowels[-1]
                _put_tone(chars, target, tone)
            elif len(vowels) == 3:
                target = vowels[1] if vowels[-1] == len(chars) - 1 else vowels[-1]
                _put_tone(chars, target, tone)
            elif qu_or_gi:
                pass  # legacy: >3 vowels with qu/gi -> unchanged (tone already stripped)
    out = "".join(chars)
    if len(out) != len(word):
        return out
    return "".join(o.upper() if w.isupper() else o for o, w in zip(out, word, strict=True))


def reposition_tone_marks(text: str) -> str:
    return _LETTER_RUN.sub(lambda m: standardize_word(m.group(0)), text)


class VietnameseNormalizer:
    """Callable normaliser configured by :class:`NormalizeOptions`."""

    def __init__(self, options: NormalizeOptions) -> None:
        self.options = options
        self._underthesea = None
        if options.use_underthesea:
            try:
                from underthesea import text_normalize as uts_normalize
            except ImportError as exc:
                raise ImportError(
                    "NormalizeOptions.use_underthesea=True needs the optional 'underthesea' package "
                    "(pip install 'vivlmoe[viet]')"
                ) from exc
            self._underthesea = uts_normalize

    def __call__(self, text: str) -> str:
        o = self.options
        if o.nfc:
            text = unicodedata.normalize("NFC", text)
        if self._underthesea is not None:
            text = self._underthesea(text)
        if o.lowercase:
            text = text.lower()
        if o.collapse_repeated_chars:
            text = _REPEATED.sub(r"\1", text)
        if o.strip_emoji:
            text = _EMOJI.sub("", text)
        if o.strip_mentions:
            text = _MENTION.sub("", text)
        if o.strip_punct:
            text = _LEGACY_PUNCT.sub("", text)
        if o.tone_marks:
            text = reposition_tone_marks(text)
        if o.collapse_spaces:
            text = _SPACES.sub(" ", text).strip()
        return text

    def __repr__(self) -> str:
        return f"VietnameseNormalizer({self.options})"


def build_normalizer(
    preset: str | NormalizeOptions | None, **overrides: bool
) -> VietnameseNormalizer | None:
    """``None`` for preset ``none``; otherwise a :class:`VietnameseNormalizer`."""
    if preset is None:
        return None
    if isinstance(preset, NormalizeOptions):
        options: NormalizeOptions | None = preset
    else:
        if preset not in PRESETS:
            raise ValueError(f"unknown text_normalize preset {preset!r}; known: {sorted(PRESETS)}")
        options = PRESETS[preset]
    if options is None:
        if overrides:
            raise ValueError("overrides cannot be applied to the 'none' preset")
        return None
    if overrides:
        options = replace(options, **overrides)
    return VietnameseNormalizer(options)


__all__ = [
    "PRESETS",
    "NormalizeOptions",
    "TextNormalizer",
    "VietnameseNormalizer",
    "build_normalizer",
    "reposition_tone_marks",
    "standardize_word",
]
