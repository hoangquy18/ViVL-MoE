from __future__ import annotations

import unicodedata

import pytest

from vivlmoe.data.text_normalize import (
    PRESETS,
    NormalizeOptions,
    VietnameseNormalizer,
    build_normalizer,
    standardize_word,
)


@pytest.mark.parametrize(
    "word, expected",
    [
        ("hoà", "hòa"),
        ("hòa", "hòa"),
        ("thuỷ", "thủy"),
        ("quả", "quả"),
        ("tất", "tất"),
        ("người", "người"),
        ("toán", "toán"),
        ("Hoà", "Hòa"),
        ("HOÀ", "HÒA"),
        ("của", "của"),
        ("chuyện", "chuyện"),
        ("english", "english"),
        ("giờ", "giờ"),
    ],
)
def test_tone_repositioning(word: str, expected: str) -> None:
    assert standardize_word(word) == expected


def test_pretrain_preset_nfc_tone_marks_no_collapse() -> None:
    norm = build_normalizer("pretrain")
    assert norm is not None
    decomposed = unicodedata.normalize("NFD", "tất cả")
    assert norm(decomposed) == "tất cả"  # NFC + repeated 't' intact
    assert norm("  hoà,  bình  ") == "hòa, bình"  # punctuation does not block repositioning
    assert norm("Hà Nội") == "Hà Nội"  # no lowercasing in pretrain


def test_vimacsa_legacy_preset() -> None:
    norm = build_normalizer("vimacsa_legacy")
    assert norm is not None
    out = norm("Phòng ĐẸP quá!!! 😀 @user #tag (giá ok)")
    assert out == "phòng đẹp quá giá ok"
    assert not PRESETS["vimacsa_legacy"].collapse_repeated_chars
    assert norm("tất cả") == "tất cả"


def test_build_normalizer_presets_and_overrides() -> None:
    assert build_normalizer("none") is None and build_normalizer(None) is None
    with pytest.raises(ValueError):
        build_normalizer("bogus")
    with pytest.raises(ValueError):
        build_normalizer("none", lowercase=True)
    n = build_normalizer("pretrain", lowercase=True)
    assert n is not None and n("Hoà") == "hòa"
    legacy = VietnameseNormalizer(NormalizeOptions(collapse_repeated_chars=True))
    assert legacy("xoong 2000") == "xong 20"  # documents why the legacy rule is off by default
    assert build_normalizer("pretrain")("xoong 2000 tất") == "xoong 2000 tất"


def test_underthesea_optional() -> None:
    try:
        import underthesea  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="underthesea"):
            VietnameseNormalizer(NormalizeOptions(use_underthesea=True))
    else:
        assert VietnameseNormalizer(NormalizeOptions(use_underthesea=True))("hoà") == "hòa"
