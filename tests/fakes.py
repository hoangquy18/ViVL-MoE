"""Hermetic test doubles: real in-memory ``PreTrainedTokenizerFast`` objects (no downloads)."""

from __future__ import annotations

from transformers import PreTrainedTokenizerFast

from vivlmoe.data.tokenization import TokenizerBundle, tiny_wordlevel_tokenizer


def tiny_roberta_tokenizer(vocab_size: int = 1000) -> PreTrainedTokenizerFast:
    """XLM-R layout: ``<s>``=0, ``<pad>``=1, ``</s>``=2, sep ``</s>`` (pair: ``<s> A </s></s> B </s>``)."""
    return tiny_wordlevel_tokenizer("roberta", vocab_size)


def tiny_clip_tokenizer(vocab_size: int = 1000) -> PreTrainedTokenizerFast:
    """CLIP layout: eos is the largest id, pad == eos, lower-cased, ``model_max_length`` 77."""
    return tiny_wordlevel_tokenizer("clip", vocab_size, model_max_length=77)


def tiny_t5_tokenizer(vocab_size: int = 1000) -> PreTrainedTokenizerFast:
    """T5 layout: ``<pad>``=0, ``</s>``=1 (appended to every sequence)."""
    return tiny_wordlevel_tokenizer("t5", vocab_size)


def build_fake_bundle(
    vi_max_len: int = 16, en_max_len: int = 77, decoder_max_len: int = 16, cls_max_len: int = 32
) -> TokenizerBundle:
    return TokenizerBundle(
        vi=tiny_roberta_tokenizer(),
        clip_en=tiny_clip_tokenizer(),
        decoder=tiny_t5_tokenizer(),
        vi_max_len=vi_max_len,
        en_max_len=en_max_len,
        decoder_max_len=decoder_max_len,
        cls_max_len=cls_max_len,
    )


__all__ = [
    "build_fake_bundle",
    "tiny_clip_tokenizer",
    "tiny_roberta_tokenizer",
    "tiny_t5_tokenizer",
]
