"""Tokenizer bundle shared by the collators.

``vi`` is the XLM-R tokenizer (student / joint text), ``clip_en`` the CLIP teacher tokenizer
(Stage 1, max 77), ``decoder`` the ViT5 tokenizer (captioning / VQA targets). Towers configured with
``pretrained: false`` (random tiny configs) get deterministic in-memory WordLevel tokenizers whose
vocabulary fits ``tiny_config.vocab_size`` — this is logged loudly and only meant for tests and the
CPU smoke chain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast

from vivlmoe.config.schema import DataCfg, ModelCfg
from vivlmoe.utils.hf import resolve_model_path

log = logging.getLogger(__name__)

TinyStyle = Literal["roberta", "clip", "t5"]
TINY_WORDS = (
    "một hai ba bốn năm sáu bảy tám chín mười người đàn ông phụ nữ cậu bé cô gái con chó mèo "
    "chiếc xe đạp máy ô tô thuyền sông núi biển bãi cỏ cây hoa nhà cửa bàn ghế đang ngồi đứng "
    "chạy đi chơi ăn uống cầm mặc áo quần màu đỏ xanh vàng trắng đen trên dưới trong ngoài với "
    "và của có không là ở này kia rất nhiều ít lớn nhỏ đẹp what is the a an of on in and man "
    "woman dog cat bike car boat river mountain sea grass tree flower house table chair sitting "
    "standing running playing eating drinking holding wearing red green yellow white black room "
    "food service location facilities public area hotel good bad ok yes no , . ? ! : empty"
).split()


def _tiny_vocab(specials: list[str], vocab_size: int) -> dict[str, int]:
    """Deterministic vocabulary: specials first, then filler words, then ``w<i>`` tokens."""
    if vocab_size < len(specials) + 8:
        raise ValueError(f"vocab_size {vocab_size} too small for {len(specials)} specials")
    vocab: dict[str, int] = {}
    for tok in specials:
        vocab[tok] = len(vocab)
    for w in TINY_WORDS:
        if len(vocab) >= vocab_size:
            break
        if w not in vocab:
            vocab[w] = len(vocab)
    i = 0
    while len(vocab) < vocab_size:
        tok = f"w{i}"
        if tok not in vocab:
            vocab[tok] = len(vocab)
        i += 1
    return vocab


def tiny_wordlevel_tokenizer(
    style: TinyStyle, vocab_size: int = 1000, model_max_length: int | None = None
) -> PreTrainedTokenizerFast:
    """Real in-memory ``PreTrainedTokenizerFast`` (WordLevel) mimicking the special-token layout of
    XLM-R (``<s>``=0, ``<pad>``=1, ``</s>``=2), CLIP (eos = max id, pad = eos, lower-cased, max 77) or
    T5 (``<pad>``=0, ``</s>``=1)."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

    if style == "roberta":
        specials = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
        vocab = _tiny_vocab(specials, vocab_size)
        tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> </s> $B </s>",
            special_tokens=[("<s>", 0), ("</s>", 2)],
        )
        return PreTrainedTokenizerFast(
            tokenizer_object=tok,
            bos_token="<s>",
            eos_token="</s>",
            sep_token="</s>",
            cls_token="<s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            model_max_length=model_max_length or 512,
        )
    if style == "clip":
        specials = ["<|startoftext|>", "<unk>"]  # real CLIP has no <unk>; a distinct id keeps
        vocab = _tiny_vocab(specials, vocab_size - 1)  # argmax-EOS pooling valid for fakes
        vocab["<|endoftext|>"] = vocab_size - 1  # eos is the largest id, like the real CLIP BPE
        tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        tok.normalizer = normalizers.Lowercase()
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.post_processor = processors.TemplateProcessing(
            single="<|startoftext|> $A <|endoftext|>",
            special_tokens=[("<|startoftext|>", 0), ("<|endoftext|>", vocab_size - 1)],
        )
        return PreTrainedTokenizerFast(
            tokenizer_object=tok,
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
            unk_token="<unk>",
            pad_token="<|endoftext|>",
            model_max_length=model_max_length or 77,
        )
    if style == "t5":
        specials = ["<pad>", "</s>", "<unk>"]
        vocab = _tiny_vocab(specials, vocab_size)
        tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.post_processor = processors.TemplateProcessing(
            single="$A </s>", pair="$A </s> $B </s>", special_tokens=[("</s>", 1)]
        )
        return PreTrainedTokenizerFast(
            tokenizer_object=tok,
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            model_max_length=model_max_length or 512,
        )
    raise ValueError(f"unknown tiny tokenizer style {style!r}")


@dataclass
class TokenizerBundle:
    vi: PreTrainedTokenizerBase
    clip_en: PreTrainedTokenizerBase | None = None
    decoder: PreTrainedTokenizerBase | None = None
    vi_max_len: int = 64
    en_max_len: int = 77
    decoder_max_len: int = 64
    cls_max_len: int = 170

    def require_clip(self) -> PreTrainedTokenizerBase:
        if self.clip_en is None:
            raise ValueError("this task needs the CLIP tokenizer; build_tokenizers(need_clip=True)")
        return self.clip_en

    def require_decoder(self) -> PreTrainedTokenizerBase:
        if self.decoder is None:
            raise ValueError(
                "this task needs the decoder tokenizer; build_tokenizers(need_decoder=True)"
            )
        return self.decoder


def _load(
    name: str, pretrained: bool, tiny: dict[str, Any] | None, style: TinyStyle
) -> PreTrainedTokenizerBase:
    if not pretrained:
        vocab_size = int((tiny or {}).get("vocab_size", 1000))
        log.warning(
            "%s: pretrained=false -> deterministic tiny %s WordLevel tokenizer (vocab %d); "
            "tests / smoke runs only",
            name,
            style,
            vocab_size,
        )
        return tiny_wordlevel_tokenizer(style, vocab_size)
    path = resolve_model_path(name)
    return AutoTokenizer.from_pretrained(path)


def build_tokenizers(
    model_cfg: ModelCfg, data_cfg: DataCfg, need_clip: bool = False, need_decoder: bool = False
) -> TokenizerBundle:
    tk = data_cfg.tokenizer
    if tk.en_max_len > 77:
        raise ValueError(f"data.tokenizer.en_max_len must be <= 77 (CLIP), got {tk.en_max_len}")
    vi = _load(
        model_cfg.text.name, model_cfg.text.pretrained, model_cfg.text.tiny_config, "roberta"
    )
    clip = None
    if need_clip:
        v = model_cfg.vision
        tiny = model_cfg.stage1.teacher_tiny_config or v.tiny_config
        clip = _load(v.name, v.pretrained, tiny, "clip")
    dec = None
    if need_decoder:
        g = model_cfg.generation
        dec = _load(g.name, g.pretrained, g.tiny_config, "t5")
    return TokenizerBundle(
        vi=vi,
        clip_en=clip,
        decoder=dec,
        vi_max_len=tk.vi_max_len,
        en_max_len=tk.en_max_len,
        decoder_max_len=tk.decoder_max_len,
        cls_max_len=tk.cls_max_len,
    )


__all__ = ["TINY_WORDS", "TokenizerBundle", "build_tokenizers", "tiny_wordlevel_tokenizer"]
