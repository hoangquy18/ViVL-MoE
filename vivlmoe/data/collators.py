"""Collators: list of item dicts -> batch dict (tensors + raw string lists).

Batch keys are the contract shared with the task models (``forward(batch, mode)``):

* contrastive: ``pixel_values [B,3,H,W]``, ``input_ids [B,L]``, ``attention_mask [B,L]``, ``uids``,
  ``image_ids``
* stage 1: ``en_input_ids [B,77]``, ``en_attention_mask``, ``vi_input_ids [B,Lv]``,
  ``vi_attention_mask``, ``uids``
* seq2seq: ``pixel_values``, (``input_ids``, ``attention_mask`` when the task has text),
  ``labels [B,Ld]`` with ``-100`` pads (train) | ``refs`` (eval), ``prompts``, ``uids``, ``image_ids``
* classification: ``pixel_values [B,M,3,H,W]``, ``image_mask [B,M]``, ``input_ids [B,A,L]``,
  ``attention_mask [B,A,L]``, ``labels [B,A]``, ``uids``
* retrieval image side: ``pixel_values``, ``uids``, ``image_ids``, ``img_idx``; text side:
  ``input_ids``, ``attention_mask``, ``uids``, ``image_ids``, ``cap_idx``, ``img_idx``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

Batch = dict[str, Any]


def _encode(
    tok: PreTrainedTokenizerBase,
    texts: Sequence[str],
    max_len: int,
    padding: str = "longest",
    pair: Sequence[str] | None = None,
    truncation: bool | str = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    enc = tok(
        list(texts),
        list(pair) if pair is not None else None,
        padding=padding,
        truncation=truncation,
        max_length=max_len,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    return enc["input_ids"].long(), enc["attention_mask"].long()


def _stack(items: Sequence[Batch], key: str) -> torch.Tensor:
    return torch.stack([it[key] for it in items])


class ContrastiveCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_len: int) -> None:
        self.tok, self.max_len = tokenizer, max_len

    def __call__(self, items: Sequence[Batch]) -> Batch:
        ids, mask = _encode(self.tok, [it["text"] for it in items], self.max_len)
        return {
            "pixel_values": _stack(items, "pixel_values"),
            "input_ids": ids,
            "attention_mask": mask,
            "uids": [int(it["uid"]) for it in items],
            "image_ids": [it["image_id"] for it in items],
        }


class TeacherCollator:
    """Stage 1: CLIP tokens padded to ``en_max_len`` (77, ``padding=max_length``), XLM-R longest."""

    def __init__(
        self,
        clip_tokenizer: PreTrainedTokenizerBase,
        vi_tokenizer: PreTrainedTokenizerBase,
        en_max_len: int = 77,
        vi_max_len: int = 64,
    ) -> None:
        if en_max_len > 77:
            raise ValueError("CLIP text encoder supports at most 77 tokens")
        self.clip, self.vi = clip_tokenizer, vi_tokenizer
        self.en_max_len, self.vi_max_len = en_max_len, vi_max_len

    def __call__(self, items: Sequence[Batch]) -> Batch:
        en_ids, en_mask = _encode(
            self.clip, [it["src"] for it in items], self.en_max_len, padding="max_length"
        )
        vi_ids, vi_mask = _encode(self.vi, [it["tgt"] for it in items], self.vi_max_len)
        return {
            "en_input_ids": en_ids,
            "en_attention_mask": en_mask,
            "vi_input_ids": vi_ids,
            "vi_attention_mask": vi_mask,
            "uids": [int(it["uid"]) for it in items],
        }


class Seq2SeqCollator:
    """Captioning (``with_text=False``) and VQA (``with_text=True``, ``prompt_template``)."""

    def __init__(
        self,
        vi_tokenizer: PreTrainedTokenizerBase | None,
        decoder_tokenizer: PreTrainedTokenizerBase,
        with_text: bool,
        train: bool,
        prompt_template: str = "{question}",
        max_source_len: int = 64,
        max_target_len: int = 48,
    ) -> None:
        if with_text and vi_tokenizer is None:
            raise ValueError("with_text=True requires the XLM-R tokenizer")
        if "{question}" not in prompt_template:
            raise ValueError("prompt_template must contain '{question}'")
        self.vi, self.dec = vi_tokenizer, decoder_tokenizer
        self.with_text, self.train = with_text, train
        self.template = prompt_template
        self.max_source_len, self.max_target_len = max_source_len, max_target_len
        if self.dec.pad_token_id is None:
            raise ValueError("decoder tokenizer has no pad token")

    def __call__(self, items: Sequence[Batch]) -> Batch:
        batch: Batch = {
            "pixel_values": _stack(items, "pixel_values"),
            "uids": [int(it["uid"]) for it in items],
            "image_ids": [it["image_id"] for it in items],
        }
        if self.with_text:
            prompts = [self.template.format(question=it["question"]) for it in items]
            assert self.vi is not None
            ids, mask = _encode(self.vi, prompts, self.max_source_len)
            batch["input_ids"], batch["attention_mask"] = ids, mask
        else:
            prompts = [""] * len(items)
        batch["prompts"] = prompts
        if self.train:
            targets = [it["answer"] if "answer" in it else it["text"] for it in items]
            labels, _ = _encode(self.dec, targets, self.max_target_len)
            labels = labels.clone()
            labels[labels == self.dec.pad_token_id] = -100
            batch["labels"] = labels
        else:
            batch["refs"] = [list(it["refs"]) for it in items]
        return batch


class MultiImageClsCollator:
    """ViMACSA per-aspect pair encoding ``"<aspect> </s></s> <review>"`` (segment 1, truncated
    with ``only_first``) + image aspect tags (segment 2, ``"tag1 , tag2"``, ``"empty"`` if none).
    Pads to the longest sequence, or to ``cls_max_len`` when ``legacy_parity`` (legacy 170).

    ``legacy_parity`` also applies the legacy ``.lower().replace("_", " ")`` to the *whole* first
    segment and to the joined tags (``train_macsa.py:354-358``); aspects and tags are always
    lower-cased / de-underscored. Exact token parity with the legacy code additionally needs the
    review pre-processed with ``text_normalize: vimacsa_legacy``.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        aspects: Sequence[str],
        use_image_tags: bool = True,
        cls_max_len: int = 170,
        legacy_parity: bool = False,
    ) -> None:
        if tokenizer.sep_token is None:
            raise ValueError("classification tokenizer needs a sep token (XLM-R: </s>)")
        self.tok = tokenizer
        self.aspects = [a.lower().replace("_", " ") for a in aspects]
        self.use_image_tags = use_image_tags
        self.cls_max_len = cls_max_len
        self.legacy_parity = legacy_parity
        self.padding = "max_length" if legacy_parity else "longest"

    @staticmethod
    def _legacy(text: str) -> str:
        return text.lower().replace("_", " ")

    def _segments(self, it: Batch) -> tuple[list[str], list[str]]:
        sep = self.tok.sep_token
        firsts = [f"{asp} {sep}{sep} {it['text']}" for asp in self.aspects]
        second = " , ".join(self._legacy(t) for t in it.get("image_tags", [])) or "empty"
        if self.legacy_parity:
            firsts = [self._legacy(f) for f in firsts]
            second = self._legacy(second)
        return firsts, [second] * len(self.aspects)

    def __call__(self, items: Sequence[Batch]) -> Batch:
        firsts: list[str] = []
        seconds: list[str] = []
        for it in items:
            f, s = self._segments(it)
            firsts.extend(f)
            seconds.extend(s)
        ids, mask = _encode(
            self.tok,
            firsts,
            self.cls_max_len,
            padding=self.padding,
            pair=seconds if self.use_image_tags else None,
            truncation="only_first" if self.use_image_tags else True,
        )
        b, a = len(items), len(self.aspects)
        return {
            "pixel_values": _stack(items, "pixel_values"),
            "image_mask": _stack(items, "image_mask"),
            "input_ids": ids.view(b, a, -1),
            "attention_mask": mask.view(b, a, -1),
            "labels": _stack(items, "labels"),
            "uids": [int(it["uid"]) for it in items],
        }


class RetrievalImageCollator:
    def __call__(self, items: Sequence[Batch]) -> Batch:
        return {
            "pixel_values": _stack(items, "pixel_values"),
            "uids": [int(it["uid"]) for it in items],
            "image_ids": [it["image_id"] for it in items],
            "img_idx": torch.tensor([int(it["img_idx"]) for it in items], dtype=torch.long),
        }


class RetrievalTextCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_len: int) -> None:
        self.tok, self.max_len = tokenizer, max_len

    def __call__(self, items: Sequence[Batch]) -> Batch:
        ids, mask = _encode(self.tok, [it["text"] for it in items], self.max_len)
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "uids": [int(it["uid"]) for it in items],
            "image_ids": [it["image_id"] for it in items],
            "cap_idx": torch.tensor([int(it["cap_idx"]) for it in items], dtype=torch.long),
            "img_idx": torch.tensor([int(it["img_idx"]) for it in items], dtype=torch.long),
        }


__all__ = [
    "Batch",
    "ContrastiveCollator",
    "MultiImageClsCollator",
    "RetrievalImageCollator",
    "RetrievalTextCollator",
    "Seq2SeqCollator",
    "TeacherCollator",
]
