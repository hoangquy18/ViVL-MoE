from __future__ import annotations

import pytest
import torch

from vivlmoe.data.collators import (
    CausalLMCollator,
    ContrastiveCollator,
    RetrievalImageCollator,
    RetrievalTextCollator,
    Seq2SeqCollator,
    TeacherCollator,
)


def _img_items(n: int, text_key: str = "text") -> list[dict]:
    return [
        {
            "uid": i,
            "image_id": f"img{i}",
            "pixel_values": torch.full((3, 8, 8), float(i)),
            text_key: "một con chó " * (i + 1),
        }
        for i in range(n)
    ]


def test_contrastive_collator_contract(bundle) -> None:
    batch = ContrastiveCollator(bundle.vi, max_len=6)(_img_items(3))
    assert set(batch) == {"pixel_values", "input_ids", "attention_mask", "uids", "image_ids"}
    assert batch["pixel_values"].shape == (3, 3, 8, 8) and batch["input_ids"].shape == (3, 6)
    assert batch["input_ids"].dtype == torch.long and batch["attention_mask"].dtype == torch.long
    assert batch["attention_mask"][0].sum() < 6 and batch["attention_mask"][2].sum() == 6
    assert batch["uids"] == [0, 1, 2] and batch["image_ids"] == ["img0", "img1", "img2"]


def test_teacher_collator_clip_77_and_longest(bundle) -> None:
    items = [
        {"uid": 0, "src": "a dog", "tgt": "một con chó"},
        {"uid": 1, "src": "a dog runs on the grass", "tgt": "chó"},
    ]
    batch = TeacherCollator(bundle.clip_en, bundle.vi, en_max_len=77, vi_max_len=16)(items)
    assert set(batch) == {
        "en_input_ids",
        "en_attention_mask",
        "vi_input_ids",
        "vi_attention_mask",
        "uids",
    }
    assert batch["en_input_ids"].shape == (2, 77) and batch["en_attention_mask"].shape == (2, 77)
    assert batch["vi_input_ids"].shape == (2, 5)  # <s> 3 words </s>
    assert batch["en_input_ids"][0, 0] == bundle.clip_en.bos_token_id
    assert (batch["en_input_ids"][0] == bundle.clip_en.eos_token_id).sum() >= 1
    assert batch["en_attention_mask"][0].sum() == 4  # bos a dog eos
    assert batch["vi_input_ids"][1, -1] == bundle.vi.pad_token_id


def test_teacher_collator_lowercases_the_teacher_side_only(bundle) -> None:
    items = [{"uid": 0, "src": "Một Con Chó", "tgt": "Một Con Chó"}]
    collate = TeacherCollator(
        bundle.vi, bundle.vi, en_max_len=64, vi_max_len=16, lowercase_teacher=True
    )  # a case-sensitive teacher tokenizer (the fake CLIP one lower-cases by itself)
    batch = collate(items)
    assert batch["en_input_ids"].shape == (1, 64)
    assert batch["en_input_ids"][0, :5].tolist() == bundle.vi("một con chó")["input_ids"]
    assert batch["vi_input_ids"][0].tolist() == bundle.vi("Một Con Chó")["input_ids"]
    assert bundle.vi.unk_token_id in batch["vi_input_ids"][0].tolist()


def test_seq2seq_collator_caption_and_vqa(bundle) -> None:
    items = _img_items(2)
    train = Seq2SeqCollator(None, bundle.decoder, with_text=False, train=True, max_target_len=8)(
        items
    )
    assert set(train) == {"pixel_values", "uids", "image_ids", "prompts", "labels"}
    labels = train["labels"]
    assert labels.shape == (2, 7) and labels.dtype == torch.long  # 6 words + </s>
    assert (labels[0] == -100).sum() > 0 and (labels == bundle.decoder.pad_token_id).sum() == 0
    assert labels[0][labels[0] != -100][-1] == bundle.decoder.eos_token_id
    assert train["prompts"] == ["", ""]
    eval_items = [dict(it, refs=["r1", "r2"]) for it in items]
    ev = Seq2SeqCollator(None, bundle.decoder, with_text=False, train=False)(eval_items)
    assert "labels" not in ev and ev["refs"] == [["r1", "r2"], ["r1", "r2"]]
    vqa_items = [
        dict(it, question=f"cái gì {i}?", answer="con chó", refs=["con chó"])
        for i, it in enumerate(items)
    ]
    vq = Seq2SeqCollator(
        bundle.vi,
        bundle.decoder,
        with_text=True,
        train=True,
        prompt_template="Hỏi: {question}",
        max_source_len=6,
        max_target_len=4,
    )(vqa_items)
    assert set(vq) == {
        "pixel_values",
        "uids",
        "image_ids",
        "prompts",
        "labels",
        "input_ids",
        "attention_mask",
    }
    assert vq["prompts"][1] == "Hỏi: cái gì 1?" and vq["input_ids"].shape == (2, 6)
    assert vq["labels"].shape[1] <= 4
    ev = Seq2SeqCollator(bundle.vi, bundle.decoder, with_text=True, train=False)(vqa_items)
    assert ev["refs"] == [["con chó"], ["con chó"]] and "input_ids" in ev


def test_retrieval_collators(bundle) -> None:
    img = RetrievalImageCollator()(
        [{"uid": 0, "img_idx": 0, "image_id": "a", "pixel_values": torch.zeros(3, 4, 4)}]
    )
    assert set(img) == {"pixel_values", "uids", "image_ids", "img_idx"} and img[
        "img_idx"
    ].tolist() == [0]
    txt = RetrievalTextCollator(bundle.vi, 8)(
        [{"uid": 3, "cap_idx": 3, "img_idx": 1, "image_id": "b", "text": "chó"}]
    )
    assert set(txt) == {"input_ids", "attention_mask", "uids", "image_ids", "cap_idx", "img_idx"}
    assert txt["cap_idx"].dtype == torch.long and txt["uids"] == [3]


def test_causal_lm_collator_prompt_and_eos_terminated_answers(bundle) -> None:
    dec = bundle.decoder
    items = [
        {"question": "con chó", "answer": "một con chó màu đỏ"},
        {"question": "mèo", "answer": "hai"},
    ]
    for i, it in enumerate(items):
        it.update(uid=i, image_id=f"img{i}", pixel_values=torch.zeros(3, 8, 8), refs=["x"])
    collate = CausalLMCollator(
        bundle.vi, dec, with_text=True, train=True, llm_prompt_template="{question} ?",
        max_target_len=4,
    )  # fmt: skip
    batch = collate(items)
    assert (
        batch["llm_input_ids"][0].tolist()
        == dec("con chó ?", add_special_tokens=False)["input_ids"]
    )
    assert batch["llm_attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    words = dec("một con chó", add_special_tokens=False)["input_ids"]  # 3 tokens + EOS = 4
    assert batch["labels"][0].tolist() == [*words, dec.eos_token_id]
    hai = dec("hai", add_special_tokens=False)["input_ids"]
    assert batch["labels"][1].tolist() == [*hai, dec.eos_token_id, -100, -100]
    assert batch["input_ids"].shape[0] == 2  # the backbone still encodes the question
    caption = CausalLMCollator(None, dec, with_text=False, train=False)(items)
    assert caption["llm_input_ids"].shape == (2, 0) and caption["refs"] == [["x"], ["x"]]
    assert "labels" not in caption and "input_ids" not in caption
    with pytest.raises(ValueError, match="has no {question}"):
        CausalLMCollator(None, dec, with_text=False, train=True, llm_prompt_template="{question}")
