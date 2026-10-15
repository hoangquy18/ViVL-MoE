from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tests.unit.data.conftest import TINY_SPEC, write_images
from vivlmoe.config.schema import DEFAULT_ASPECTS, DataSourceCfg
from vivlmoe.data.adapters.vimacsa import ASPECTS, LABELS, parse_labels
from vivlmoe.data.collators import MultiImageClsCollator
from vivlmoe.data.datasets import MultiImageClsDataset
from vivlmoe.data.registry import build_adapter


def make_vimacsa(root: Path) -> None:
    rows = [
        {
            "comment": "Phòng rất đẹp, khu vực công cộng bẩn",
            "images": ["i1.jpg", "i2.jpg"],
            "unused": 0,
            "labels": ["Room#Positive", "Public_area#Negative"],
        },
        {
            "comment": "Nhiều ảnh " * 3,
            "images": [f"m{k}.jpg" for k in range(8)],
            "unused": 1,
            "labels": ["Food#Neutral"],
        },
        {"comment": "Không ảnh", "images": [], "unused": 2, "labels": []},
    ]
    for split in ("train", "dev", "test"):
        (root / f"{split}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    tags = {"i1.jpg": ["Room", "Facilities"], "i2.jpg": ["Room", "Public_area"], "m0.jpg": ["Food"]}
    (root / "resnet152_image_label.json").write_text(json.dumps(tags), encoding="utf-8")
    write_images(root / "image", ["i1.jpg", "i2.jpg"] + [f"m{k}.jpg" for k in range(8)], size=20)


def test_parse_labels_and_constants() -> None:
    assert list(ASPECTS) == DEFAULT_ASPECTS and LABELS == (
        "None",
        "Negative",
        "Neutral",
        "Positive",
    )
    assert parse_labels(["Room#Positive", "Public_area#Negative"], ASPECTS, "t") == (
        0,
        0,
        3,
        0,
        0,
        1,
    )
    with pytest.raises(ValueError):
        parse_labels(["Pool#Positive"], ASPECTS, "t")
    with pytest.raises(ValueError):
        parse_labels(["Room#Great"], ASPECTS, "t")


@pytest.mark.parametrize("reader", ["pandas", "json"])
def test_vimacsa_records_tags_truncation(tmp_path: Path, reader: str) -> None:
    make_vimacsa(tmp_path)
    ad = build_adapter(
        DataSourceCfg(
            kind="vimacsa",
            root=str(tmp_path),
            text_normalize="vimacsa_legacy",
            options={"reader": reader},
        )
    )
    recs = ad.load_split("train")
    r0, r1, r2 = recs[0], recs[1], recs[2]
    assert r0.labels == (0, 0, 3, 0, 0, 1)
    assert r0.image_tags == ("Facilities", "Public_area", "Room")
    assert (
        r0.text == "phòng rất đẹp, khu vực công cộng bẩn"
    )  # legacy: lowercase; "," is not in the stripped class
    assert len(r1.images) == 6 and r1.image_tags == ("Food",) and r1.labels == (0, 2, 0, 0, 0, 0)
    assert r2.images == () and r2.labels == (0,) * 6 and r2.image_tags == ()
    assert [r.uid for r in recs] == [0, 1, 2] and r1.sample_id == "train:1"
    assert ad.image_store("dev").root == tmp_path / "image"


def test_vimacsa_dataset_and_collator_segments(tmp_path: Path, tiny_transform, bundle) -> None:
    make_vimacsa(tmp_path)
    ad = build_adapter(DataSourceCfg(kind="vimacsa", root=str(tmp_path)))
    recs = ad.load_split("train")
    ds = MultiImageClsDataset(
        recs, ad.image_store("train"), tiny_transform, (TINY_SPEC.crop_size,) * 2, max_images=6
    )
    it = ds[0]
    assert it["pixel_values"].shape == (6, 3, 32, 32) and it["image_mask"].tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert ds[2]["image_mask"].sum() == 0 and torch.count_nonzero(ds[2]["pixel_values"]) == 0
    assert it["labels"].tolist() == [0, 0, 3, 0, 0, 1]
    coll = MultiImageClsCollator(bundle.vi, DEFAULT_ASPECTS, use_image_tags=True, cls_max_len=32)
    batch = coll([ds[0], ds[2]])
    assert (
        batch["input_ids"].shape[:2] == (2, 6)
        and batch["attention_mask"].shape == batch["input_ids"].shape
    )
    assert batch["pixel_values"].shape == (2, 6, 3, 32, 32) and batch["image_mask"].shape == (2, 6)
    assert batch["labels"].shape == (2, 6) and batch["uids"] == [0, 2]
    dec = bundle.vi.decode(batch["input_ids"][0, 2], skip_special_tokens=False)
    assert dec.startswith("<s> room </s> </s>") or dec.startswith("<s> room </s></s>")
    assert "facilities , public area , room" in dec.replace("  ", " ")
    dec_last = bundle.vi.decode(batch["input_ids"][0, 5], skip_special_tokens=False)
    assert "public area" in dec_last.split("</s>")[0]
    dec_empty = bundle.vi.decode(batch["input_ids"][1, 0], skip_special_tokens=False)
    assert "empty" in dec_empty
    # pair encoding: first segment ends with </s></s> before the tag segment; only_first truncation
    ids = bundle.vi(
        ["room </s></s> " + "một " * 100], ["food"], truncation="only_first", max_length=16
    )["input_ids"][0]
    assert len(ids) == 16 and ids[-1] == bundle.vi.eos_token_id
    legacy = MultiImageClsCollator(
        bundle.vi, DEFAULT_ASPECTS, use_image_tags=True, cls_max_len=40, legacy_parity=True
    )
    assert legacy([ds[0]])["input_ids"].shape == (1, 6, 40)
    no_tags = MultiImageClsCollator(
        bundle.vi, DEFAULT_ASPECTS, use_image_tags=False, cls_max_len=40
    )
    assert "empty" not in bundle.vi.decode(no_tags([ds[2]])["input_ids"][0, 0])


def test_legacy_parity_lowercases_and_deunderscores_full_segment(bundle) -> None:
    """legacy train_macsa.py:354-358 applies .lower().replace('_', ' ') to the whole
    ``"<aspect> </s></s> <review>"`` string and to the joined tags."""
    tok = bundle.vi
    zeros = torch.zeros(1, 3, 4, 4)
    item = {
        "uid": 0,
        "text": "Phòng Snake_case ĐẸP",
        "image_tags": ("Public_area", "Room"),
        "pixel_values": zeros,
        "image_mask": torch.zeros(1, dtype=torch.bool),
        "labels": torch.zeros(6, dtype=torch.long),
    }
    aspects = ["Room", "Public_area"]
    legacy = MultiImageClsCollator(tok, aspects, cls_max_len=40, legacy_parity=True)
    ids = legacy([item])["input_ids"]
    assert ids.shape == (1, 2, 40)
    ref = tok(
        ["room </s></s> phòng snake case đẹp", "public area </s></s> phòng snake case đẹp"],
        ["public area , room"] * 2,
        padding="max_length",
        truncation="only_first",
        max_length=40,
        return_tensors="pt",
    )["input_ids"]
    assert torch.equal(ids[0], ref)
    plain = MultiImageClsCollator(tok, aspects, cls_max_len=40, legacy_parity=False)
    plain_ids = plain([item])["input_ids"]
    assert plain_ids.shape[-1] < 40  # pads to longest
    ref_plain = tok(
        ["room </s></s> Phòng Snake_case ĐẸP"], ["public area , room"], return_tensors="pt"
    )["input_ids"][0]
    n = ref_plain.numel()  # row 0 is padded to the longer "public area" row
    assert torch.equal(plain_ids[0, 0, :n], ref_plain)  # review untouched without legacy_parity
    assert (plain_ids[0, 0, n:] == tok.pad_token_id).all()
