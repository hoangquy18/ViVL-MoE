from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.unit.data.conftest import write_images
from vivlmoe.data.datasets import (
    CaptionEvalDataset,
    ImageTextDataset,
    ParallelTextDataset,
    RetrievalEvalDataset,
    VQADataset,
)
from vivlmoe.data.image_store import ImageLoadError, LocalImageStore
from vivlmoe.data.records import (
    ImageRef,
    ImageTextRecord,
    ListRecordSet,
    ParallelTextRecord,
    VQARecord,
)


def _records() -> ListRecordSet:
    def r(i, iid, text, refs):
        return ImageTextRecord(i, iid, ImageRef(0, f"{iid}.jpg"), text, refs)

    return ListRecordSet(
        [
            r(0, "a", "a0", ("a0", "a1")),
            r(1, "a", "a1", ("a0", "a1")),
            r(2, "b", "b0", ("b0",)),
            r(3, "gone", "x", ()),
        ]
    )


def test_image_text_and_error(tmp_path: Path, tiny_transform) -> None:
    write_images(tmp_path, ["a.jpg", "b.jpg"], size=20)
    ds = ImageTextDataset(_records(), LocalImageStore(tmp_path), tiny_transform, draft_size=32)
    it = ds[1]
    assert it["pixel_values"].shape == (3, 32, 32) and it == {
        **it,
        "uid": 1,
        "image_id": "a",
        "text": "a1",
    }
    assert ds.groups() == [[0, 1], [2], [3]]
    with pytest.raises(ImageLoadError, match="uid=3"):
        ds[3]
    with pytest.raises(TypeError):
        ImageTextDataset(
            ListRecordSet([ParallelTextRecord(0, "a", "b")]),
            LocalImageStore(tmp_path),
            tiny_transform,
        )


def test_retrieval_and_caption_eval(tmp_path: Path, tiny_transform) -> None:
    write_images(tmp_path, ["a.jpg", "b.jpg", "gone.jpg"], size=20)
    recs = _records()
    ret = RetrievalEvalDataset(recs, LocalImageStore(tmp_path), tiny_transform)
    assert ret.image_ids == ["a", "b", "gone"] and ret.captions == ["a0", "a1", "b0", "x"]
    assert ret.img2caps == [[0, 1], [2], [3]] and ret.cap2img == [0, 0, 1, 2]
    assert ret.images[2]["uid"] == 2 and ret.images[2]["pixel_values"].shape == (3, 32, 32)
    assert ret.texts[1] == {"uid": 1, "cap_idx": 1, "img_idx": 0, "image_id": "a", "text": "a1"}
    cap = CaptionEvalDataset(recs, LocalImageStore(tmp_path), tiny_transform)
    assert (
        len(cap) == 3
        and cap[0]["refs"] == ["a0", "a1"]
        and cap[2]["refs"] == ["x"]
        and cap[1]["uid"] == 1
    )


def test_vqa_and_parallel(tmp_path: Path, tiny_transform) -> None:
    write_images(tmp_path, ["q.jpg"], size=20)
    recs = ListRecordSet([VQARecord(0, "q", ImageRef(0, "q.jpg"), "gì?", ("a", "b", "c"))])
    ds = VQADataset(recs, LocalImageStore(tmp_path), tiny_transform)
    assert len(ds) == 1 and ds[0]["answer"] == "a" and ds[0]["refs"] == ["a", "b", "c"]
    ex = VQADataset(recs, LocalImageStore(tmp_path), tiny_transform, expand_answers=True)
    assert (
        len(ex) == 3
        and [ex[i]["answer"] for i in range(3)] == ["a", "b", "c"]
        and ex[2]["record_uid"] == 0
    )
    assert torch.equal(ex[0]["pixel_values"], ex[2]["pixel_values"])
    par = ParallelTextDataset(ListRecordSet([ParallelTextRecord(0, "en", "vi")]))
    assert par[0] == {"uid": 0, "src": "en", "tgt": "vi"}
