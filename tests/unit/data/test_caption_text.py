"""v2 distillation data: caption_text, ColumnarCaptionText, ConcatRecordSet and text mixtures."""

from __future__ import annotations

import dataclasses
import json
import pickle
from pathlib import Path

import pytest

from vivlmoe.config.schema import DataSourceCfg, SplitSpec
from vivlmoe.data.datasets import ParallelTextDataset
from vivlmoe.data.records import (
    ColumnarCaptionText,
    ColumnarImageText,
    ConcatRecordSet,
    ListRecordSet,
    ParallelTextRecord,
)
from vivlmoe.data.registry import build_adapter


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _caption_cfg(root: Path, **kw) -> DataSourceCfg:
    return DataSourceCfg(
        kind="caption_text", root=str(root), splits={"train": SplitSpec(file="c.jsonl")}, **kw
    )


def test_caption_text_dedupes_and_normalises_both_sides(tmp_path: Path) -> None:
    _jsonl(
        tmp_path / "c.jsonl",
        [
            {"image": "a.jpg", "caption": "Một con chó"},
            {"image": "b.jpg", "caption": "Một con chó"},  # the same caption on another image
            {"image": "missing.jpg", "caption": "hoà bình"},  # images are never opened
        ],
    )
    cfg = _caption_cfg(tmp_path, text_normalize="pretrain")
    adapter = build_adapter(cfg)
    recs = adapter.load_split("train")
    assert isinstance(recs, ColumnarCaptionText) and recs.record_type is ParallelTextRecord
    assert [(r.uid, r.src, r.tgt) for r in recs] == [
        (0, "Một con chó", "Một con chó"),
        (1, "hòa bình", "hòa bình"),  # the teacher reads the normalised caption too
    ]
    assert recs.groups() == [[0], [1]]
    with pytest.raises(TypeError, match="text-only"):
        adapter.image_store("train")
    every = build_adapter(dataclasses.replace(cfg, options={"dedupe": False})).load_split("train")
    assert len(every) == 3


def test_caption_text_invalid_lines(tmp_path: Path) -> None:
    (tmp_path / "c.jsonl").write_text(
        '{"caption": "tốt"}\nnot json\n{"caption": "  "}\n{"text": "x"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid JSON"):
        build_adapter(_caption_cfg(tmp_path)).load_split("train")
    skip = build_adapter(_caption_cfg(tmp_path, options={"skip_invalid": True}))
    assert [r.tgt for r in skip.load_split("train")] == ["tốt"]


def test_concat_record_set_contract() -> None:
    captions = ColumnarCaptionText(["một", "hai", "ba"])
    pairs = ListRecordSet(
        [ParallelTextRecord(0, "four", "bốn"), ParallelTextRecord(1, "five", "năm")]
    )
    cat = ConcatRecordSet([captions, pairs])
    assert len(cat) == 5 and cat.record_type is ParallelTextRecord
    assert [(r.uid, r.src, r.tgt) for r in cat] == [
        (0, "một", "một"),
        (1, "hai", "hai"),
        (2, "ba", "ba"),
        (3, "four", "bốn"),
        (4, "five", "năm"),
    ]
    assert cat[-1].tgt == "năm"
    with pytest.raises(IndexError):
        cat[5]
    assert cat.image_ids() == ["0:0", "0:1", "0:2", "1:0", "1:1"]  # unique across parts

    sub = cat.select([4, 0, 3])
    assert [(r.uid, r.tgt) for r in sub] == [(0, "năm"), (1, "một"), (2, "bốn")]
    assert sub.parts[0] is captions  # parts are shared, not copied
    assert sub.image_ids() == ["1:1", "0:0", "1:0"]
    assert [r.tgt for r in sub.select([2, 0])] == ["bốn", "năm"]  # nested selects compose

    upper = sub.map_text(str.upper)  # Vietnamese side: both sides of a caption, tgt of a pair
    assert [(r.src, r.tgt) for r in upper] == [("five", "NĂM"), ("MỘT", "MỘT"), ("four", "BỐN")]
    assert pickle.loads(pickle.dumps(upper))[1].src == "MỘT"
    with pytest.raises(TypeError, match="different types"):
        ConcatRecordSet([captions, ColumnarImageText(["a.jpg"], ["x"])])


def _write_phomt(root: Path, en: list[str], vi: list[str]) -> None:
    d = root / "tokenization" / "train"
    d.mkdir(parents=True)
    (d / "train.en").write_text("\n".join(en) + "\n", encoding="utf-8")
    (d / "train.vi").write_text("\n".join(vi) + "\n", encoding="utf-8")


def test_text_mixture_of_captions_and_phomt(tmp_path: Path) -> None:
    _jsonl(tmp_path / "cap" / "c.jsonl", [{"image": "a.jpg", "caption": "chó"}])
    _write_phomt(tmp_path / "phomt", ["a dog", "a cat"], ["con chó", "con mèo"])
    cap_src = {"kind": "caption_text", "root": str(tmp_path / "cap")}
    cap_src["splits"] = {"train": {"file": "c.jsonl"}}
    phomt_src = {"kind": "phomt", "root": str(tmp_path / "phomt"), "limit": 1}
    mix = DataSourceCfg(kind="mixture", options={"sources": [cap_src, phomt_src]})
    recs = build_adapter(mix).load_split("train")
    assert isinstance(recs, ConcatRecordSet)
    ds = ParallelTextDataset(recs)
    assert [ds[i] for i in range(len(ds))] == [
        {"uid": 0, "src": "chó", "tgt": "chó"},
        {"uid": 1, "src": "a dog", "tgt": "con chó"},
    ]
    image_src = {"kind": "jsonl", "root": str(tmp_path / "cap"), "image_check": "none"}
    image_src["splits"] = {"train": {"file": "c.jsonl"}}
    mixed = DataSourceCfg(kind="mixture", options={"sources": [image_src, cap_src]})
    with pytest.raises(TypeError, match="caption_text yields ParallelTextRecord"):
        build_adapter(mixed).load_split("train")


def test_named_mixture_sources_can_be_overridden_and_disabled(tmp_path: Path) -> None:
    from vivlmoe.config.loader import deep_merge, parse_overrides

    _jsonl(tmp_path / "cap" / "c.jsonl", [{"caption": "chó"}])
    _write_phomt(tmp_path / "phomt", ["a dog"], ["con chó"])
    options = {
        "sources": {
            "captions": {
                "kind": "caption_text",
                "root": "unset",
                "splits": {"train": {"file": "c.jsonl"}},
            },
            "phomt": {"kind": "phomt", "root": str(tmp_path / "phomt")},
        }
    }
    cli = parse_overrides([f"sources.captions.root={tmp_path / 'cap'}"])
    recs = build_adapter(DataSourceCfg(kind="mixture", options=deep_merge(options, cli)))
    assert [r.tgt for r in recs.load_split("train")] == ["chó", "con chó"]
    off = deep_merge(
        options,
        parse_overrides([f"sources.captions.root={tmp_path / 'cap'}", "sources.phomt=null"]),
    )
    assert [
        r.tgt for r in build_adapter(DataSourceCfg(kind="mixture", options=off)).load_split("train")
    ] == ["chó"]
    with pytest.raises(ValueError, match="at least one source"):
        build_adapter(DataSourceCfg(kind="mixture", options={"sources": {"a": None}}))
