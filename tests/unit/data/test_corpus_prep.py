from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.data.conftest import write_images
from vivlmoe.config.schema import DataSourceCfg, SplitSpec
from vivlmoe.data.corpus_prep import (
    CaptionFile,
    CorpusBuildError,
    build_corpus,
    coco_id,
    index_images,
    parse_name,
    uitviic_official_split_ids,
)
from vivlmoe.data.registry import build_adapter


def test_parse_name_and_coco_id() -> None:
    assert parse_name("images/coco2017_000000203564.jpg") == ("coco2017", "000000203564")
    assert parse_name("oxford102_06403.jpg") == ("oxford102", "06403")
    assert parse_name("images/english_3.png") == ("english", "3")
    assert coco_id("coco2017", "000000203564") == 203564
    assert coco_id("vivqa", "000000068857") == 68857
    assert coco_id("openvivqa", "000000001624") is None  # OpenViVQA numbering is not COCO's
    with pytest.raises(ValueError):
        parse_name("images/noprefix.jpg")


def _dump(root: Path) -> list[CaptionFile]:
    """A miniature raw dump (one flat images/ folder): every drop reason once, plus English."""
    write_images(
        root / "images",
        [
            "coco2017_000000000001.jpg",
            "coco2017_000000000002.jpg",
            "vilc_000000000002.jpg",
            "vitextvqa_0.png",
            "vivqa_000000000003.jpg",
            "localization_0.png",
            "english_0.jpg",
        ],
    )
    (root / "images" / "notes.txt").write_text("not an image\n", encoding="utf-8")
    vi = {
        "10": {"image_id": 5, "caption": "không có ảnh"},  # invalid (sorted last)
        "0": {"image_id": "images/coco2017_000000000001.jpg", "caption": "  một   con mèo "},
        "1": {"image_id": "images/coco2017_000000000001.jpg", "caption": "một con mèo"},
        "2": {"image_id": "images/coco2017_000000000002.jpg", "caption": "ảnh test"},
        "3": {"image_id": "images/vilc_000000000002.jpg", "caption": "ảnh test"},
        "4": {"image_id": "images/vitextvqa_0.png", "caption": "biển hiệu"},
        "5": {"image_id": "images/localization_0.png", "caption": "   "},
        "6": {"image_id": "images/localization_0.png", "caption": "phố cổ"},
        "7": {"image_id": "images/oxford102_00001.jpg", "caption": "hoa"},  # image not on disk
        "8": {"image_id": "images/vivqa_000000000003.jpg", "caption": "quả bóng"},
    }
    (root / "vi.json").write_text(json.dumps(vi, ensure_ascii=False), encoding="utf-8")
    (root / "en.jsonl").write_text(
        json.dumps({"image": "images/english_0.jpg", "caption": "a cat"}) + "\n\n", encoding="utf-8"
    )
    return [
        CaptionFile(root / "vi.json", "json_dict", "image_id"),
        CaptionFile(root / "en.jsonl", "jsonl", "image"),
    ]


def test_build_corpus_decontaminates_cleans_and_reports(tmp_path: Path) -> None:
    files = _dump(tmp_path)
    out = tmp_path / "corpus" / "c.jsonl"
    report = build_corpus(tmp_path, files, out, exclude_coco_ids=[2], max_missing=0.5)

    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert lines == [
        {
            "image": "images/coco2017_000000000001.jpg",
            "caption": "một con mèo",
            "source": "coco2017",
        },
        {"image": "images/localization_0.png", "caption": "phố cổ", "source": "localization"},
        {"image": "images/vivqa_000000000003.jpg", "caption": "quả bóng", "source": "vivqa"},
        {"image": "images/english_0.jpg", "caption": "a cat", "source": "english"},
    ]
    dropped = {src: row["dropped"] for src, row in report["per_source"].items()}
    assert dropped == {
        "coco2017": {"excluded_coco_id": 1, "duplicate": 1},
        "localization": {"empty_caption": 1},
        "vilc": {"excluded_coco_id": 1},
        "vitextvqa": {"excluded_source": 1},
        "oxford102": {"missing_image": 1},
        "vivqa": {},
        "?": {"invalid_entry": 1},
        "english": {},
    }
    assert report["totals"]["captions_in"] == 11 and report["totals"]["captions_out"] == 4
    assert report["per_source"]["coco2017"]["images_in"] == 2
    assert report["per_source"]["coco2017"]["images_out"] == 1
    assert report["missing_images"]["examples"] == ["oxford102_00001.jpg"]
    assert report["output"]["lines"] == 4
    saved = json.loads((tmp_path / "corpus" / "c.report.json").read_text(encoding="utf-8"))
    assert saved["output"]["sha1"] == report["output"]["sha1"]
    assert saved["decontamination"]["excluded_sources"] == ["openvivqa", "viocrvqa", "vitextvqa"]


def test_build_corpus_without_decontamination_keeps_eval_images(tmp_path: Path) -> None:
    files = _dump(tmp_path)
    out = tmp_path / "c.jsonl"
    report = build_corpus(tmp_path, files, out, exclude_sources=(), max_missing=0.5)
    sources = [json.loads(x)["source"] for x in out.read_text(encoding="utf-8").splitlines()]
    assert sources.count("coco2017") == 2 and "vilc" in sources and "vitextvqa" in sources
    assert report["totals"]["captions_out"] == 7


def test_build_corpus_refuses_missing_images(tmp_path: Path) -> None:
    files = _dump(tmp_path)
    out = tmp_path / "c.jsonl"
    with pytest.raises(CorpusBuildError, match="fully extracted"):
        build_corpus(tmp_path, files, out, exclude_coco_ids=[2])
    assert not out.exists()
    report = json.loads((tmp_path / "c.report.json").read_text(encoding="utf-8"))
    assert report["output"] is None and report["missing_images"]["captions"] == 1


def test_build_corpus_finds_images_across_several_folders(tmp_path: Path) -> None:
    write_images(tmp_path / "moe" / "images", ["coco2017_000000000001.jpg"])
    write_images(tmp_path / "localization" / "images", ["localization_0.png"])
    vi = {
        "0": {"image_id": "images/coco2017_000000000001.jpg", "caption": "một con mèo"},
        "1": {"image_id": "images/localization_0.png", "caption": "phố cổ"},
    }
    (tmp_path / "vi.json").write_text(json.dumps(vi, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "c.jsonl"
    build_corpus(
        tmp_path,
        [CaptionFile(tmp_path / "vi.json", "json_dict", "image_id")],
        out,
        image_dirs=["moe/images", "localization/images"],
    )
    images = [json.loads(x)["image"] for x in out.read_text(encoding="utf-8").splitlines()]
    assert images == [
        "moe/images/coco2017_000000000001.jpg",
        "localization/images/localization_0.png",
    ]


def test_index_images_rejects_the_same_name_in_two_folders(tmp_path: Path) -> None:
    write_images(tmp_path / "a", ["x_1.jpg"])
    write_images(tmp_path / "b", ["x_1.jpg"])
    with pytest.raises(CorpusBuildError, match="several folders"):
        index_images(tmp_path, ["a", "b"])


def test_built_corpus_loads_with_the_jsonl_adapter(tmp_path: Path) -> None:
    files = _dump(tmp_path)
    build_corpus(
        tmp_path, files, tmp_path / "corpus" / "c.jsonl", exclude_coco_ids=[2], max_missing=0.5
    )
    cfg = DataSourceCfg(
        kind="jsonl",
        root=str(tmp_path),
        splits={"train": SplitSpec(file="corpus/c.jsonl", image_root=".")},
        image_check="decode",
        missing_image_policy="error",
    )
    recs = build_adapter(cfg).load_split("train")
    assert len(recs) == 4
    assert recs[0].image.key == "images/coco2017_000000000001.jpg" and recs[3].text == "a cat"


def test_uitviic_official_split_ids() -> None:
    names = [f"images/vilc_{i:012d}.jpg" for i in (1, 2, 3, 4, 5)] + [
        "images/coco2017_000000000009.jpg"
    ]
    out = uitviic_official_split_ids(["1", "2"], [5], names)
    assert out == {"test": ["5"], "official_dev": ["3", "4"]}
