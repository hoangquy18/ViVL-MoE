from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.data.conftest import write_images
from vivlmoe.config.schema import DataSourceCfg, SplitSpec
from vivlmoe.data.image_store import CompositeImageStore, ImageLoadError
from vivlmoe.data.records import ColumnarImageText
from vivlmoe.data.registry import build_adapter


def _json_dict(root: Path, entries: dict[str, dict], name: str = "data.json") -> None:
    (root / name).write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")


def test_json_dict_corpus_sorted_by_index_with_gaps(tmp_path: Path) -> None:
    write_images(tmp_path / "images", ["a.jpg", "b.jpg"])
    _json_dict(
        tmp_path,
        {
            "10": {"image_id": "images/b.jpg", "caption": "hai"},
            "2": {"image_id": "images/a.jpg", "caption": "một"},
        },
    )
    cfg = DataSourceCfg(
        kind="json_dict", root=str(tmp_path), splits={"train": SplitSpec(file="data.json")}
    )
    recs = build_adapter(cfg).load_split("train")
    assert isinstance(recs, ColumnarImageText)
    assert [r.text for r in recs] == ["một", "hai"]
    assert recs[0].image.key == "images/a.jpg" and recs[0].image_id == "images/a.jpg"


def test_jsonl_corpus_and_invalid_entries(tmp_path: Path) -> None:
    write_images(tmp_path / "images", ["a.jpg"])
    lines = [
        json.dumps({"image": "images/a.jpg", "caption": "một"}),
        "",
        "not json",
        json.dumps({"image": "images/a.jpg"}),
    ]
    (tmp_path / "c.jsonl").write_text("\n".join(lines), encoding="utf-8")
    cfg = DataSourceCfg(
        kind="jsonl", root=str(tmp_path), splits={"train": SplitSpec(file="c.jsonl")}
    )
    with pytest.raises(ValueError):
        build_adapter(cfg).load_split("train")
    cfg.options = {"skip_invalid": True}
    recs = build_adapter(cfg).load_split("train")
    assert len(recs) == 1 and recs[0].text == "một"


def test_missing_image_policies(tmp_path: Path) -> None:
    write_images(tmp_path / "images", ["a.jpg"])
    (tmp_path / "images" / "bad.jpg").write_bytes(b"not an image")
    _json_dict(
        tmp_path,
        {
            "0": {"image_id": "images/a.jpg", "caption": "ok"},
            "1": {"image_id": "images/missing.jpg", "caption": "gone"},
            "2": {"image_id": "images/bad.jpg", "caption": "corrupt"},
        },
    )
    base = dict(kind="json_dict", root=str(tmp_path), splits={"train": SplitSpec(file="data.json")})
    with pytest.raises(ImageLoadError):
        build_adapter(DataSourceCfg(**base, missing_image_policy="error")).load_split("train")
    ad = build_adapter(DataSourceCfg(**base, missing_image_policy="skip", image_check="exists"))
    recs = ad.load_split("train")
    assert [r.text for r in recs] == ["ok", "corrupt"] and [r.uid for r in recs] == [0, 1]
    assert [d.image_key for d in ad.dropped["train"]] == ["images/missing.jpg"]
    ad = build_adapter(
        DataSourceCfg(**base, missing_image_policy="skip", image_check="decode"),
        run_dir=tmp_path / "run",
    )
    recs = ad.load_split("train")
    assert [r.text for r in recs] == ["ok"]
    out = ad.write_dropped("train")
    assert out is not None and len(json.loads(out.read_text())) == 2
    recs = build_adapter(DataSourceCfg(**base, image_check="none")).load_split("train")
    assert len(recs) == 3


def test_mixture_offsets_sources_and_composite_store(tmp_path: Path, tiny_transform) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    write_images(a / "img", ["1.jpg", "2.jpg"], seed=1)
    write_images(b / "pics", ["x.jpg"], seed=2)
    _json_dict(
        a,
        {
            "0": {"image_id": "img/1.jpg", "caption": "a1"},
            "1": {"image_id": "img/2.jpg", "caption": "a2"},
        },
    )
    (b / "c.jsonl").write_text(
        json.dumps({"image": "pics/x.jpg", "caption": "b1"}) + "\n", encoding="utf-8"
    )
    cfg = DataSourceCfg(
        kind="mixture",
        options={
            "sources": [
                {"kind": "json_dict", "root": str(a), "splits": {"train": {"file": "data.json"}}},
                {
                    "kind": "jsonl",
                    "root": str(b),
                    "splits": {"train": {"file": "c.jsonl"}},
                    "limit": 1,
                },
            ]
        },
    )
    ad = build_adapter(cfg)
    recs = ad.load_split("train")
    assert [r.image.source for r in recs] == [0, 0, 1]
    assert [r.uid for r in recs] == [0, 1, 2]
    store = ad.image_store("train")
    assert isinstance(store, CompositeImageStore) and len(store.stores) == 2
    from vivlmoe.data.datasets import ImageTextDataset

    ds = ImageTextDataset(recs, store, tiny_transform)
    assert ds[2]["pixel_values"].shape == (3, 32, 32) and ds[2]["text"] == "b1"
    with pytest.raises(ValueError):
        build_adapter(DataSourceCfg(kind="mixture", options={"sources": []}))


def test_hf_image_caption_adapter(tmp_path: Path, tiny_transform) -> None:
    datasets = pytest.importorskip("datasets")
    write_images(tmp_path / "raw", ["0.png", "1.png"], size=24)
    ds = datasets.Dataset.from_dict(
        {
            "image": [str(tmp_path / "raw" / "0.png"), str(tmp_path / "raw" / "1.png")],
            "caption": ["một", "hai"],
        }
    )
    ds = ds.cast_column("image", datasets.Image())
    ds.save_to_disk(str(tmp_path / "hfds"))
    cfg = DataSourceCfg(kind="hf_image_caption", options={"dataset": str(tmp_path / "hfds")})
    ad = build_adapter(cfg)
    recs = ad.load_split("train")
    assert len(recs) == 2 and recs[1].image_id == "hf:1" and recs[1].image.key == "1"
    store = ad.image_store("train")
    img = store.open(recs[0].image)
    assert img.size == (32, 24)
    import pickle

    store2 = pickle.loads(pickle.dumps(store))
    assert store2._ds is None and store2.open(recs[1].image).mode == "RGB"
    from vivlmoe.data.datasets import ImageTextDataset

    assert ImageTextDataset(recs, store, tiny_transform)[1]["pixel_values"].shape == (3, 32, 32)
    with pytest.raises(KeyError):
        build_adapter(
            DataSourceCfg(
                kind="hf_image_caption",
                options={"dataset": str(tmp_path / "hfds"), "caption_column": "nope"},
            )
        ).load_split("train")
