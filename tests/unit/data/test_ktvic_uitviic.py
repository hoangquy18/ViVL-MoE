from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.data.conftest import write_images
from vivlmoe.config.schema import DataSourceCfg, HoldoutSpec
from vivlmoe.data.registry import build_adapter
from vivlmoe.data.splits import ids_sha1, load_ids


def make_ktvic(root: Path, n_images: int = 10, caps: int = 5) -> None:
    images = [{"id": i + 1, "filename": f"{i + 1:011d}.jpg"} for i in range(n_images)]
    anns = [
        {
            "id": i * caps + c,
            "image_id": i + 1,
            "caption": f"ảnh {i + 1} câu {c}",
            "segment_caption": f"ảnh_{i + 1} câu {c}",
        }
        for i in range(n_images)
        for c in range(caps)
    ]
    (root / "train_data.json").write_text(
        json.dumps({"images": images, "annotations": anns}), encoding="utf-8"
    )
    (root / "test_data.json").write_text(
        json.dumps({"images": images[:2], "annotations": anns[: 2 * caps]}), encoding="utf-8"
    )
    write_images(root / "train-images", [im["filename"] for im in images], size=20)
    write_images(root / "public-test-images", [im["filename"] for im in images[:2]], size=20)


def make_uitviic(root: Path) -> None:
    images = [{"id": 100 + i, "file_name": f"{100 + i:012d}.jpg"} for i in range(6)]
    anns = []
    for i in range(6):
        for c in range(i):  # image 100 has zero captions, image 105 has five
            anns.append({"image_id": 100 + i, "caption": f"chú thích {i}-{c}", "id": len(anns)})
    d = {"images": images, "annotations": anns}
    (root / "uitvic_captions_train2017.json").write_text(json.dumps(d), encoding="utf-8")
    (root / "uitvic_captions_test2017.json").write_text(json.dumps(d), encoding="utf-8")
    write_images(
        root / "coco_uitvic_train" / "coco_uitvic_train",
        [im["file_name"] for im in images],
        size=20,
    )
    write_images(
        root / "coco_uitvic_test" / "coco_uitvic_test", [im["file_name"] for im in images], size=20
    )


def test_ktvic_records_refs_and_holdout(tmp_path: Path) -> None:
    make_ktvic(tmp_path)
    hold = HoldoutSpec(fraction=0.2, seed=0)
    train = build_adapter(DataSourceCfg(kind="ktvic", root=str(tmp_path), holdout=hold)).load_split(
        "train"
    )
    dev_ad = build_adapter(
        DataSourceCfg(kind="ktvic", root=str(tmp_path), holdout=hold), run_dir=tmp_path / "run"
    )
    dev = dev_ad.load_split("dev")
    assert len(train) == 40 and len(dev) == 10  # 8 + 2 images x 5 captions
    assert all(len(r.refs) == 5 for r in train)
    assert set(train.image_ids()).isdisjoint(set(dev.image_ids()))
    assert [r.uid for r in dev] == list(range(10))
    dev2 = build_adapter(DataSourceCfg(kind="ktvic", root=str(tmp_path), holdout=hold)).load_split(
        "dev"
    )
    assert dev.image_ids() == dev2.image_ids()  # deterministic
    dev3 = build_adapter(
        DataSourceCfg(kind="ktvic", root=str(tmp_path), holdout=HoldoutSpec(fraction=0.2, seed=1))
    ).load_split("dev")
    assert dev.image_ids() != dev3.image_ids()
    info = dev_ad.holdout_info["dev"]
    assert info is not None and info.sha1 == ids_sha1(set(dev.image_ids()))
    saved = tmp_path / "run" / "splits" / "dev_image_ids.json"
    assert load_ids(saved) == sorted(set(dev.image_ids()))
    # committed id file is authoritative
    from_file = build_adapter(
        DataSourceCfg(kind="ktvic", root=str(tmp_path), holdout=HoldoutSpec(ids_file=str(saved)))
    ).load_split("dev")
    assert from_file.image_ids() == dev.image_ids()
    assert dev_ad.image_store("dev").root.name == "train-images"
    test = build_adapter(DataSourceCfg(kind="ktvic", root=str(tmp_path), holdout=hold)).load_split(
        "test"
    )
    assert len(test) == 10 and test[0].image.key == "00000000001.jpg"
    seg = build_adapter(
        DataSourceCfg(kind="ktvic", root=str(tmp_path), options={"caption_key": "segment_caption"})
    ).load_split("test")
    assert seg[0].text.startswith("ảnh_1")


def test_ktvic_holdout_id_file_mismatch_and_exclusion(tmp_path: Path) -> None:
    make_ktvic(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(["999"]), encoding="utf-8")
    with pytest.raises(ValueError, match="not in"):
        build_adapter(
            DataSourceCfg(kind="ktvic", root=str(tmp_path), holdout=HoldoutSpec(ids_file=str(bad)))
        ).load_split("dev")
    excl = tmp_path / "excl.json"
    excl.write_text(json.dumps(["1", "2", "3"]), encoding="utf-8")
    recs = build_adapter(
        DataSourceCfg(
            kind="ktvic",
            root=str(tmp_path),
            holdout=HoldoutSpec(fraction=0.2, exclude_ids_file=str(excl)),
        )
    ).load_split("train")
    assert set(recs.image_ids()).isdisjoint({"1", "2", "3"})
    assert len(recs) <= 35 and [r.uid for r in recs] == list(range(len(recs)))
    with pytest.raises(ValueError, match="no records left"):
        build_adapter(
            DataSourceCfg(
                kind="ktvic",
                root=str(tmp_path),
                holdout=HoldoutSpec(fraction=0.2, exclude_ids_file=str(excl)),
            )
        ).load_split("test")


def test_uitviic_variable_refs_and_zero_caption_drop(tmp_path: Path) -> None:
    make_uitviic(tmp_path)
    recs = build_adapter(DataSourceCfg(kind="uitviic", root=str(tmp_path))).load_split("train")
    assert len(recs) == 15  # 1+2+3+4+5
    assert "100" not in set(recs.image_ids())
    by_image = {
        iid: len(recs[g[0]].refs)
        for iid, g in zip(dict.fromkeys(recs.image_ids()), recs.groups(), strict=True)
    }
    assert by_image == {"101": 1, "102": 2, "103": 3, "104": 4, "105": 5}
    assert recs[0].image.key == "000000000101.jpg"
    test = build_adapter(DataSourceCfg(kind="uitviic", root=str(tmp_path))).load_split("test")
    assert len(test) == 15
