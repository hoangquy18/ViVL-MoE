from __future__ import annotations

from pathlib import Path

import pytest

from vivlmoe.config.schema import HoldoutSpec
from vivlmoe.data.records import ImageRef, ImageTextRecord, ListRecordSet
from vivlmoe.data.splits import apply_holdout, holdout_image_ids, ids_sha1, load_ids, save_ids


def test_holdout_ids_deterministic_sorted_and_order_independent() -> None:
    ids = [f"img{i}" for i in range(50)]
    a = holdout_image_ids(ids, fraction=0.2, seed=0)
    b = holdout_image_ids(list(reversed(ids)) + ids[:5], fraction=0.2, seed=0)
    assert a == b == sorted(a) and len(a) == 10
    assert holdout_image_ids(ids, fraction=0.2, seed=1) != a
    assert len(holdout_image_ids(ids, num_images=7)) == 7
    with pytest.raises(ValueError):
        holdout_image_ids(ids, fraction=0.2, num_images=3)
    with pytest.raises(ValueError):
        holdout_image_ids(ids, fraction=1.0)
    with pytest.raises(ValueError):
        holdout_image_ids(["x"], fraction=0.5)


def test_save_load_and_sha1(tmp_path: Path) -> None:
    p = save_ids(tmp_path / "s" / "ids.json", ["b", "a", "a"])
    assert p.read_text(encoding="utf-8").startswith("[")
    assert load_ids(p) == ["a", "b"]
    assert ids_sha1(["b", "a"]) == ids_sha1(["a", "b"]) != ids_sha1(["a"])
    with pytest.raises(FileNotFoundError):
        load_ids(tmp_path / "nope.json")


def test_apply_holdout_splits(tmp_path: Path) -> None:
    recs = ListRecordSet(
        [ImageTextRecord(i, f"im{i // 2}", ImageRef(0, "x"), f"t{i}") for i in range(20)]
    )
    spec = HoldoutSpec(fraction=0.3, seed=0)
    dev, info = apply_holdout(spec, "dev", recs, tmp_path)
    train, _ = apply_holdout(spec, "train", recs)
    assert info is not None and len(set(dev.image_ids())) == 3 and len(train) + len(dev) == 20
    assert set(dev.image_ids()).isdisjoint(train.image_ids())
    assert [r.uid for r in train] == list(range(len(train)))
    assert (tmp_path / "splits" / "dev_image_ids.json").is_file()
    same, _ = apply_holdout(spec, "test", recs)
    assert len(same) == 20
    untouched, none = apply_holdout(None, "train", recs)
    assert none is None and len(untouched) == 20
