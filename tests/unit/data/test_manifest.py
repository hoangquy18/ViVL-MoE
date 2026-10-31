from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.data.conftest import write_images
from vivlmoe.data.image_store import ImageLoadError, LocalImageStore
from vivlmoe.data.manifest import check_image_keys, validate_records, write_dropped
from vivlmoe.data.records import (
    ColumnarImageText,
    ImageRef,
    ListRecordSet,
    MultiImageClsRecord,
)


def _corpus() -> ColumnarImageText:
    return ColumnarImageText(["ok.jpg", "missing.jpg", "bad.jpg", "ok.jpg"], ["a", "b", "c", "d"])


@pytest.fixture()
def store(tmp_path: Path) -> LocalImageStore:
    write_images(tmp_path, ["ok.jpg"])
    (tmp_path / "bad.jpg").write_bytes(b"\xff\xd8junk")
    return LocalImageStore(tmp_path)


def test_levels_and_policies(store: LocalImageStore) -> None:
    recs, dropped = validate_records(_corpus(), store, level="none")
    assert len(recs) == 4 and dropped == []
    recs, dropped = validate_records(_corpus(), store, level="exists", policy="skip")
    assert [r.text for r in recs] == ["a", "c", "d"] and [r.uid for r in recs] == [0, 1, 2]
    assert [(d.uid, d.image_key, d.reason) for d in dropped] == [(1, "missing.jpg", "missing")]
    recs, dropped = validate_records(_corpus(), store, level="decode", policy="skip")
    assert [r.text for r in recs] == ["a", "d"] and dropped[1].reason.startswith("undecodable")
    with pytest.raises(ImageLoadError, match="missing"):
        validate_records(_corpus(), store, level="exists", policy="error")


def test_cache_hit_and_staleness(store: LocalImageStore, tmp_path: Path, monkeypatch) -> None:
    import vivlmoe.data.manifest as m

    calls = {"n": 0}
    orig = m.check_image_keys

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(m, "check_image_keys", counting)
    cache = tmp_path / "cache" / "m.json.gz"
    validate_records(_corpus(), store, "exists", "skip", cache_path=cache)
    assert cache.is_file() and calls["n"] == 1
    validate_records(_corpus(), store, "exists", "skip", cache_path=cache)
    assert calls["n"] == 1  # hit
    validate_records(_corpus(), store, "decode", "skip", cache_path=cache)
    assert calls["n"] == 2  # different level -> stale
    validate_records(_corpus().select([0, 1]), store, "decode", "skip", cache_path=cache)
    assert calls["n"] == 3  # different key count -> stale
    cache.write_bytes(b"broken")
    validate_records(_corpus().select([0, 1]), store, "decode", "skip", cache_path=cache)
    assert calls["n"] == 4


def test_multi_image_records_lose_only_bad_images(store: LocalImageStore, tmp_path: Path) -> None:
    recs = ListRecordSet(
        [
            MultiImageClsRecord(
                0, "s0", "t", (ImageRef(0, "ok.jpg"), ImageRef(0, "missing.jpg")), (), (0,) * 6
            ),
            MultiImageClsRecord(1, "s1", "t", (ImageRef(0, "missing.jpg"),), (), (0,) * 6),
        ]
    )
    out, dropped = validate_records(recs, store, "exists", "skip")
    assert len(out) == 2 and out[0].images == (ImageRef(0, "ok.jpg"),) and out[1].images == ()
    assert len(dropped) == 2
    p = write_dropped(tmp_path / "d.json", dropped)
    assert p.read_text().count("missing.jpg") == 2
    with pytest.raises(ImageLoadError):
        validate_records(recs, store, "exists", "error")


def test_parallel_image_check_finds_the_same_bad_images(tmp_path: Path) -> None:
    names = [f"img_{i}.jpg" for i in range(10)]
    write_images(tmp_path, names)
    (tmp_path / "img_3.jpg").write_bytes(b"\xff\xd8junk")  # corrupt
    (tmp_path / "img_7.jpg").unlink()  # missing
    store = LocalImageStore(tmp_path)
    refs = [ImageRef(source=0, key=n) for n in names + names[:4]]  # duplicates too
    serial = check_image_keys(store, refs, "decode")
    assert sorted(serial) == ["0:img_3.jpg", "0:img_7.jpg"]
    assert check_image_keys(store, refs, "decode", workers=2, chunk=3) == serial
