from __future__ import annotations

import pytest

from vivlmoe.data.records import (
    ColumnarImageText,
    ImageRef,
    ImageTextRecord,
    ListRecordSet,
    ParallelTextRecord,
    VQARecord,
    groups_from_ids,
    map_record_text,
)


def _rec(i: int, image_id: str, text: str) -> ImageTextRecord:
    return ImageTextRecord(
        uid=i, image_id=image_id, image=ImageRef(0, f"{image_id}.jpg"), text=text
    )


def test_list_recordset_select_renumbers_and_groups() -> None:
    rs = ListRecordSet([_rec(0, "a", "x"), _rec(1, "b", "y"), _rec(2, "a", "z")])
    assert rs.groups() == [[0, 2], [1]]
    sub = rs.select([2, 1])
    assert [r.uid for r in sub] == [0, 1]
    assert [r.text for r in sub] == ["z", "y"]
    assert sub.record_type is ImageTextRecord


def test_map_text_touches_vietnamese_fields_only() -> None:
    up = str.upper
    it = map_record_text(ImageTextRecord(0, "a", ImageRef(0, "a"), "t", refs=("r1", "r2")), up)
    assert (it.text, it.refs) == ("T", ("R1", "R2"))
    vq = map_record_text(VQARecord(0, "a", ImageRef(0, "a"), "q", ("a1", "a2")), up)
    assert (vq.question, vq.answers) == ("Q", ("A1", "A2"))
    pt = map_record_text(ParallelTextRecord(0, "english", "tiếng việt"), up)
    assert (pt.src, pt.tgt) == ("english", "TIẾNG VIỆT")


def test_columnar_roundtrip_select_and_map() -> None:
    keys = ["images/a.jpg", "images/b.jpg", "images/a.jpg"]
    texts = ["một", "hai", "ba"]
    col = ColumnarImageText(keys, texts)
    assert len(col) == 3
    assert col[2] == ImageTextRecord(2, "images/a.jpg", ImageRef(0, "images/a.jpg"), "ba")
    assert col[-1].uid == 2
    assert col.groups() == [[0, 2], [1]]
    sub = col.select([2, 0])
    assert [r.text for r in sub] == ["ba", "một"] and [r.uid for r in sub] == [0, 1]
    mapped = col.map_text(str.upper)
    assert [r.text for r in mapped] == ["MỘT", "HAI", "BA"]
    assert mapped[0].image_id == "images/a.jpg"
    with pytest.raises(IndexError):
        col[3]


def test_columnar_concat_offsets_sources_by_store_count() -> None:
    a = ColumnarImageText(["a1", "a2"], ["t1", "t2"])
    b = ColumnarImageText(["b1"], ["t3"], sources=[1])  # b already spans two stores
    c = ColumnarImageText(["c1"], ["t4"])
    cat = ColumnarImageText.concat([a, b, c], store_counts=[1, 2, 1])
    assert [r.image.source for r in cat] == [0, 0, 2, 3]
    assert [r.image.key for r in cat] == ["a1", "a2", "b1", "c1"]
    with pytest.raises(ValueError):
        ColumnarImageText.concat([a, b], store_counts=[1, 1])


def test_records_pickle_for_spawn_workers() -> None:
    import pickle

    from vivlmoe.data.records import FrozenDict

    r = ImageTextRecord(0, "a", ImageRef(0, "a.jpg"), "t")
    assert pickle.loads(pickle.dumps(r)) == r
    v = pickle.loads(
        pickle.dumps(VQARecord(0, "a", ImageRef(0, "a.jpg"), "q", ("a",), meta={"id": "1"}))
    )
    assert v.meta["id"] == "1"
    with pytest.raises(TypeError):
        r.meta["x"] = 1  # type: ignore[index]
    fd = pickle.loads(pickle.dumps(FrozenDict({"k": 2})))
    assert fd == {"k": 2} and isinstance(fd, FrozenDict)


def test_columnar_groups_numpy_matches_reference_and_is_not_cached() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    ids = [f"img/{int(k):03d}.jpg" for k in rng.integers(0, 40, size=200)] + ["", "x", ""]
    col = ColumnarImageText([f"k{i}" for i in range(len(ids))], ["t"] * len(ids), image_ids=ids)
    expect: dict[str, list[int]] = {}
    for i, iid in enumerate(ids):
        expect.setdefault(iid, []).append(i)
    assert col.groups() == list(expect.values())
    assert col.groups() is not col.groups()  # recomputed, never cached on the record set
    assert "_groups" not in col.__dict__
    assert groups_from_ids(np.asarray([], dtype="S1")) == []
    assert ColumnarImageText([], []).groups() == []
    assert ColumnarImageText(["a", "b"], ["", ""], image_ids=["", ""]).groups() == [[0, 1]]
