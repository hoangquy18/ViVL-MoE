from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import numpy as np
import pytest

from vivlmoe.config.schema import DataSourceCfg
from vivlmoe.data.adapters.phomt import LineOffsetIndex, PhoMTRecordSet
from vivlmoe.data.registry import build_adapter


def test_line_offset_index_handles_crlf_and_missing_final_newline(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_bytes("một\r\nhai\n\nbốn".encode())
    idx = LineOffsetIndex(p, cache=False)
    assert len(idx) == 4
    assert [idx.line(i) for i in range(4)] == ["một", "hai", "", "bốn"]
    assert list(idx.iter_lines()) == ["một", "hai", "", "bốn"]
    assert list(idx.iter_lines(start=2)) == ["", "bốn"]
    p2 = tmp_path / "y.txt"
    p2.write_bytes(b"a\nb\n")
    assert len(LineOffsetIndex(p2, cache=False)) == 2
    assert (
        len(LineOffsetIndex(tmp_path / "e.txt", cache=False)) == 0
        if (tmp_path / "e.txt").write_bytes(b"") == 0
        else True
    )
    with pytest.raises(IndexError):
        idx.line(4)


def test_line_offset_index_cache_reuse_and_staleness(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "z.txt"
    p.write_text("a\nb\nc\n", encoding="utf-8")
    calls = {"scan": 0}
    orig = LineOffsetIndex._scan

    def counting(self):
        calls["scan"] += 1
        return orig(self)

    monkeypatch.setattr(LineOffsetIndex, "_scan", counting)
    idx = LineOffsetIndex(p)
    npy, meta = idx.cache_paths
    assert npy.is_file() and meta.is_file() and calls["scan"] == 1
    LineOffsetIndex(p)
    assert calls["scan"] == 1  # cache hit
    time.sleep(0.01)
    p.write_text("a\nb\nc\nd\n", encoding="utf-8")  # size changes -> stale
    idx3 = LineOffsetIndex(p)
    assert calls["scan"] == 2 and len(idx3) == 4
    # same size, different mtime -> stale too
    p.write_text("a\nb\nc\ne\n", encoding="utf-8")
    os.utime(p, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns + 5_000_000))
    LineOffsetIndex(p)
    assert calls["scan"] == 3
    # corrupt cache -> rebuilt
    npy.write_bytes(b"garbage")
    LineOffsetIndex(p)
    assert calls["scan"] == 4


def test_line_offset_index_is_picklable(tmp_path: Path) -> None:
    p = tmp_path / "p.txt"
    p.write_text("x\ny\n", encoding="utf-8")
    idx = LineOffsetIndex(p, cache=False)
    assert idx.line(1) == "y"  # opens the handle
    clone = pickle.loads(pickle.dumps(idx))
    assert clone._fh is None and clone.line(0) == "x"
    rs = PhoMTRecordSet(idx, idx, np.array([1, 0]))
    clone_rs = pickle.loads(pickle.dumps(rs))
    assert clone_rs[0].src == "y" and clone_rs[1].uid == 1
    from vivlmoe.data.text_normalize import build_normalizer

    composed = rs.map_text(build_normalizer("pretrain")).map_text(str.upper)
    assert pickle.loads(pickle.dumps(composed))[0].tgt == "Y"


def test_line_index_strips_a_leading_byte_order_mark(tmp_path: Path) -> None:
    p = tmp_path / "dev.en"
    p.write_bytes("\ufeffHurricane Dorian\nsecond\n".encode())
    idx = LineOffsetIndex(p, cache=False)
    assert idx.line(0) == "Hurricane Dorian"
    assert list(idx.iter_lines()) == ["Hurricane Dorian", "second"]


def _write_phomt(root: Path, en: list[str], vi: list[str], split: str = "train") -> None:
    d = root / "tokenization" / split
    d.mkdir(parents=True)
    (d / f"{split}.en").write_text("\n".join(en) + "\n", encoding="utf-8")
    (d / f"{split}.vi").write_text("\n".join(vi) + "\n", encoding="utf-8")


def test_phomt_adapter_filters_subsets_and_normalizes(tmp_path: Path) -> None:
    en = [f"english {i}" for i in range(10)]
    vi = [f"hoà {i}" for i in range(10)]
    en[3], vi[7] = "", "   "  # broken pairs
    _write_phomt(tmp_path, en, vi)
    _write_phomt(tmp_path, ["dev en"], ["dev vi"], "dev")
    cfg = DataSourceCfg(
        kind="phomt",
        root=str(tmp_path),
        text_normalize="pretrain",
        options={"max_train_samples": 4},
    )
    recs = build_adapter(cfg).load_split("train")
    assert len(recs) == 4
    assert [r.uid for r in recs] == [0, 1, 2, 3]
    assert recs[3].src == "english 4" and recs[3].tgt == "hòa 4"  # 3 skipped, tone repositioned
    assert isinstance(recs, PhoMTRecordSet) and recs.groups() == [[0], [1], [2], [3]]
    rnd = build_adapter(
        DataSourceCfg(
            kind="phomt",
            root=str(tmp_path),
            options={"max_train_samples": 4, "subset_strategy": "random", "subset_seed": 1},
        )
    )
    r1 = rnd.load_split("train")
    r2 = build_adapter(
        DataSourceCfg(
            kind="phomt",
            root=str(tmp_path),
            options={"max_train_samples": 4, "subset_strategy": "random", "subset_seed": 1},
        )
    ).load_split("train")
    assert len(r1) == 4 and r1.line_numbers.tolist() == r2.line_numbers.tolist()
    assert 3 not in r1.line_numbers and 7 not in r1.line_numbers
    dev = build_adapter(
        DataSourceCfg(kind="phomt", root=str(tmp_path), options={"max_train_samples": 4})
    ).load_split("dev")
    assert len(dev) == 1 and dev[0].tgt == "dev vi"  # limit applies to train only
    full = build_adapter(DataSourceCfg(kind="phomt", root=str(tmp_path), limit=2)).load_split(
        "train"
    )
    assert len(full) == 2
    with pytest.raises(TypeError):
        build_adapter(cfg).image_store("train")


def test_phomt_length_mismatch_raises(tmp_path: Path) -> None:
    _write_phomt(tmp_path, ["a", "b"], ["x"])
    with pytest.raises(ValueError, match="lines"):
        build_adapter(DataSourceCfg(kind="phomt", root=str(tmp_path))).load_split("train")
