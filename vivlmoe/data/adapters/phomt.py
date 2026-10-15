"""PhoMT parallel corpus (Stage 1 teacher learning, Sec. 4.1).

Files: ``<root>/<variant>/<split>/<split>.{en,vi}`` line-aligned; ``variant`` is ``detokenization``
(raw text, the default in the configs) or ``tokenization`` (Moses-style spaces around punctuation,
which XLM-R's SentencePiece segments differently from raw captions). Train files are large (PhoMT:
2.98M pairs; the paper uses 2M), so lines are addressed through a byte-offset index
(:class:`LineOffsetIndex`, cached as ``<file>.offsets.npy``) and records materialise text lazily -
nothing calls ``readlines()`` on the whole file.
"""

from __future__ import annotations

import json
import logging
import os
from array import array
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal

import numpy as np

from vivlmoe.data.adapters.base import BaseAdapter
from vivlmoe.data.manifest import CheckLevel
from vivlmoe.data.records import ParallelTextRecord, RecordSet
from vivlmoe.data.registry import register_dataset

log = logging.getLogger(__name__)
_INDEX_VERSION = 1
_CHUNK = 8 << 20


class LineOffsetIndex:
    """Byte offsets of every line of a text file (handles ``\\r\\n`` and a missing final newline).

    The index is cached next to the file (``<name>.offsets.npy`` + ``.offsets.json`` with size and
    mtime) and rebuilt when stale. The file handle is opened lazily per process (picklable).
    """

    def __init__(self, path: str | Path, cache: bool = True) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"text file not found: {self.path}")
        self._fh: IO[bytes] | None = None
        st = self.path.stat()
        self._size, self._mtime_ns = st.st_size, st.st_mtime_ns
        offsets = self._load_cache() if cache else None
        if offsets is None:
            offsets = self._scan()
            if cache:
                self._save_cache(offsets)
        self._offsets = offsets

    # ------------------------------------------------------------------------------ cache
    @property
    def cache_paths(self) -> tuple[Path, Path]:
        base = self.path.with_name(self.path.name + ".offsets")
        return base.with_suffix(".offsets.npy"), base.with_suffix(".offsets.json")

    def _meta(self) -> dict[str, Any]:
        return {"version": _INDEX_VERSION, "size": self._size, "mtime_ns": self._mtime_ns}

    def _load_cache(self) -> np.ndarray | None:
        npy, meta = self.cache_paths
        if not (npy.is_file() and meta.is_file()):
            return None
        try:
            with meta.open("r", encoding="utf-8") as fh:
                if json.load(fh) != self._meta():
                    log.info("offset index for %s is stale; rebuilding", self.path.name)
                    return None
            arr = np.load(npy)
        except (OSError, ValueError) as exc:
            log.warning("offset index for %s unreadable (%s); rebuilding", self.path.name, exc)
            return None
        return arr.astype(np.int64, copy=False)

    def _save_cache(self, offsets: np.ndarray) -> None:
        npy, meta = self.cache_paths
        try:
            tmp = npy.with_name(
                f"{npy.stem}.tmp{os.getpid()}.npy"
            )  # np.save appends .npy otherwise
            np.save(tmp, offsets)
            os.replace(tmp, npy)
            with meta.open("w", encoding="utf-8") as fh:
                json.dump(self._meta(), fh)
        except OSError as exc:  # read-only data dir: still usable, just uncached
            log.warning("cannot write offset index cache for %s: %s", self.path, exc)

    def _scan(self) -> np.ndarray:
        starts = [np.zeros(1, dtype=np.int64)]
        pos = 0
        with self.path.open("rb") as fh:
            while True:
                chunk = fh.read(_CHUNK)
                if not chunk:
                    break
                nl = np.flatnonzero(np.frombuffer(chunk, dtype=np.uint8) == 10)
                if nl.size:
                    starts.append(nl.astype(np.int64) + (pos + 1))
                pos += len(chunk)
        offsets = np.concatenate(starts)
        if offsets[-1] == pos:  # file ends with a newline: no extra empty line
            offsets = offsets[:-1]
        return np.append(offsets, pos)  # [n+1]; offsets[n] == file size

    # ------------------------------------------------------------------------------ access
    def __len__(self) -> int:
        return int(self._offsets.size - 1)

    def _handle(self) -> IO[bytes]:
        if self._fh is None:
            self._fh = self.path.open("rb")
        return self._fh

    @staticmethod
    def _decode(raw: bytes) -> str:
        # PhoMT's dev.en starts with a UTF-8 byte-order mark
        return raw.rstrip(b"\r\n").decode("utf-8").lstrip("\ufeff")

    def line(self, i: int) -> str:
        if not 0 <= i < len(self):
            raise IndexError(i)
        start, end = int(self._offsets[i]), int(self._offsets[i + 1])
        fh = self._handle()
        fh.seek(start)
        return self._decode(fh.read(end - start))

    def iter_lines(self, start: int = 0) -> Iterator[str]:
        """Sequential streaming (no per-line seek)."""
        fh = self.path.open("rb")
        try:
            fh.seek(int(self._offsets[start]))
            for i in range(start, len(self)):
                yield self._decode(fh.read(int(self._offsets[i + 1] - self._offsets[i])))
        finally:
            fh.close()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_fh"] = None
        return state

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class PhoMTRecordSet(RecordSet):
    """Lazy parallel records: ``line_numbers[i]`` -> (src line, tgt line)."""

    record_type = ParallelTextRecord

    def __init__(
        self,
        src: LineOffsetIndex,
        tgt: LineOffsetIndex,
        line_numbers: np.ndarray,
        normalizer: Callable[[str], str] | None = None,
    ) -> None:
        self.src, self.tgt = src, tgt
        self.line_numbers = np.asarray(line_numbers, dtype=np.int64)
        self.normalizer = normalizer

    def __len__(self) -> int:
        return int(self.line_numbers.size)

    def __getitem__(self, i):  # type: ignore[override]
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        ln = int(self.line_numbers[i])
        tgt = self.tgt.line(ln)
        if self.normalizer is not None:
            tgt = self.normalizer(tgt)
        return ParallelTextRecord(uid=i, src=self.src.line(ln), tgt=tgt)

    def image_ids(self) -> list[str]:
        return [str(i) for i in range(len(self))]

    def groups(self) -> list[list[int]]:
        return [[i] for i in range(len(self))]

    def select(self, indices: Sequence[int]) -> PhoMTRecordSet:
        idx = np.asarray(list(indices), dtype=np.int64)
        return PhoMTRecordSet(self.src, self.tgt, self.line_numbers[idx], self.normalizer)

    def map_text(self, fn: Callable[[str], str]) -> PhoMTRecordSet:
        prev = self.normalizer
        composed = fn if prev is None else _Composed(prev, fn)
        return PhoMTRecordSet(self.src, self.tgt, self.line_numbers, composed)


class _Composed:
    """``second(first(text))`` as a picklable object (lambdas cannot cross spawn workers)."""

    def __init__(self, first: Callable[[str], str], second: Callable[[str], str]) -> None:
        self.first, self.second = first, second

    def __call__(self, text: str) -> str:
        return self.second(self.first(text))


@register_dataset("phomt")
class PhoMTAdapter(BaseAdapter):
    @dataclass
    class Options:
        variant: Literal["tokenization", "detokenization"] = "tokenization"
        src_lang: str = "en"  # CLIP teacher side
        tgt_lang: str = "vi"  # XLM-R student side
        max_train_samples: int | None = None  # paper: 2M of PhoMT's 2.98M train pairs (train only)
        subset_strategy: Literal["head", "random"] = "head"
        subset_seed: int = 0
        min_chars: int = 1  # pair filter: both sides at least this long after stripping
        max_chars: int | None = 2000
        index_cache: bool = True

    def _image_check(self) -> CheckLevel:
        return "none"

    def image_store(self, split: str) -> Any:
        raise TypeError("phomt is a text-only dataset and has no image store")

    def split_stem(self, split: str) -> Path:
        spec = self.cfg.splits.get(split)
        stem = spec.file if spec is not None else f"{self.options.variant}/{split}/{split}"
        return self.root / stem

    def _keep(self, src: str, tgt: str) -> bool:
        o = self.options
        s, t = src.strip(), tgt.strip()
        if len(s) < o.min_chars or len(t) < o.min_chars:
            return False
        return o.max_chars is None or (len(s) <= o.max_chars and len(t) <= o.max_chars)

    def _raw_records(self, split: str) -> RecordSet:
        o = self.options
        stem = self.split_stem(split)
        src = LineOffsetIndex(f"{stem}.{o.src_lang}", cache=o.index_cache)
        tgt = LineOffsetIndex(f"{stem}.{o.tgt_lang}", cache=o.index_cache)
        if len(src) != len(tgt):
            raise ValueError(
                f"phomt/{split}: {src.path.name} has {len(src)} lines but {tgt.path.name} has {len(tgt)}"
            )
        limit = o.max_train_samples if split == "train" else None
        kept = array("q")  # compact int64 buffer: 20M kept lines cost 160 MB, not a list of ints
        if limit is not None and o.subset_strategy == "random":
            order = np.random.default_rng(o.subset_seed).permutation(len(src))
            for ln in order:
                if self._keep(src.line(int(ln)), tgt.line(int(ln))):
                    kept.append(int(ln))
                    if len(kept) >= limit:
                        break
        else:
            for ln, (s, t) in enumerate(zip(src.iter_lines(), tgt.iter_lines(), strict=True)):
                if self._keep(s, t):
                    kept.append(ln)
                    if limit is not None and len(kept) >= limit:
                        break
        if limit is not None and len(kept) < limit:
            log.warning(
                "phomt/%s: only %d pairs pass the filter (max_train_samples=%d)",
                split,
                len(kept),
                limit,
            )
        log.info("phomt/%s: %d pairs selected from %d lines (%s)", split, len(kept), len(src), stem)
        lines = np.frombuffer(kept, dtype=np.int64) if kept else np.zeros(0, dtype=np.int64)
        return PhoMTRecordSet(src, tgt, np.sort(lines))


__all__ = ["LineOffsetIndex", "PhoMTAdapter", "PhoMTRecordSet"]
