"""Common, immutable sample schema shared by adapters, datasets, collators, evaluators and dumps.

``uid`` is the dense 0..N-1 index within a split (used to gather predictions in order across ranks).
``image_id`` is the grouping key (one caption per image per batch; holdouts by image id).
Records never hold tensors so DataLoader workers share them copy-on-write.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np


class FrozenDict(dict):
    """Read-only dict that pickles (``MappingProxyType`` does not, which breaks spawn workers)."""

    __slots__ = ()

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("record meta is immutable")

    __setitem__ = __delitem__ = update = pop = popitem = clear = setdefault = _immutable  # type: ignore[assignment]

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        return (FrozenDict, (dict(self),))


_EMPTY: Any = FrozenDict()


@dataclass(frozen=True, slots=True)
class ImageRef:
    source: int  # index into CompositeImageStore.stores (0 for single-source datasets)
    key: str | int  # relative path (LocalImageStore) or row index (HFImageStore)


@dataclass(frozen=True, slots=True)
class ImageTextRecord:
    uid: int
    image_id: str
    image: ImageRef
    text: str  # the caption of this example ('' for eval image-side records)
    refs: tuple[str, ...] = ()  # all references of the image (captioning / retrieval eval)
    meta: Any = field(default_factory=lambda: _EMPTY)


@dataclass(frozen=True, slots=True)
class VQARecord:
    uid: int
    image_id: str
    image: ImageRef
    question: str
    answers: tuple[str, ...]  # >= 1 gold answers; answers[0] is the training target
    meta: Any = field(default_factory=lambda: _EMPTY)


@dataclass(frozen=True, slots=True)
class MultiImageClsRecord:
    uid: int
    sample_id: str
    text: str
    images: tuple[ImageRef, ...]  # 0..max_images
    image_tags: tuple[str, ...]  # union of image aspect tags, sorted; () if none
    labels: tuple[int, ...]  # one label per aspect: 0 None, 1 Negative, 2 Neutral, 3 Positive
    meta: Any = field(default_factory=lambda: _EMPTY)


@dataclass(frozen=True, slots=True)
class ParallelTextRecord:
    uid: int
    src: str  # English (CLIP teacher)
    tgt: str  # Vietnamese (XLM-R student)


Record = ImageTextRecord | VQARecord | MultiImageClsRecord | ParallelTextRecord


class RecordSet(Sequence[Record]):
    """Abstract sequence of records with image-id grouping."""

    record_type: type

    def __len__(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def __getitem__(self, i):  # type: ignore[override]  # pragma: no cover - abstract
        raise NotImplementedError

    def __iter__(self) -> Iterator[Record]:
        for i in range(len(self)):
            yield self[i]

    def image_ids(self) -> list[str]:
        return [getattr(r, "image_id", str(getattr(r, "sample_id", i))) for i, r in enumerate(self)]

    def groups(self) -> list[list[int]]:
        """Indices grouped by image_id, in first-occurrence order (singletons for unique images)."""
        order: dict[str, list[int]] = {}
        for i, image_id in enumerate(self.image_ids()):
            order.setdefault(image_id, []).append(i)
        return list(order.values())

    def select(self, indices: Sequence[int]) -> RecordSet:
        """Subset in the given order with ``uid`` renumbered to 0..len-1."""
        raise NotImplementedError

    def map_text(self, fn: Callable[[str], str]) -> RecordSet:
        """Apply ``fn`` to every Vietnamese text field (text/refs/question/answers/tgt)."""
        raise NotImplementedError


def with_uid(record: Record, uid: int) -> Record:
    return replace(record, uid=uid)


def map_record_text(record: Record, fn: Callable[[str], str]) -> Record:
    """Return ``record`` with ``fn`` applied to its Vietnamese text fields."""
    if isinstance(record, ImageTextRecord):
        return replace(record, text=fn(record.text), refs=tuple(fn(r) for r in record.refs))
    if isinstance(record, VQARecord):
        return replace(
            record, question=fn(record.question), answers=tuple(fn(a) for a in record.answers)
        )
    if isinstance(record, MultiImageClsRecord):
        return replace(record, text=fn(record.text))
    if isinstance(record, ParallelTextRecord):
        return replace(record, tgt=fn(record.tgt))
    raise TypeError(f"unsupported record type {type(record).__name__}")


class ListRecordSet(RecordSet):
    def __init__(self, records: Sequence[Record]) -> None:
        self._records = list(records)
        self.record_type = type(self._records[0]) if self._records else ImageTextRecord

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, i):  # type: ignore[override]
        return self._records[i]

    def map(self, fn: Callable[[Record], Record]) -> ListRecordSet:
        return ListRecordSet([fn(r) for r in self._records])

    def select(self, indices: Sequence[int]) -> ListRecordSet:
        out = ListRecordSet([with_uid(self._records[int(i)], k) for k, i in enumerate(indices)])
        out.record_type = self.record_type
        return out

    def map_text(self, fn: Callable[[str], str]) -> ListRecordSet:
        out = ListRecordSet([map_record_text(r, fn) for r in self._records])
        out.record_type = self.record_type
        return out


class ColumnarImageText(RecordSet):
    """Memory-lean storage for ~1M image-text pairs: UTF-8 byte buffers + int64 offsets.

    Strings are materialised only in ``__getitem__`` so DataLoader workers do not duplicate
    millions of Python objects.
    """

    record_type = ImageTextRecord

    def __init__(
        self,
        image_keys: Sequence[str],
        texts: Sequence[str],
        image_ids: Sequence[str] | None = None,
        sources: Sequence[int] | None = None,
    ) -> None:
        n = len(image_keys)
        if len(texts) != n:
            raise ValueError("image_keys and texts must have the same length")
        self._keys_buf, self._keys_off = _pack(image_keys)
        self._text_buf, self._text_off = _pack(texts)
        if image_ids is None or image_ids is image_keys:
            self._ids_buf, self._ids_off = self._keys_buf, self._keys_off
        else:
            if len(image_ids) != n:
                raise ValueError("image_ids must have the same length as image_keys")
            self._ids_buf, self._ids_off = _pack(image_ids)
        self._sources = (
            np.zeros(n, dtype=np.int16) if sources is None else np.asarray(sources, dtype=np.int16)
        )
        self._n = n

    def __len__(self) -> int:
        return self._n

    def _get(self, buf: np.ndarray, off: np.ndarray, i: int) -> str:
        return buf[off[i] : off[i + 1]].tobytes().decode("utf-8")

    def __getitem__(self, i):  # type: ignore[override]
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError(i)
        return ImageTextRecord(
            uid=i,
            image_id=self._get(self._ids_buf, self._ids_off, i),
            image=ImageRef(int(self._sources[i]), self._get(self._keys_buf, self._keys_off, i)),
            text=self._get(self._text_buf, self._text_off, i),
        )

    def image_ids(self) -> list[str]:
        return [self._get(self._ids_buf, self._ids_off, i) for i in range(self._n)]

    def groups(self) -> list[list[int]]:
        """Image groups computed in numpy from the id byte buffer (never cached on the record set,
        which is pickled into every spawn worker)."""
        return groups_from_ids(_fixed_width(self._ids_buf, self._ids_off))

    def select(self, indices: Sequence[int]) -> ColumnarImageText:
        idx = np.asarray(indices, dtype=np.int64)
        return ColumnarImageText(
            [self._get(self._keys_buf, self._keys_off, int(i)) for i in idx],
            [self._get(self._text_buf, self._text_off, int(i)) for i in idx],
            [self._get(self._ids_buf, self._ids_off, int(i)) for i in idx],
            self._sources[idx],
        )

    def map_text(self, fn: Callable[[str], str]) -> ColumnarImageText:
        n = self._n
        out = ColumnarImageText.__new__(ColumnarImageText)
        out._keys_buf, out._keys_off = self._keys_buf, self._keys_off
        out._ids_buf, out._ids_off = self._ids_buf, self._ids_off
        out._text_buf, out._text_off = _pack(
            [fn(self._get(self._text_buf, self._text_off, i)) for i in range(n)]
        )
        out._sources = self._sources
        out._n = n
        return out

    def sources(self) -> np.ndarray:
        return self._sources

    @staticmethod
    def concat(
        parts: Sequence[ColumnarImageText], store_counts: Sequence[int] | None = None
    ) -> ColumnarImageText:
        """Concatenate corpora; ``ImageRef.source`` of part *p* is offset by the number of image
        stores of the preceding parts (``store_counts``, default 1 each), so a
        :class:`CompositeImageStore` over the parts' stores in the same order resolves every ref."""
        counts = [1] * len(parts) if store_counts is None else list(store_counts)
        if len(counts) != len(parts):
            raise ValueError("store_counts must have one entry per part")
        keys: list[str] = []
        texts: list[str] = []
        ids: list[str] = []
        sources: list[np.ndarray] = []
        offset = 0
        for part, count in zip(parts, counts, strict=True):
            n = len(part)
            if n and int(part._sources.max()) >= count:
                raise ValueError(
                    f"part references source {int(part._sources.max())} >= {count} stores"
                )
            keys.extend(part._get(part._keys_buf, part._keys_off, i) for i in range(n))
            texts.extend(part._get(part._text_buf, part._text_off, i) for i in range(n))
            ids.extend(part._get(part._ids_buf, part._ids_off, i) for i in range(n))
            sources.append(part._sources.astype(np.int16) + offset)
            offset += count
        src = np.concatenate(sources) if sources else np.zeros(0, dtype=np.int16)
        return ColumnarImageText(keys, texts, ids, src)


def _fixed_width(buf: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Packed UTF-8 strings -> fixed-width ``S<max_len>`` array (vectorised scatter, no Python
    strings); transient memory is ``n * max_len`` bytes."""
    n = off.size - 1
    lengths = np.diff(off)
    width = int(lengths.max()) if n else 0
    if width == 0:
        return np.zeros(n, dtype="S1")
    grid = np.zeros((n, width), dtype=np.uint8)
    rows = np.repeat(np.arange(n), lengths)
    cols = np.arange(int(off[-1])) - np.repeat(off[:-1], lengths)
    grid[rows, cols] = buf[: int(off[-1])]
    return grid.view(f"S{width}").ravel()


def groups_from_ids(ids: np.ndarray) -> list[list[int]]:
    """``np.unique(return_inverse)`` grouping: indices per distinct id, groups in first-occurrence
    order and indices ascending inside each group."""
    n = ids.size
    if n == 0:
        return []
    _, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(inverse, kind="stable")  # members of group 0, then 1, ... (ascending)
    bounds = np.cumsum(np.bincount(inverse))[:-1]
    members = np.split(order, bounds)
    return [members[g].tolist() for g in np.argsort(first, kind="stable")]


def _pack(strings: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    encoded = [s.encode("utf-8") for s in strings]
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    if encoded:
        offsets[1:] = np.cumsum([len(b) for b in encoded])
    buf = (
        np.frombuffer(b"".join(encoded), dtype=np.uint8) if encoded else np.zeros(0, dtype=np.uint8)
    )
    return buf, offsets


__all__ = [
    "ImageRef",
    "ImageTextRecord",
    "VQARecord",
    "MultiImageClsRecord",
    "ParallelTextRecord",
    "Record",
    "RecordSet",
    "ListRecordSet",
    "ColumnarImageText",
    "FrozenDict",
    "field",
    "groups_from_ids",
    "map_record_text",
    "with_uid",
]
