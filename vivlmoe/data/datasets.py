"""Task-generic ``torch`` datasets over records. Items are plain dicts; tokenisation happens in the
collators. Image failures raise :class:`~vivlmoe.data.image_store.ImageLoadError` (never a
placeholder image)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch.utils.data import Dataset

from vivlmoe.data.image_store import ImageLoadError, ImageStore
from vivlmoe.data.records import (
    ImageRef,
    ImageTextRecord,
    MultiImageClsRecord,
    ParallelTextRecord,
    RecordSet,
    VQARecord,
)

Transform = Callable[[Any], torch.Tensor]


def load_image_tensor(
    store: ImageStore, ref: ImageRef, transform: Transform, draft_size: int | None, uid: int
) -> torch.Tensor:
    try:
        img = store.open(ref, draft_size)
    except ImageLoadError as exc:
        raise ImageLoadError(f"uid={uid}: {exc}") from exc
    return transform(img)


def _refs_by_image(records: RecordSet) -> list[tuple[int, list[str]]]:
    """``(first record index, references)`` per unique image in first-occurrence order.
    References are the record's ``refs`` when present, else all texts of the image group."""
    out = []
    for g in records.groups():
        first = records[g[0]]
        refs = list(first.refs) if first.refs else [records[i].text for i in g]
        out.append((g[0], refs))
    return out


class ImageTextDataset(Dataset[dict[str, Any]]):
    """Contrastive training: one image + one caption per item."""

    def __init__(
        self,
        records: RecordSet,
        store: ImageStore,
        transform: Transform,
        draft_size: int | None = None,
    ) -> None:
        if records.record_type is not ImageTextRecord:
            raise TypeError(f"ImageTextDataset needs ImageTextRecords, got {records.record_type}")
        self.records, self.store, self.transform, self.draft_size = (
            records,
            store,
            transform,
            draft_size,
        )

    def __len__(self) -> int:
        return len(self.records)

    def groups(self) -> list[list[int]]:
        return self.records.groups()

    def __getitem__(self, i: int) -> dict[str, Any]:
        r: ImageTextRecord = self.records[i]
        pv = load_image_tensor(self.store, r.image, self.transform, self.draft_size, r.uid)
        return {"uid": r.uid, "image_id": r.image_id, "pixel_values": pv, "text": r.text}


class RetrievalEvalDataset:
    """Image side + text side of a retrieval benchmark with ``img2caps`` / ``cap2img``.

    ``images[k]`` yields ``uid == img_idx == k``; ``texts[c]`` yields ``uid == cap_idx == c``.
    """

    def __init__(
        self,
        records: RecordSet,
        store: ImageStore,
        transform: Transform,
        draft_size: int | None = None,
    ) -> None:
        self.store, self.transform, self.draft_size = store, transform, draft_size
        self.image_ids: list[str] = []
        self.image_refs: list[ImageRef] = []
        self.captions: list[str] = []
        self.cap2img: list[int] = []
        self.img2caps: list[list[int]] = []
        for first, refs in _refs_by_image(records):
            r = records[first]
            k = len(self.image_ids)
            self.image_ids.append(r.image_id)
            self.image_refs.append(r.image)
            caps = []
            for text in refs:
                caps.append(len(self.captions))
                self.captions.append(text)
                self.cap2img.append(k)
            self.img2caps.append(caps)
        if not self.captions:
            raise ValueError("retrieval dataset has no captions")
        self.images = _RetrievalImageSide(self)
        self.texts = _RetrievalTextSide(self)


class _RetrievalImageSide(Dataset[dict[str, Any]]):
    def __init__(self, parent: RetrievalEvalDataset) -> None:
        self.p = parent

    def __len__(self) -> int:
        return len(self.p.image_ids)

    def __getitem__(self, i: int) -> dict[str, Any]:
        pv = load_image_tensor(
            self.p.store, self.p.image_refs[i], self.p.transform, self.p.draft_size, i
        )
        return {"uid": i, "img_idx": i, "image_id": self.p.image_ids[i], "pixel_values": pv}


class _RetrievalTextSide(Dataset[dict[str, Any]]):
    def __init__(self, parent: RetrievalEvalDataset) -> None:
        self.p = parent

    def __len__(self) -> int:
        return len(self.p.captions)

    def __getitem__(self, i: int) -> dict[str, Any]:
        k = self.p.cap2img[i]
        return {
            "uid": i,
            "cap_idx": i,
            "img_idx": k,
            "image_id": self.p.image_ids[k],
            "text": self.p.captions[i],
        }


class CaptionEvalDataset(Dataset[dict[str, Any]]):
    """One item per unique image with all references (multi-reference BLEU / ROUGE / CIDEr)."""

    def __init__(
        self,
        records: RecordSet,
        store: ImageStore,
        transform: Transform,
        draft_size: int | None = None,
    ) -> None:
        self.store, self.transform, self.draft_size = store, transform, draft_size
        self.items: list[tuple[str, ImageRef, list[str]]] = []
        for first, refs in _refs_by_image(records):
            r = records[first]
            self.items.append((r.image_id, r.image, refs))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, Any]:
        image_id, ref, refs = self.items[i]
        pv = load_image_tensor(self.store, ref, self.transform, self.draft_size, i)
        return {"uid": i, "image_id": image_id, "pixel_values": pv, "refs": list(refs)}


class VQADataset(Dataset[dict[str, Any]]):
    """Image + question -> answer. ``expand_answers`` makes one item per gold answer (train)."""

    def __init__(
        self,
        records: RecordSet,
        store: ImageStore,
        transform: Transform,
        draft_size: int | None = None,
        expand_answers: bool = False,
    ) -> None:
        if records.record_type is not VQARecord:
            raise TypeError(f"VQADataset needs VQARecords, got {records.record_type}")
        self.records, self.store, self.transform, self.draft_size = (
            records,
            store,
            transform,
            draft_size,
        )
        self.index: list[tuple[int, int]] = []  # (record index, answer index)
        for i, r in enumerate(records):
            if expand_answers:
                self.index.extend((i, a) for a in range(len(r.answers)))
            else:
                self.index.append((i, 0))

    def __len__(self) -> int:
        return len(self.index)

    def groups(self) -> list[list[int]]:
        order: dict[str, list[int]] = {}
        for k, (i, _) in enumerate(self.index):
            order.setdefault(self.records[i].image_id, []).append(k)
        return list(order.values())

    def __getitem__(self, k: int) -> dict[str, Any]:
        i, a = self.index[k]
        r: VQARecord = self.records[i]
        pv = load_image_tensor(self.store, r.image, self.transform, self.draft_size, r.uid)
        return {
            "uid": k,
            "record_uid": r.uid,
            "image_id": r.image_id,
            "pixel_values": pv,
            "question": r.question,
            "answer": r.answers[a],
            "refs": list(r.answers),
        }


class MultiImageClsDataset(Dataset[dict[str, Any]]):
    """ViMACSA: up to ``max_images`` zero-padded images ``[M,3,H,W]`` + ``image_mask[M]``."""

    def __init__(
        self,
        records: RecordSet,
        store: ImageStore,
        transform: Transform,
        image_hw: tuple[int, int],
        max_images: int = 6,
        draft_size: int | None = None,
    ) -> None:
        if records.record_type is not MultiImageClsRecord:
            raise TypeError(
                f"MultiImageClsDataset needs MultiImageClsRecords, got {records.record_type}"
            )
        self.records, self.store, self.transform, self.draft_size = (
            records,
            store,
            transform,
            draft_size,
        )
        self.image_hw = image_hw
        self.max_images = max_images

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict[str, Any]:
        r: MultiImageClsRecord = self.records[i]
        h, w = self.image_hw
        pv = torch.zeros(self.max_images, 3, h, w)
        mask = torch.zeros(self.max_images, dtype=torch.bool)
        for m, ref in enumerate(r.images[: self.max_images]):
            t = load_image_tensor(self.store, ref, self.transform, self.draft_size, r.uid)
            if tuple(t.shape[-2:]) != (h, w):
                raise ValueError(f"transform produced {tuple(t.shape)}, expected [3,{h},{w}]")
            pv[m] = t
            mask[m] = True
        return {
            "uid": r.uid,
            "sample_id": r.sample_id,
            "pixel_values": pv,
            "image_mask": mask,
            "text": r.text,
            "image_tags": list(r.image_tags),
            "labels": torch.tensor(r.labels, dtype=torch.long),
        }


class ParallelTextDataset(Dataset[dict[str, Any]]):
    """Stage 1: English source (teacher) / Vietnamese target (student)."""

    def __init__(self, records: RecordSet) -> None:
        if records.record_type is not ParallelTextRecord:
            raise TypeError(
                f"ParallelTextDataset needs ParallelTextRecords, got {records.record_type}"
            )
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict[str, Any]:
        r: ParallelTextRecord = self.records[i]
        return {"uid": r.uid, "src": r.src, "tgt": r.tgt}


def singleton_groups(n: int) -> list[list[int]]:
    return [[i] for i in range(n)]


__all__ = [
    "CaptionEvalDataset",
    "ImageTextDataset",
    "MultiImageClsDataset",
    "ParallelTextDataset",
    "RetrievalEvalDataset",
    "VQADataset",
    "load_image_tensor",
    "singleton_groups",
]
_ = Sequence  # re-exported typing helper for adapters
