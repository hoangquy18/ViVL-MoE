"""Synthetic datasets for unit tests and the CPU smoke chain (``configs/_smoke``).

Tiny random RGB PNGs are written once into ``cfg.root`` (or a deterministic temp directory)
and Vietnamese-looking sentences are drawn from the tiny tokenizer vocabulary, all seeded by
``options.seed`` + the split name, so every rank/process builds identical records.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vivlmoe.config.schema import DEFAULT_ASPECTS, DataSourceCfg
from vivlmoe.data.adapters.base import BaseAdapter
from vivlmoe.data.image_store import ImageStore, LocalImageStore
from vivlmoe.data.records import (
    ImageRef,
    ImageTextRecord,
    ListRecordSet,
    MultiImageClsRecord,
    ParallelTextRecord,
    RecordSet,
    VQARecord,
)
from vivlmoe.data.registry import register_dataset
from vivlmoe.data.tokenization import TINY_WORDS

log = logging.getLogger(__name__)
_VI_WORDS = [w for w in TINY_WORDS if not w.isascii()] + ["một", "con", "đang", "trên"]
_EN_WORDS = [w for w in TINY_WORDS if w.isascii() and not w.startswith("w")]


@dataclass
class SyntheticOptions:
    num_samples: int = 16  # records (pairs: images x captions_per_image)
    captions_per_image: int = 2
    image_size: int = 32
    seed: int = 0
    max_images: int = 3  # vimacsa: images per sample in [0, max_images]
    aspects: list[str] = field(default_factory=lambda: list(DEFAULT_ASPECTS))
    num_answers: int = 2  # vqa: gold answers per question


class _SyntheticBase(BaseAdapter):
    Options = SyntheticOptions

    def __init__(self, cfg: DataSourceCfg, options: Any, run_dir: str | Path | None = None) -> None:
        super().__init__(cfg, options, run_dir)
        self._dirs: dict[str, Path] = {}

    def _rng(self, split: str) -> np.random.Generator:
        return np.random.default_rng(self.options.seed * 1009 + zlib.crc32(split.encode()))

    def _digest(self, split: str) -> str:
        o = self.options
        key = f"{self.kind}|{split}|{o.seed}|{o.image_size}|{self._num_images(split)}"
        return hashlib.sha1(key.encode()).hexdigest()[:12]

    def _num_images(self, split: str) -> int:
        return self.options.num_samples

    def image_dir(self, split: str) -> Path:
        if split not in self._dirs:
            base = (
                Path(self.cfg.root)
                if self.cfg.root
                else Path(tempfile.gettempdir()) / "vivlmoe_synthetic"
            )
            d = base / f"{self.kind}_{split}_{self._digest(split)}"
            n = self._num_images(split)
            if not (d / f"img_{n - 1:05d}.png").is_file() if n else not d.is_dir():
                self._write_images(d, n, split)
            self._dirs[split] = d
        return self._dirs[split]

    def _write_images(self, d: Path, n: int, split: str) -> None:
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(self.options.seed * 7 + zlib.crc32(split.encode()) + 1)
        s = self.options.image_size
        for k in range(n):
            arr = rng.integers(0, 256, size=(s, s, 3), dtype=np.uint8)
            Image.fromarray(arr, "RGB").save(d / f"img_{k:05d}.png")
        log.info("synthetic %s/%s: wrote %d images to %s", self.kind, split, n, d)

    def image_store(self, split: str) -> ImageStore:
        return LocalImageStore(self.image_dir(self.source_split(split)))

    @staticmethod
    def _sentence(rng: np.random.Generator, words: list[str], lo: int = 3, hi: int = 8) -> str:
        n = int(rng.integers(lo, hi + 1))
        return " ".join(words[int(i)] for i in rng.integers(0, len(words), size=n))


@register_dataset("synthetic_pairs")
class SyntheticPairs(_SyntheticBase):
    """Image-text pairs; ``captions_per_image`` captions per image (refs filled)."""

    def _num_images(self, split: str) -> int:
        return max(1, self.options.num_samples // self.options.captions_per_image)

    def _raw_records(self, split: str) -> RecordSet:
        rng = self._rng(split)
        self.image_dir(split)
        records: list[ImageTextRecord] = []
        for k in range(self._num_images(split)):
            caps = tuple(
                self._sentence(rng, _VI_WORDS) for _ in range(self.options.captions_per_image)
            )
            for cap in caps:
                records.append(
                    ImageTextRecord(
                        uid=len(records),
                        image_id=f"img{k}",
                        image=ImageRef(0, f"img_{k:05d}.png"),
                        text=cap,
                        refs=caps,
                    )
                )
        return ListRecordSet(records)


@register_dataset("synthetic_caption")
class SyntheticCaption(SyntheticPairs):
    """Alias of ``synthetic_pairs`` for captioning configs (kept separate for readability)."""


@register_dataset("synthetic_parallel")
class SyntheticParallel(_SyntheticBase):
    def image_store(self, split: str) -> ImageStore:
        raise TypeError("synthetic_parallel is text-only")

    def _image_check(self) -> str:  # type: ignore[override]
        return "none"

    def _raw_records(self, split: str) -> RecordSet:
        rng = self._rng(split)
        return ListRecordSet(
            [
                ParallelTextRecord(
                    uid=i, src=self._sentence(rng, _EN_WORDS), tgt=self._sentence(rng, _VI_WORDS)
                )
                for i in range(self.options.num_samples)
            ]
        )


@register_dataset("synthetic_vqa")
class SyntheticVQA(_SyntheticBase):
    def _raw_records(self, split: str) -> RecordSet:
        rng = self._rng(split)
        self.image_dir(split)
        records = []
        for i in range(self.options.num_samples):
            answers = tuple(
                self._sentence(rng, _VI_WORDS, 1, 2) for _ in range(self.options.num_answers)
            )
            records.append(
                VQARecord(
                    uid=i,
                    image_id=f"img{i}",
                    image=ImageRef(0, f"img_{i:05d}.png"),
                    question=self._sentence(rng, _VI_WORDS) + " ?",
                    answers=answers,
                )
            )
        return ListRecordSet(records)


@register_dataset("synthetic_vimacsa")
class SyntheticViMACSA(_SyntheticBase):
    def _raw_records(self, split: str) -> RecordSet:
        o = self.options
        rng = self._rng(split)
        self.image_dir(split)
        tags_pool = [a.lower() for a in o.aspects]
        records = []
        for i in range(o.num_samples):
            n_img = int(rng.integers(0, o.max_images + 1))
            imgs = tuple(
                ImageRef(0, f"img_{int(k):05d}.png")
                for k in rng.integers(0, o.num_samples, size=n_img)
            )
            tags = sorted({tags_pool[int(t)] for t in rng.integers(0, len(tags_pool), size=n_img)})
            records.append(
                MultiImageClsRecord(
                    uid=i,
                    sample_id=f"{split}:{i}",
                    text=self._sentence(rng, _VI_WORDS, 4, 12),
                    images=imgs,
                    image_tags=tuple(tags),
                    labels=tuple(int(x) for x in rng.integers(0, 4, size=len(o.aspects))),
                )
            )
        return ListRecordSet(records)


__all__ = [
    "SyntheticCaption",
    "SyntheticPairs",
    "SyntheticParallel",
    "SyntheticVQA",
    "SyntheticViMACSA",
]
