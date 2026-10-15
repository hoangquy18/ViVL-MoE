"""ViMACSA multimodal aspect-category sentiment (Sec. 6, legacy ``train_macsa.py`` protocol).

Per sample: a review, up to ``max_images`` images (``image/``), and ``["Aspect#Polarity", ...]``.
The record carries one label per aspect (0 None, 1 Negative, 2 Neutral, 3 Positive; aspects
without a label get ``None``) and the sorted union of ResNet-152 image aspect tags
(``resnet152_image_label.json``: ``{"<image file>": ["tag", ...]}``; images without an entry add
nothing, as in the legacy code). The pandas reader uses *positional* columns of
``pd.read_json`` (0 comment, 1 image list, 3 labels) exactly like the legacy script; the json
reader uses explicit keys.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from vivlmoe.config.schema import DEFAULT_ASPECTS, SplitSpec
from vivlmoe.data.adapters.base import BaseAdapter
from vivlmoe.data.records import ImageRef, ListRecordSet, MultiImageClsRecord, RecordSet
from vivlmoe.data.registry import register_dataset

log = logging.getLogger(__name__)

ASPECTS: tuple[str, ...] = tuple(DEFAULT_ASPECTS)
LABELS: tuple[str, ...] = ("None", "Negative", "Neutral", "Positive")
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}


def parse_labels(
    pairs: list[str], aspects: tuple[str, ...] | list[str], where: str
) -> tuple[int, ...]:
    """``["Room#Positive", "Public_area#Negative"]`` -> ``(0, 0, 3, 0, 0, 1)`` for the default aspects."""
    out = [LABEL_TO_ID["None"]] * len(aspects)
    for item in pairs:
        if not isinstance(item, str) or "#" not in item:
            raise ValueError(f"{where}: label must look like 'Aspect#Polarity', got {item!r}")
        asp, pol = item.split("#", 1)
        if asp not in aspects:
            raise ValueError(f"{where}: unknown aspect {asp!r}; expected one of {list(aspects)}")
        if pol not in LABEL_TO_ID:
            raise ValueError(f"{where}: unknown polarity {pol!r}; expected one of {list(LABELS)}")
        out[list(aspects).index(asp)] = LABEL_TO_ID[pol]
    return tuple(out)


@register_dataset("vimacsa")
class ViMACSAAdapter(BaseAdapter):
    @dataclass
    class Options:
        reader: Literal["pandas", "json"] = "pandas"
        image_aspect_json: str | None = (
            "resnet152_image_label.json"  # relative to root; None = no tags
        )
        max_images: int = 6
        aspects: list[str] = field(default_factory=lambda: list(DEFAULT_ASPECTS))
        text_col: int = 0  # pandas positional columns (legacy)
        images_col: int = 1
        labels_col: int = 3
        text_key: str = "comment"  # json reader keys
        images_key: str = "images"
        labels_key: str = "labels"

    default_splits: ClassVar[dict[str, SplitSpec]] = {
        "train": SplitSpec(file="train.json", image_root="image"),
        "dev": SplitSpec(file="dev.json", image_root="image"),
        "test": SplitSpec(file="test.json", image_root="image"),
    }

    def _rows(self, split: str) -> list[tuple[str, list[str], list[str]]]:
        path = self.split_file(split)
        o = self.options
        rows: list[tuple[str, list[str], list[str]]] = []
        if o.reader == "pandas":
            import pandas as pd

            df = pd.read_json(path)
            need = max(o.text_col, o.images_col, o.labels_col) + 1
            if df.shape[1] < need:
                raise ValueError(
                    f"{path}: expected >= {need} columns, got {df.shape[1]} ({list(df.columns)})"
                )
            for i in range(df.shape[0]):
                vals = df.iloc[i, :].values
                rows.append((vals[o.text_col], vals[o.images_col], vals[o.labels_col]))
        else:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                data = list(data.values())
            for entry in data:
                rows.append((entry[o.text_key], entry[o.images_key], entry[o.labels_key]))
        return rows

    def _tags(self) -> dict[str, list[str]]:
        rel = self.options.image_aspect_json
        if not rel:
            return {}
        path = self.root / rel
        if not path.is_file():
            raise FileNotFoundError(
                f"vimacsa: image_aspect_json {path} not found (set it to null to disable tags)"
            )
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return {str(k): [str(t) for t in v] for k, v in data.items()}

    def _raw_records(self, split: str) -> RecordSet:
        o = self.options
        aspects = tuple(o.aspects)
        tags = self._tags()
        records: list[MultiImageClsRecord] = []
        truncated = 0
        for i, (text, images, labels) in enumerate(self._rows(split)):
            where = f"vimacsa/{split}[{i}]"
            if not isinstance(text, str):
                raise ValueError(f"{where}: comment must be a string, got {type(text).__name__}")
            images = [
                str(x) for x in (images if isinstance(images, (list, tuple)) else list(images))
            ]
            if len(images) > o.max_images:
                truncated += 1
                images = images[: o.max_images]
            union: set[str] = set()
            for name in images:
                union.update(tags.get(name, []))
            records.append(
                MultiImageClsRecord(
                    uid=i,
                    sample_id=f"{split}:{i}",
                    text=text,
                    images=tuple(ImageRef(0, name) for name in images),
                    image_tags=tuple(sorted(union)),
                    labels=parse_labels(list(labels), aspects, where),
                )
            )
        if truncated:
            log.info(
                "vimacsa/%s: %d samples truncated to %d images", split, truncated, o.max_images
            )
        return ListRecordSet(records)


__all__ = ["ASPECTS", "LABELS", "LABEL_TO_ID", "ViMACSAAdapter", "parse_labels"]
