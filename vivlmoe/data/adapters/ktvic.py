"""KTVIC (Vietnamese image captioning, Sec. 6 / Appendix A).

``{"images": [{"id", "filename"}], "annotations": [{"id", "image_id", "caption", "segment_caption"}]}``
with exactly 5 captions per image and no dev split: the 20 % dev holdout is carved out of train by
image id (``holdout.ids_file: configs/data/splits/ktvic_dev_image_ids.json``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from vivlmoe.config.schema import SplitSpec
from vivlmoe.data.adapters.base import BaseAdapter
from vivlmoe.data.records import ImageRef, ImageTextRecord, ListRecordSet, RecordSet
from vivlmoe.data.registry import register_dataset

log = logging.getLogger(__name__)


def build_caption_records(
    data: dict[str, Any], filename_key: str, caption_key: str, where: str
) -> RecordSet:
    """COCO-style ``images``/``annotations`` -> one ImageTextRecord per caption with all refs."""
    for k in ("images", "annotations"):
        if k not in data or not isinstance(data[k], list):
            raise ValueError(f"{where}: missing list {k!r}")
    filenames: dict[str, str] = {}
    for img in data["images"]:
        if "id" not in img or filename_key not in img:
            raise ValueError(f"{where}: image entry needs 'id' and {filename_key!r}: {img!r}")
        filenames[str(img["id"])] = str(img[filename_key])
    captions: dict[str, list[str]] = {iid: [] for iid in filenames}
    for ann in data["annotations"]:
        iid = str(ann.get("image_id"))
        if iid not in filenames:
            raise ValueError(f"{where}: annotation refers to unknown image_id {iid!r}")
        cap = ann.get(caption_key)
        if not isinstance(cap, str) or not cap.strip():
            raise ValueError(f"{where}: empty or missing {caption_key!r} for image {iid}")
        captions[iid].append(cap.strip())
    records: list[ImageTextRecord] = []
    empty = 0
    for iid, filename in filenames.items():  # file order (deterministic)
        caps = captions[iid]
        if not caps:
            empty += 1
            continue
        refs = tuple(caps)
        for cap in caps:
            records.append(
                ImageTextRecord(
                    uid=len(records), image_id=iid, image=ImageRef(0, filename), text=cap, refs=refs
                )
            )
    if empty:
        log.warning("%s: dropped %d images without captions", where, empty)
    log.info("%s: %d captions over %d images", where, len(records), len(filenames) - empty)
    return ListRecordSet(records)


@register_dataset("ktvic")
class KTVICAdapter(BaseAdapter):
    @dataclass
    class Options:
        caption_key: Literal["caption", "segment_caption"] = "caption"

    default_splits: ClassVar[dict[str, SplitSpec]] = {
        "train": SplitSpec(file="train_data.json", image_root="train-images"),
        "test": SplitSpec(file="test_data.json", image_root="public-test-images"),
    }

    def _raw_records(self, split: str) -> RecordSet:
        path: Path = self.split_file(split)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return build_caption_records(data, "filename", self.options.caption_key, f"ktvic/{split}")


__all__ = ["KTVICAdapter", "build_caption_records"]
