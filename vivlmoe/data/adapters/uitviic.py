"""UIT-ViIC (COCO-style Vietnamese captions, 1-10 per image; no dev -> 20 % holdout by image id)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import ClassVar

from vivlmoe.config.schema import SplitSpec
from vivlmoe.data.adapters.base import BaseAdapter
from vivlmoe.data.adapters.ktvic import build_caption_records
from vivlmoe.data.records import RecordSet
from vivlmoe.data.registry import register_dataset


@register_dataset("uitviic")
class UITViICAdapter(BaseAdapter):
    @dataclass
    class Options:
        caption_key: str = "caption"

    default_splits: ClassVar[dict[str, SplitSpec]] = {
        "train": SplitSpec(
            file="uitvic_captions_train2017.json", image_root="coco_uitvic_train/coco_uitvic_train"
        ),
        "test": SplitSpec(
            file="uitvic_captions_test2017.json", image_root="coco_uitvic_test/coco_uitvic_test"
        ),
    }

    def _raw_records(self, split: str) -> RecordSet:
        path = self.split_file(split)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return build_caption_records(
            data, "file_name", self.options.caption_key, f"uitviic/{split}"
        )


__all__ = ["UITViICAdapter"]
