"""``caption_text``: the captions of an image-caption jsonl corpus as text-only distillation pairs.

v2 distils mmBERT from the multilingual SigLIP 2 text tower on Vietnamese captions, so each caption is
both the teacher input and the student input (:class:`~vivlmoe.data.records.ColumnarCaptionText`,
``src`` is ``tgt``). Images are never opened (no image check, no image store), and by default each
distinct caption is kept once (corpora repeat captions across images). Combine it with PhoMT
(English teacher input) through a ``mixture`` source.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from vivlmoe.data.adapters.base import BaseAdapter
from vivlmoe.data.manifest import CheckLevel
from vivlmoe.data.records import ColumnarCaptionText, RecordSet
from vivlmoe.data.registry import register_dataset

log = logging.getLogger(__name__)


@register_dataset("caption_text")
class CaptionTextCorpus(BaseAdapter):
    @dataclass
    class Options:
        caption_key: str = "caption"
        dedupe: bool = True  # keep the first occurrence of each distinct caption
        skip_invalid: bool = False  # drop bad lines / empty captions instead of raising

    def _image_check(self) -> CheckLevel:
        return "none"

    def image_store(self, split: str) -> Any:
        raise TypeError("caption_text is a text-only dataset and has no image store")

    def _caption(self, line: str, where: str) -> str | None:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            if self.options.skip_invalid:
                return None
            raise ValueError(f"{where}: invalid JSON ({exc})") from None
        key = self.options.caption_key
        caption = entry.get(key) if isinstance(entry, dict) else None
        if not isinstance(caption, str) or not caption.strip():
            if self.options.skip_invalid:
                return None
            raise ValueError(f"{where}: expected a non-empty string under {key!r}, got {entry!r}")
        return caption

    def _raw_records(self, split: str) -> RecordSet:
        path = self.split_file(split)
        captions: list[str] = []
        seen: set[str] = set()
        n_lines = skipped = 0
        with path.open("r", encoding="utf-8") as fh:
            for ln, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                n_lines += 1
                caption = self._caption(line, f"{path}:{ln + 1}")
                if caption is None:
                    skipped += 1
                    continue
                if self.options.dedupe:
                    if caption in seen:
                        continue
                    seen.add(caption)
                captions.append(caption)
        if skipped:
            log.warning("%s: skipped %d invalid lines", path, skipped)
        log.info(
            "caption_text/%s: %d captions from %d lines (dedupe=%s)",
            split,
            len(captions),
            n_lines,
            self.options.dedupe,
        )
        return ColumnarCaptionText(captions)


__all__ = ["CaptionTextCorpus"]
