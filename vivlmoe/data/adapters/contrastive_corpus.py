"""Image-caption corpora for Stages 2/3 and sparse training (Sec. 4.2-4.3):

* ``json_dict``  ``{"<idx>": {"image_id": "images/x.jpg", "caption": "..."}}`` (``filtered_data.json``)
* ``jsonl``      one ``{"image": "...", "caption": "..."}`` per line (``english.jsonl``, and the
  decontaminated corpus written by :mod:`vivlmoe.data.corpus_prep`)
* ``hf_image_caption``  a Hugging Face dataset with image + caption columns
* ``mixture``    concatenation of the above with a :class:`CompositeImageStore`

All produce :class:`~vivlmoe.data.records.ColumnarImageText` (byte buffers, no Python string lists).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vivlmoe.config.loader import from_dict
from vivlmoe.config.schema import DataSourceCfg
from vivlmoe.data.adapters.base import BaseAdapter
from vivlmoe.data.image_store import CompositeImageStore, HFImageStore, ImageStore
from vivlmoe.data.manifest import CheckLevel
from vivlmoe.data.records import ColumnarImageText, ImageTextRecord, RecordSet
from vivlmoe.data.registry import build_adapter, register_dataset

log = logging.getLogger(__name__)


@dataclass
class _KeyedOptions:
    image_key: str = "image_id"
    caption_key: str = "caption"
    skip_invalid: bool = False  # drop entries missing a key / empty caption instead of raising


def _entry(
    entry: Any, image_key: str, caption_key: str, where: str, skip_invalid: bool
) -> tuple[str, str] | None:
    if not isinstance(entry, dict) or image_key not in entry or caption_key not in entry:
        if skip_invalid:
            return None
        raise ValueError(f"{where}: expected keys {image_key!r} and {caption_key!r}, got {entry!r}")
    image, caption = entry[image_key], entry[caption_key]
    if not isinstance(image, str) or not isinstance(caption, str) or not caption.strip():
        if skip_invalid:
            return None
        raise ValueError(f"{where}: image must be a path and caption a non-empty string: {entry!r}")
    return image, caption


def _columnar(pairs: list[tuple[str, str]]) -> ColumnarImageText:
    keys = [p[0] for p in pairs]
    return ColumnarImageText(keys, [p[1] for p in pairs], image_ids=keys)


@register_dataset("json_dict")
class JsonDictCorpus(BaseAdapter):
    Options = _KeyedOptions

    def _raw_records(self, split: str) -> RecordSet:
        path = self.split_file(split)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: json_dict corpus must be a mapping idx -> entry")
        keys = list(data)
        if keys and all(k.isdigit() for k in keys):
            keys.sort(key=int)  # index gaps are fine; order by numeric index
        o = self.options
        pairs, skipped = [], 0
        for k in keys:
            e = _entry(data[k], o.image_key, o.caption_key, f"{path}[{k}]", o.skip_invalid)
            if e is None:
                skipped += 1
            else:
                pairs.append(e)
        if skipped:
            log.warning("%s: skipped %d invalid entries", path, skipped)
        return _columnar(pairs)


@register_dataset("jsonl")
class JsonlCorpus(BaseAdapter):
    @dataclass
    class Options(_KeyedOptions):
        image_key: str = "image"

    def _raw_records(self, split: str) -> RecordSet:
        path = self.split_file(split)
        o = self.options
        pairs, skipped = [], 0
        with path.open("r", encoding="utf-8") as fh:
            for ln, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    if o.skip_invalid:
                        skipped += 1
                        continue
                    raise ValueError(f"{path}:{ln + 1}: invalid JSON ({exc})") from None
                e = _entry(entry, o.image_key, o.caption_key, f"{path}:{ln + 1}", o.skip_invalid)
                if e is None:
                    skipped += 1
                else:
                    pairs.append(e)
        if skipped:
            log.warning("%s: skipped %d invalid lines", path, skipped)
        return _columnar(pairs)


@register_dataset("hf_image_caption")
class HFImageCaptionCorpus(BaseAdapter):
    @dataclass
    class Options:
        dataset: str  # HF hub id or a ``save_to_disk`` directory
        hf_split: str = "train"
        image_column: str = "image"
        caption_column: str = "caption"
        load_kwargs: dict[str, Any] = field(default_factory=dict)

    def image_store(self, split: str) -> ImageStore:
        o = self.options
        return HFImageStore(o.dataset, o.hf_split, o.image_column, o.load_kwargs)

    def _raw_records(self, split: str) -> RecordSet:
        o = self.options
        ds = HFImageStore.load_hf_dataset(o.dataset, o.hf_split, **o.load_kwargs)
        for col in (o.image_column, o.caption_column):
            if col not in ds.column_names:
                raise KeyError(f"HF dataset {o.dataset!r} has no column {col!r}: {ds.column_names}")
        captions = ds[o.caption_column]
        n = len(captions)
        keys = [str(i) for i in range(n)]
        ids = [f"hf:{i}" for i in range(n)]
        return ColumnarImageText(keys, [str(c) for c in captions], image_ids=ids)


@register_dataset("mixture")
class MixtureCorpus(BaseAdapter):
    """``options.sources``: list of DataSourceCfg mappings (each validated by its own adapter and
    run through its own pipeline, including image checks)."""

    @dataclass
    class Options:
        sources: list[dict[str, Any]] = field(default_factory=list)

    def __init__(self, cfg: DataSourceCfg, options: Any, run_dir: str | Path | None = None) -> None:
        super().__init__(cfg, options, run_dir)
        if not options.sources:
            raise ValueError("mixture: options.sources must list at least one source")
        self.sub_cfgs = [
            from_dict(DataSourceCfg, src, path=f"data.<mixture>.options.sources[{i}]")
            for i, src in enumerate(options.sources)
        ]
        self.adapters = [build_adapter(c, run_dir) for c in self.sub_cfgs]

    def _image_check(self) -> CheckLevel:
        return "none"  # each source validated its own images

    def _store_list(self, split: str) -> list[ImageStore]:
        stores: list[ImageStore] = []
        for ad in self.adapters:
            st = ad.image_store(split)
            stores.extend(st.stores if isinstance(st, CompositeImageStore) else [st])
        return stores

    def image_store(self, split: str) -> ImageStore:
        return CompositeImageStore(self._store_list(split))

    def _raw_records(self, split: str) -> RecordSet:
        parts: list[ColumnarImageText] = []
        counts: list[int] = []
        for ad in self.adapters:
            recs = ad.load_split(split)
            if not isinstance(recs, ColumnarImageText):
                if recs.record_type is not ImageTextRecord:
                    raise TypeError(f"mixture source {ad.kind} yields {recs.record_type.__name__}")
                recs = ColumnarImageText(
                    [str(r.image.key) for r in recs], [r.text for r in recs], recs.image_ids()
                )
            st = ad.image_store(split)
            counts.append(len(st.stores) if isinstance(st, CompositeImageStore) else 1)
            parts.append(recs)
            log.info("mixture: source %s/%s contributes %d records", ad.kind, split, len(recs))
        return ColumnarImageText.concat(parts, counts)


__all__ = ["HFImageCaptionCorpus", "JsonDictCorpus", "JsonlCorpus", "MixtureCorpus"]
