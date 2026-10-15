"""``BaseAdapter``: the shared ``load_split`` pipeline every dataset adapter inherits.

raw records -> holdout / exclusion -> image validation (manifest) -> text normalisation -> limit ->
dense ``uid`` renumbering, with a logged kept/dropped summary. Subclasses implement
``_raw_records(split)`` (and optionally ``image_store``) and declare an ``Options`` dataclass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from vivlmoe.config.schema import DataSourceCfg, SplitSpec
from vivlmoe.data.image_store import ImageStore, LocalImageStore
from vivlmoe.data.manifest import CheckLevel, DroppedEntry, validate_records, write_dropped
from vivlmoe.data.records import ListRecordSet, RecordSet
from vivlmoe.data.splits import HoldoutInfo, apply_holdout
from vivlmoe.data.text_normalize import build_normalizer

log = logging.getLogger(__name__)


class BaseAdapter:
    kind: ClassVar[str] = ""
    default_splits: ClassVar[dict[str, SplitSpec]] = {}

    @dataclass
    class Options:
        """No options by default; subclasses override with their own dataclass."""

    def __init__(self, cfg: DataSourceCfg, options: Any, run_dir: str | Path | None = None) -> None:
        self.cfg = cfg
        self.options = options
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.dropped: dict[str, list[DroppedEntry]] = {}
        self.holdout_info: dict[str, HoldoutInfo | None] = {}

    # ----------------------------------------------------------------------------- paths
    @property
    def root(self) -> Path:
        return Path(self.cfg.root)

    def split_spec(self, split: str) -> SplitSpec:
        spec = self.cfg.splits.get(split) or self.default_splits.get(split)
        if spec is None:
            known = sorted(set(self.cfg.splits) | set(self.default_splits))
            raise KeyError(f"{self.kind}: no file configured for split {split!r}; known: {known}")
        return spec

    def source_split(self, split: str) -> str:
        """The split whose files back ``split`` (a holdout dev split is carved out of train)."""
        h = self.cfg.holdout
        if h is not None and split == h.to_split:
            return h.from_split
        return split

    def split_file(self, split: str) -> Path:
        return self.root / self.split_spec(split).file

    def image_root(self, split: str) -> Path:
        rel = self.split_spec(split).image_root
        return self.root / rel if rel else self.root

    def image_store(self, split: str) -> ImageStore:
        return LocalImageStore(self.image_root(self.source_split(split)))

    # ------------------------------------------------------------------------ pipeline hooks
    def _raw_records(self, split: str) -> RecordSet:  # pragma: no cover - abstract
        raise NotImplementedError

    def _image_check(self) -> CheckLevel:
        return self.cfg.image_check

    def manifest_cache_path(self, source_split: str) -> Path | None:
        mc = self.cfg.manifest_cache
        if not mc:
            return None
        p = Path(mc)
        if p.name.endswith(".json.gz"):
            return p
        return p / f"{self.kind}_{source_split}_{self._image_check()}.json.gz"

    # -------------------------------------------------------------------------------- main
    def load_split(self, split: str) -> RecordSet:
        src = self.source_split(split)
        records = self._raw_records(src)
        n_raw = len(records)
        records, info = apply_holdout(self.cfg.holdout, split, records, self.run_dir)
        self.holdout_info[split] = info
        n_holdout = len(records)
        level = self._image_check()
        dropped: list[DroppedEntry] = []
        if level != "none":  # text-only adapters have no image store
            records, dropped = validate_records(
                records,
                self.image_store(split),
                level=level,
                policy=self.cfg.missing_image_policy,
                cache_path=self.manifest_cache_path(src),
            )
        self.dropped[split] = dropped
        normalizer = build_normalizer(self.cfg.text_normalize)
        if normalizer is not None:
            records = records.map_text(normalizer)
        if self.cfg.limit is not None:
            records = records.select(range(min(self.cfg.limit, len(records))))
        records = _ensure_dense_uids(records)
        log.info(
            "%s/%s: %d raw -> %d after holdout -> %d after image check (%d dropped) -> %d final "
            "(normalize=%s, limit=%s)",
            self.kind,
            split,
            n_raw,
            n_holdout,
            len(records) if self.cfg.limit is None else n_holdout - len(dropped),
            len(dropped),
            len(records),
            self.cfg.text_normalize,
            self.cfg.limit,
        )
        if len(records) == 0:
            raise ValueError(f"{self.kind}/{split}: no records left after the pipeline")
        return records

    def write_dropped(self, split: str, run_dir: str | Path | None = None) -> Path | None:
        """Persist the dropped-image list (call on rank 0 only)."""
        rd = Path(run_dir) if run_dir is not None else self.run_dir
        dropped = self.dropped.get(split, [])
        if rd is None or not dropped:
            return None
        return write_dropped(rd / "manifests" / f"{self.kind}_{split}_dropped.json", dropped)


def _ensure_dense_uids(records: RecordSet) -> RecordSet:
    if isinstance(records, ListRecordSet) and any(r.uid != i for i, r in enumerate(records)):
        return records.select(range(len(records)))
    return records


__all__ = ["BaseAdapter"]
