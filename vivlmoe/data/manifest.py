"""Index-time image validation with a gzip-JSON cache.

Missing or corrupt images are *dropped* (``policy='skip'``, crawled corpora) or fatal
(``policy='error'``, benchmark datasets). The cache stores the set of bad image keys for a given
(store, level) so re-indexing a 1.1M-image corpus does not re-stat every file. Callers (rank 0)
persist the returned ``dropped`` list next to the run for auditing.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from vivlmoe.data.image_store import ImageLoadError, ImageStore
from vivlmoe.data.records import (
    ImageRef,
    ImageTextRecord,
    ListRecordSet,
    MultiImageClsRecord,
    RecordSet,
    VQARecord,
    with_uid,
)

log = logging.getLogger(__name__)

CheckLevel = Literal["none", "exists", "decode"]
MissingPolicy = Literal["skip", "error"]
_CACHE_VERSION = 1


@dataclass(frozen=True)
class DroppedEntry:
    uid: int
    image_key: str
    reason: str


def _ref_key(ref: ImageRef) -> str:
    return f"{ref.source}:{ref.key}"


def _check_ref(store: ImageStore, ref: ImageRef, level: CheckLevel) -> str | None:
    """Return a failure reason or ``None``."""
    if not store.exists(ref):
        return "missing"
    if level == "decode":
        try:
            store.open(ref, draft_size=64)
        except ImageLoadError as exc:
            return f"undecodable: {exc}"
    return None


def _cache_key(store: ImageStore, level: CheckLevel, num_keys: int) -> dict[str, object]:
    return {"version": _CACHE_VERSION, "store": store.describe(), "level": level, "keys": num_keys}


def _read_cache(path: Path, key: dict[str, object]) -> dict[str, str] | None:
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        log.warning("manifest cache %s unreadable (%s); rebuilding", path, exc)
        return None
    if data.get("key") != key:
        log.info("manifest cache %s is stale (key mismatch); rebuilding", path)
        return None
    return dict(data["bad"])


def _write_cache(path: Path, key: dict[str, object], bad: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump({"key": key, "bad": bad}, fh, ensure_ascii=False)
    os.replace(tmp, path)


def _record_refs(record: object) -> tuple[ImageRef, ...]:
    if isinstance(record, (ImageTextRecord, VQARecord)):
        return (record.image,)
    if isinstance(record, MultiImageClsRecord):
        return record.images
    return ()


def check_image_keys(
    store: ImageStore, refs: Iterable[ImageRef], level: CheckLevel
) -> dict[str, str]:
    """Return ``{ref_key: reason}`` for every bad reference (deduplicated)."""
    bad: dict[str, str] = {}
    seen: set[str] = set()
    for ref in refs:
        k = _ref_key(ref)
        if k in seen:
            continue
        seen.add(k)
        reason = _check_ref(store, ref, level)
        if reason is not None:
            bad[k] = reason
    return bad


def validate_records(
    records: RecordSet,
    store: ImageStore,
    level: CheckLevel = "exists",
    policy: MissingPolicy = "error",
    cache_path: str | Path | None = None,
) -> tuple[RecordSet, list[DroppedEntry]]:
    """Drop (or refuse) records whose images fail the check; ``uid`` is renumbered densely.

    Multi-image records lose only their bad images (a record with zero remaining images is kept:
    the classifier handles empty image sets). Returns ``(records, dropped)``.
    """
    if level == "none":
        return records, []
    all_refs = [ref for r in records for ref in _record_refs(r)]
    unique = {_ref_key(ref): ref for ref in all_refs}
    key = _cache_key(store, level, len(unique))
    cache = Path(cache_path) if cache_path is not None else None
    bad = _read_cache(cache, key) if cache is not None else None
    if bad is None:
        bad = check_image_keys(store, unique.values(), level)
        if cache is not None:
            _write_cache(cache, key, bad)
            log.info("manifest cache written to %s (%d bad of %d)", cache, len(bad), len(unique))
    else:
        log.info("manifest cache hit %s (%d bad of %d)", cache, len(bad), len(unique))
    if not bad:
        return records, []
    dropped: list[DroppedEntry] = []
    keep: list[int] = []
    rebuilt: list[object] = []
    for i, r in enumerate(records):
        refs = _record_refs(r)
        bad_here = [(ref, bad[_ref_key(ref)]) for ref in refs if _ref_key(ref) in bad]
        if not bad_here:
            keep.append(i)
            rebuilt.append(r)
            continue
        if policy == "error":
            ref, reason = bad_here[0]
            raise ImageLoadError(
                f"record uid={r.uid}: image {ref.key!r} {reason} in {store.describe()} "
                "(missing_image_policy=error)"
            )
        if isinstance(r, MultiImageClsRecord):
            good = tuple(ref for ref in refs if _ref_key(ref) not in bad)
            for ref, reason in bad_here:
                dropped.append(DroppedEntry(r.uid, str(ref.key), f"image dropped ({reason})"))
            keep.append(i)
            rebuilt.append(replace(r, images=good))
        else:
            ref, reason = bad_here[0]
            dropped.append(DroppedEntry(r.uid, str(ref.key), reason))
    if any(isinstance(r, MultiImageClsRecord) for r in rebuilt):
        out: RecordSet = ListRecordSet([with_uid(r, k) for k, r in enumerate(rebuilt)])  # type: ignore[arg-type]
    else:
        out = records.select(keep)
    log.warning(
        "image validation (%s): kept %d / %d records, %d dropped entries (policy=%s)",
        level,
        len(out),
        len(records),
        len(dropped),
        policy,
    )
    return out, dropped


def write_dropped(path: str | Path, dropped: list[DroppedEntry]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump([d.__dict__ for d in dropped], fh, ensure_ascii=False, indent=1)
    return path


__all__ = ["DroppedEntry", "check_image_keys", "validate_records", "write_dropped"]
