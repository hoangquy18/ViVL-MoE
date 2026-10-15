"""Deterministic dev holdouts by image id (Appendix A: 20 % of KTVIC / UIT-ViIC train).

The id list is committed to ``configs/data/splits/*.json`` and its sha1 is stored in every
checkpoint so a model is never selected on images it trained on. Holdouts computed on the fly (no
``ids_file``) are logged loudly.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vivlmoe.config.schema import HoldoutSpec
from vivlmoe.data.records import RecordSet
from vivlmoe.utils.hf import repo_root

log = logging.getLogger(__name__)


def holdout_image_ids(
    ids: Sequence[str],
    fraction: float | None = None,
    num_images: int | None = None,
    seed: int = 0,
) -> list[str]:
    """Pick a sorted, seeded subset of the *unique, sorted* ids (independent of input order)."""
    unique = sorted(set(str(i) for i in ids))
    if (fraction is None) == (num_images is None):
        raise ValueError("holdout: set exactly one of fraction / num_images")
    if fraction is not None:
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"holdout.fraction must be in (0, 1), got {fraction}")
        n = int(round(fraction * len(unique)))
    else:
        n = int(num_images)  # type: ignore[arg-type]
    if not 0 < n < len(unique) or len(unique) < 2:
        raise ValueError(f"holdout of {n} images out of {len(unique)} unique ids is impossible")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique))[:n]
    return sorted(unique[i] for i in perm)


def ids_sha1(ids: Sequence[str]) -> str:
    return hashlib.sha1("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def save_ids(path: str | Path, ids: Sequence[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(sorted(set(ids)), fh, indent=0, ensure_ascii=False)
        fh.write("\n")
    return path


def resolve_split_path(path: str | Path) -> Path:
    """Absolute path; relative paths are tried against cwd, then the repository root."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    alt = repo_root() / p
    return alt if alt.exists() else p


def load_ids(path: str | Path) -> list[str]:
    p = resolve_split_path(path)
    if not p.exists():
        raise FileNotFoundError(f"split id file not found: {path} (resolved {p})")
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list) or not all(isinstance(x, (str, int)) for x in data):
        raise ValueError(f"{p}: expected a JSON list of image ids")
    return sorted(str(x) for x in data)


@dataclass(frozen=True)
class HoldoutInfo:
    ids: tuple[str, ...]
    sha1: str
    source: str  # "file:<path>" or "computed(seed=..)"


def resolve_holdout_ids(spec: HoldoutSpec, records: RecordSet) -> HoldoutInfo:
    if spec.ids_file:
        ids = load_ids(spec.ids_file)
        return HoldoutInfo(tuple(ids), ids_sha1(ids), f"file:{spec.ids_file}")
    ids = holdout_image_ids(records.image_ids(), spec.fraction, spec.num_images, spec.seed)
    log.warning(
        "holdout ids computed on the fly (seed=%d, %d images); commit them with "
        "scripts/make_dev_split.py and set holdout.ids_file for reproducibility",
        spec.seed,
        len(ids),
    )
    return HoldoutInfo(tuple(ids), ids_sha1(ids), f"computed(seed={spec.seed})")


def apply_holdout(
    spec: HoldoutSpec | None, split: str, records: RecordSet, run_dir: str | Path | None = None
) -> tuple[RecordSet, HoldoutInfo | None]:
    """Filter ``records`` (loaded from ``spec.from_split``) for ``split``.

    ``split == to_split`` keeps only holdout images, ``split == from_split`` removes them, other
    splits are untouched. ``exclude_ids_file`` is applied to every split. When ``run_dir`` is
    given the effective id list is saved there for auditing.
    """
    info: HoldoutInfo | None = None
    if spec is not None and split in (spec.from_split, spec.to_split):
        info = resolve_holdout_ids(spec, records)
        held = set(info.ids)
        image_ids = records.image_ids()
        if split == spec.to_split:
            keep = [i for i, iid in enumerate(image_ids) if iid in held]
            present = {iid for iid in image_ids if iid in held}
            missing = held - present
            if missing:
                raise ValueError(
                    f"holdout for split {split!r}: {len(missing)} ids from {info.source} are not in "
                    f"the {spec.from_split!r} records (e.g. {sorted(missing)[:3]})"
                )
        else:
            keep = [i for i, iid in enumerate(image_ids) if iid not in held]
        log.info(
            "holdout %s -> %s: split %r keeps %d / %d records (%d ids, sha1 %s, %s)",
            spec.from_split,
            spec.to_split,
            split,
            len(keep),
            len(records),
            len(held),
            info.sha1[:10],
            info.source,
        )
        records = records.select(keep)
        if run_dir is not None:
            save_ids(Path(run_dir) / "splits" / f"{spec.to_split}_image_ids.json", info.ids)
    if spec is not None and spec.exclude_ids_file:
        excluded = set(load_ids(spec.exclude_ids_file))
        image_ids = records.image_ids()
        keep = [i for i, iid in enumerate(image_ids) if iid not in excluded]
        log.info(
            "exclusion list %s: split %r keeps %d / %d records",
            spec.exclude_ids_file,
            split,
            len(keep),
            len(records),
        )
        records = records.select(keep)
    return records, info


__all__ = [
    "HoldoutInfo",
    "apply_holdout",
    "holdout_image_ids",
    "ids_sha1",
    "load_ids",
    "resolve_holdout_ids",
    "resolve_split_path",
    "save_ids",
]
