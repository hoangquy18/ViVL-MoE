"""Offline preparation of the image-caption pretraining corpus (paper Sec. 4.2-4.3).

The raw dump (``dataset/`` in the repository, see ``docs/runbook.md``) keeps the legacy caption files
next to one flat image folder:

* ``filtered_data.json`` - Vietnamese captions ``{"<idx>": {"image_id": "images/<src>_<id>.<ext>",
  "caption": ...}}`` where ``<src>`` is coco2017, localization, viocrvqa, oxford102, vitextvqa,
  openvivqa, vilc (= UIT-ViIC), evjvqa or vivqa;
* ``english.jsonl`` - English captions ``{"image": "images/english_<n>.<ext>", "caption": ...}``;
* ``images/`` - every image, named ``<src>_<id>.<ext>``. Images are looked up by file name in the
  ``image_dirs`` folders, so other layouts (e.g. one folder per source) work too.

:func:`build_corpus` writes one JSONL line per caption, ``{"image": <path relative to the dataset
root>, "caption": ..., "source": <src>}``, which the ``jsonl`` adapter reads directly. On the way:

1. **Decontamination.** The paper removes "images that appear in the evaluation splits of
   ViMMU-Bench" from the corpus. Two rules apply: whole sources whose evaluation splits cannot be
   told apart are dropped (default :data:`DEFAULT_EXCLUDED_SOURCES`), and every image whose COCO id
   is on an exclusion list is dropped from all COCO-backed sources (:data:`COCO_ID_SOURCES`), since
   UIT-ViIC images also appear as ``coco2017_*`` and ``vivqa_*``.
2. **Cleaning.** Whitespace is collapsed, then empty captions and exact duplicate (image, caption)
   pairs are dropped.
3. **Image resolution.** A caption whose image file is in none of the folders is dropped. The
   build refuses to write the corpus when more than ``max_missing`` of the captions lose their
   image (a half-extracted archive must not become a silently smaller corpus).

The report written next to the corpus counts every drop per source and records the sha1 of the
inputs, the exclusion lists and the output.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)

CaptionFormat = Literal["json_dict", "jsonl"]

# File names of these sources embed the COCO image id (``coco2017_000000203564.jpg``).
COCO_ID_SOURCES = frozenset({"coco2017", "vilc", "vivqa"})
# ViMMU-Bench VQA sets whose split membership is unknown for the converted captions (the dump keeps
# no split column, and the file names are re-numbered): dropped whole by default.
DEFAULT_EXCLUDED_SOURCES = ("vitextvqa", "openvivqa", "viocrvqa")
DEFAULT_IMAGE_DIRS = ("images",)
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"})
DROP_REASONS = (
    "invalid_entry",
    "excluded_source",
    "excluded_coco_id",
    "empty_caption",
    "duplicate",
    "missing_image",
)
_NAME_RE = re.compile(r"^([A-Za-z]+[0-9]*?)_([^./]+)\.[A-Za-z0-9]+$")
_MISSING_EXAMPLES = 20


class CorpusBuildError(RuntimeError):
    """The corpus cannot be built as configured (the report is still written)."""


@dataclass(frozen=True)
class CaptionFile:
    """One caption source file and the keys of its image path / caption fields."""

    path: Path
    fmt: CaptionFormat
    image_key: str
    caption_key: str = "caption"


def parse_name(name: str) -> tuple[str, str]:
    """``images/coco2017_000000203564.jpg`` -> ``("coco2017", "000000203564")``."""
    base = name.rsplit("/", 1)[-1]
    m = _NAME_RE.match(base)
    if m is None:
        raise ValueError(f"unrecognised corpus image name {name!r} (expected <source>_<id>.<ext>)")
    return m.group(1), m.group(2)


def coco_id(source: str, stem: str) -> int | None:
    """The COCO image id encoded in a file name, for :data:`COCO_ID_SOURCES` only."""
    return int(stem) if source in COCO_ID_SOURCES and stem.isdigit() else None


def iter_captions(cf: CaptionFile) -> Iterator[tuple[Any, Any]]:
    """Yield ``(image, caption)`` in file order (``json_dict``: by numeric index)."""
    if cf.fmt == "json_dict":
        with cf.path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"{cf.path}: json_dict corpus must be a mapping idx -> entry")
        keys = sorted(data, key=lambda k: (0, int(k), "") if k.isdigit() else (1, 0, k))
        for k in keys:
            e = data[k]
            yield (
                (e.get(cf.image_key), e.get(cf.caption_key))
                if isinstance(e, dict)
                else (None, None)
            )
    elif cf.fmt == "jsonl":
        with cf.path.open("r", encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{cf.path}:{ln}: invalid JSON ({exc})") from None
                yield (
                    (e.get(cf.image_key), e.get(cf.caption_key))
                    if isinstance(e, dict)
                    else (None, None)
                )
    else:
        raise ValueError(f"unknown caption file format {cf.fmt!r}")


def index_images(root: Path, image_dirs: Sequence[str]) -> dict[str, str]:
    """``{file name: path relative to root}`` over the image folders (one directory listing each).

    A file name present in two folders is an error: the corpus refers to images by name only.
    """
    index: dict[str, str] = {}
    clashes: list[str] = []
    for rel in image_dirs:
        d = root / rel
        if not d.is_dir():
            log.warning("image folder %s does not exist", d)
            continue
        n = 0
        with os.scandir(d) as it:
            for entry in it:
                if Path(entry.name).suffix.lower() not in IMAGE_EXTENSIONS or not entry.is_file():
                    continue
                if entry.name in index:
                    clashes.append(f"{index[entry.name]} vs {rel}/{entry.name}")
                    continue
                index[entry.name] = f"{rel}/{entry.name}"
                n += 1
        log.info("indexed %d images in %s", n, d)
    if clashes:
        raise CorpusBuildError(
            f"{len(clashes)} image file names occur in several folders, e.g. {clashes[:3]}"
        )
    return index


def file_sha1(path: str | Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _new_stats() -> dict[str, Any]:
    return {
        "captions_in": 0,
        "captions_out": 0,
        "images_in": set(),
        "images_out": set(),
        "dropped": Counter(),
    }


def build_corpus(
    root: str | Path,
    caption_files: Sequence[CaptionFile],
    out_path: str | Path,
    image_dirs: Sequence[str] = DEFAULT_IMAGE_DIRS,
    exclude_sources: Iterable[str] = DEFAULT_EXCLUDED_SOURCES,
    exclude_coco_ids: Iterable[int] = (),
    exclusion_files: dict[str, str] | None = None,
    max_missing: float = 0.0,
) -> dict[str, Any]:
    """Build the corpus JSONL at ``out_path`` and its report ``<out>.report.json``.

    ``exclusion_files`` (``{path: sha1}``) is only recorded in the report. Raises
    :class:`CorpusBuildError` (after writing the report, without writing the corpus) when the
    fraction of otherwise-kept captions whose image is missing exceeds ``max_missing``.
    """
    root = Path(root)
    out_path = Path(out_path)
    excluded_sources = frozenset(exclude_sources)
    excluded_ids = frozenset(int(i) for i in exclude_coco_ids)
    index = index_images(root, image_dirs)

    stats: dict[str, dict[str, Any]] = defaultdict(_new_stats)
    seen: set[tuple[str, str]] = set()
    missing: list[str] = []
    kept: list[tuple[str, str, str]] = []
    invalid_examples: list[str] = []
    for cf in caption_files:
        n_file = 0
        for image, caption in iter_captions(cf):
            n_file += 1
            try:
                if not isinstance(image, str):
                    raise ValueError(f"image is not a path: {image!r}")
                source, stem = parse_name(image)
            except ValueError as exc:
                st = stats["?"]
                st["captions_in"] += 1
                st["dropped"]["invalid_entry"] += 1
                if len(invalid_examples) < _MISSING_EXAMPLES:
                    invalid_examples.append(f"{cf.path.name}: {exc}")
                continue
            name = image.rsplit("/", 1)[-1]
            st = stats[source]
            st["captions_in"] += 1
            st["images_in"].add(name)
            text = " ".join(caption.split()) if isinstance(caption, str) else ""
            cid = coco_id(source, stem)
            if source in excluded_sources:
                reason = "excluded_source"
            elif cid is not None and cid in excluded_ids:
                reason = "excluded_coco_id"
            elif not text:
                reason = "empty_caption"
            elif (name, text) in seen:
                reason = "duplicate"
            elif name not in index:
                reason = "missing_image"
                missing.append(name)
            else:
                seen.add((name, text))
                kept.append((index[name], text, source))
                st["captions_out"] += 1
                st["images_out"].add(name)
                continue
            st["dropped"][reason] += 1
        log.info("%s: %d entries read", cf.path, n_file)

    n_missing = len(missing)
    missing_frac = n_missing / max(1, n_missing + len(kept))
    report = _report(
        root, caption_files, image_dirs, excluded_sources, excluded_ids, exclusion_files, stats
    )
    report["missing_images"] = {
        "captions": n_missing,
        "images": len(set(missing)),
        "fraction_of_captions": round(missing_frac, 6),
        "max_missing": max_missing,
        "examples": sorted(set(missing))[:_MISSING_EXAMPLES],
    }
    report["invalid_examples"] = invalid_examples
    report_path = out_path.with_name(out_path.stem + ".report.json")
    if missing_frac > max_missing:
        report["output"] = None
        _write_json(report_path, report)
        raise CorpusBuildError(
            f"{n_missing} captions ({len(set(missing))} images, {missing_frac:.2%}) point at images "
            f"that are in none of {list(image_dirs)} under {root} (max_missing={max_missing}); "
            f"is the archive fully extracted? Report: {report_path}"
        )
    report["output"] = _write_corpus(out_path, kept)
    _write_json(report_path, report)
    return report


def _report(
    root: Path,
    caption_files: Sequence[CaptionFile],
    image_dirs: Sequence[str],
    excluded_sources: frozenset[str],
    excluded_ids: frozenset[int],
    exclusion_files: dict[str, str] | None,
    stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    per_source: dict[str, dict[str, Any]] = {}
    totals: Counter[str] = Counter()
    for src in sorted(stats, key=lambda s: -stats[s]["captions_in"]):
        st = stats[src]
        row = {
            "captions_in": st["captions_in"],
            "images_in": len(st["images_in"]),
            "captions_out": st["captions_out"],
            "images_out": len(st["images_out"]),
            "dropped": {r: st["dropped"][r] for r in DROP_REASONS if st["dropped"][r]},
        }
        per_source[src] = row
        for k in ("captions_in", "images_in", "captions_out", "images_out"):
            totals[k] += row[k]
        for r, n in row["dropped"].items():
            totals[f"dropped_{r}"] += n
    return {
        "root": str(root.resolve()),
        "inputs": [
            {"path": str(cf.path.resolve()), "format": cf.fmt, "sha1": file_sha1(cf.path)}
            for cf in caption_files
        ],
        "image_dirs": list(image_dirs),
        "decontamination": {
            "excluded_sources": sorted(excluded_sources),
            "coco_id_sources": sorted(COCO_ID_SOURCES),
            "excluded_coco_ids": len(excluded_ids),
            "exclusion_files": dict(exclusion_files or {}),
        },
        "per_source": per_source,
        "totals": dict(totals),
    }


def _write_corpus(path: Path, kept: list[tuple[str, str, str]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    h = hashlib.sha1()
    with tmp.open("w", encoding="utf-8") as fh:
        for image, caption, source in kept:
            line = json.dumps(
                {"image": image, "caption": caption, "source": source}, ensure_ascii=False
            )
            line += "\n"
            fh.write(line)
            h.update(line.encode("utf-8"))
    os.replace(tmp, path)
    return {"path": str(path.resolve()), "lines": len(kept), "sha1": h.hexdigest()}


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def uitviic_official_split_ids(
    train_ids: Iterable[str], test_ids: Iterable[str], corpus_names: Iterable[str]
) -> dict[str, list[str]]:
    """Recover the official UIT-ViIC dev split (924 images) that the local copy lacks.

    UIT-ViIC is 2,695 train / 924 dev / 231 test COCO images, and the corpus' ``vilc_*`` files hold
    all 3,850 of them; dev is therefore ``vilc - train - test``. Ids are COCO ids as strings.
    """
    train, test = {str(int(i)) for i in train_ids}, {str(int(i)) for i in test_ids}
    vilc: set[str] = set()
    for name in corpus_names:
        try:
            source, stem = parse_name(name)
        except ValueError:
            continue
        if source == "vilc" and stem.isdigit():
            vilc.add(str(int(stem)))
    if not test <= vilc or not train <= vilc:
        log.warning(
            "corpus vilc_* misses %d train / %d test UIT-ViIC images",
            len(train - vilc),
            len(test - vilc),
        )
    return {"test": sorted(test), "official_dev": sorted(vilc - train - test)}


__all__ = [
    "COCO_ID_SOURCES",
    "DEFAULT_EXCLUDED_SOURCES",
    "DEFAULT_IMAGE_DIRS",
    "CaptionFile",
    "CorpusBuildError",
    "build_corpus",
    "coco_id",
    "file_sha1",
    "index_images",
    "iter_captions",
    "parse_name",
    "uitviic_official_split_ids",
]
