#!/usr/bin/env python
"""Shrink the pretraining corpus images in place so their shorter side is at most ``--short-side``.

    python scripts/shrink_images.py --root /data/vivlmoe/dataset [--short-side 512] [--quality 90]
        [--workers 8] [--dry-run]

For disk-limited GPU boxes. The vision towers squash every image to 512 x 512 (C-RADIO; SigLIP: 384),
so an image whose shorter side is already <= 512 is left untouched, and a larger one keeps both sides
>= 512: the model input loses nothing but a second resampling step. The 4-8 MP crawled images
dominate the corpus size.

Each image is resized with Lanczos (aspect kept) and re-encoded in its own format (JPEG/MPO -> JPEG
at ``--quality``, PNG, WebP), keeping its EXIF and ICC profile. The result goes to a temporary file
that replaces the original only when it is smaller: an interrupted run leaves no half-written image,
and a rerun skips what is already small. Files Pillow cannot decode, and other formats, are left
untouched (the manifest step drops undecodable ones). File names do not change, so the corpus JSONL
stays valid.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

log = logging.getLogger("shrink_images")
TMP_SUFFIX = ".shrink.tmp"
SAVE_FORMAT = {"JPEG": "JPEG", "MPO": "JPEG", "PNG": "PNG", "WEBP": "WEBP"}


def iter_images(images_dir: Path) -> Iterator[Path]:
    for dirpath, _dirs, files in os.walk(images_dir):
        for name in files:
            if not name.startswith(".") and not name.endswith(TMP_SUFFIX):
                yield Path(dirpath) / name


def target_size(size: tuple[int, int], short_side: int) -> tuple[int, int] | None:
    """New ``(w, h)`` with the shorter side at ``short_side``, or None when already small enough."""
    w, h = size
    if min(w, h) <= short_side:
        return None
    scale = short_side / min(w, h)
    return max(short_side, round(w * scale)), max(short_side, round(h * scale))


def shrink_one(
    path: Path, short_side: int, quality: int, dry_run: bool = False
) -> tuple[str, int, int]:
    """``(status, bytes_before, bytes_after)``; status: shrunk | small | kept | skipped | error."""
    before = path.stat().st_size
    tmp = path.with_name(path.name + TMP_SUFFIX)
    try:
        with Image.open(path) as img:
            new = target_size(img.size, short_side)
            if new is None:
                return "small", before, before
            fmt = SAVE_FORMAT.get(img.format or "")
            if fmt is None:
                return "skipped", before, before
            if dry_run:
                return "shrunk", before, before
            exif, icc = img.info.get("exif"), img.info.get("icc_profile")
            if img.format in ("JPEG", "MPO"):
                img.draft(img.mode, new)  # DCT downscale-on-decode, never below the target size
            im = (
                img.convert("RGBA" if "transparency" in img.info else "RGB")
                if img.mode in ("P", "1")
                else img
            )
            if fmt == "JPEG" and im.mode not in ("RGB", "L", "CMYK"):
                im = im.convert("RGB")
            im = im.resize(new, Image.Resampling.LANCZOS)
            extra = {k: v for k, v in (("exif", exif), ("icc_profile", icc)) if v}
            if fmt == "JPEG":
                im.save(tmp, format="JPEG", quality=quality, optimize=True, **extra)
            elif fmt == "WEBP":
                im.save(tmp, format="WEBP", quality=quality, **extra)
            else:
                im.save(tmp, format="PNG", optimize=True, **extra)
        after = tmp.stat().st_size
        if after >= before:
            tmp.unlink()
            return "kept", before, before
        os.replace(tmp, path)
        return "shrunk", before, after
    except Exception as exc:  # noqa: BLE001 - a bad file must not stop the run
        tmp.unlink(missing_ok=True)
        log.debug("cannot shrink %s: %s", path, exc)
        return "error", before, before


def _job(args: tuple[str, int, int, bool]) -> tuple[str, int, int]:
    path, short_side, quality, dry_run = args
    return shrink_one(Path(path), short_side, quality, dry_run)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", required=True, help="dataset root holding images/")
    ap.add_argument("--short-side", type=int, default=512)
    ap.add_argument("--quality", type=int, default=90, help="JPEG / WebP quality")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--dry-run", action="store_true", help="only count what would be shrunk")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    images = Path(args.root) / "images"
    if not images.is_dir():
        raise SystemExit(f"{images} does not exist")
    stale = [p for p in images.rglob(f"*{TMP_SUFFIX}")]
    for p in stale:  # leftovers of an interrupted run; the originals are intact
        p.unlink()
    paths = [str(p) for p in iter_images(images)]
    log.info("%d files under %s (removed %d stale temporary files)", len(paths), images, len(stale))

    counts: Counter[str] = Counter()
    jobs = ((p, args.short_side, args.quality, args.dry_run) for p in paths)

    def tally(results: Iterator[tuple[str, int, int]]) -> tuple[int, int]:
        total_before = total_after = 0
        for i, (status, before, after) in enumerate(results, 1):
            counts[status] += 1
            total_before += before
            total_after += after
            if i % 20000 == 0 or i == len(paths):
                log.info(
                    "%d/%d files, %s, %.1f GB -> %.1f GB",
                    i,
                    len(paths),
                    dict(counts),
                    total_before / 1e9,
                    total_after / 1e9,
                )
        return total_before, total_after

    if args.workers <= 1:
        total_before, total_after = tally(map(_job, jobs))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            total_before, total_after = tally(ex.map(_job, jobs, chunksize=256))
    verb = "would shrink" if args.dry_run else "shrank"
    log.info(
        "%s %d images (short side > %d); saved %.1f GB; %d undecodable left as is",
        verb,
        counts["shrunk"],
        args.short_side,
        (total_before - total_after) / 1e9,
        counts["error"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
