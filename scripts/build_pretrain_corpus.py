#!/usr/bin/env python
"""Build the decontaminated image-caption pretraining corpus from the raw dump in ``dataset/``.

    python scripts/build_pretrain_corpus.py \
        [--root dataset] [--vi <root>/filtered_data.json] [--en <root>/english.jsonl] \
        [--out dataset/corpus/pretrain_corpus.jsonl] [--image-dir DIR ...] \
        [--exclude-sources vitextvqa,openvivqa,viocrvqa] [--exclude-coco-ids FILE ...] \
        [--no-decontamination] [--max-missing 0.0]

Writes one JSONL line per caption (image path relative to ``--root``) for the ``jsonl`` adapter of
``configs/data/pretrain_corpus.yaml``, plus ``<out stem>.report.json`` with every drop counted per
source. Default decontamination (paper Sec. 4.2: evaluation-split images are removed): drop the
ViTextVQA / OpenViVQA / ViOCRVQA conversions whole, and drop the UIT-ViIC test, official dev and
dev-holdout images (by COCO id) from coco2017 / vilc / vivqa. See vivlmoe/data/corpus_prep.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vivlmoe.data.corpus_prep import (  # noqa: E402
    DEFAULT_EXCLUDED_SOURCES,
    DEFAULT_IMAGE_DIRS,
    CaptionFile,
    CorpusBuildError,
    build_corpus,
)
from vivlmoe.data.splits import ids_sha1, load_ids, resolve_split_path  # noqa: E402

DEFAULT_EXCLUDE_IDS = (
    "configs/data/splits/uitviic_test_image_ids.json",
    "configs/data/splits/uitviic_official_dev_image_ids.json",
    "configs/data/splits/uitviic_dev_image_ids.json",
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root", default="dataset", help="dataset root; image paths are relative to it"
    )
    ap.add_argument(
        "--vi",
        default=None,
        help="Vietnamese json_dict captions (default <root>/filtered_data.json)",
    )
    ap.add_argument(
        "--en", default=None, help="English jsonl captions (default <root>/english.jsonl; '' skips)"
    )
    ap.add_argument("--out", default=None, help="default <root>/corpus/pretrain_corpus.jsonl")
    ap.add_argument(
        "--image-dir",
        action="append",
        default=None,
        help=f"image folder relative to --root, repeatable (default {list(DEFAULT_IMAGE_DIRS)})",
    )
    ap.add_argument(
        "--exclude-sources",
        default=",".join(DEFAULT_EXCLUDED_SOURCES),
        help="comma-separated sources dropped whole ('' keeps all)",
    )
    ap.add_argument(
        "--exclude-coco-ids",
        action="append",
        default=None,
        help=f"JSON id list of COCO images to drop, repeatable (default {list(DEFAULT_EXCLUDE_IDS)})",
    )
    ap.add_argument("--no-decontamination", action="store_true", help="keep every source and image")
    ap.add_argument(
        "--max-missing",
        type=float,
        default=0.0,
        help="max fraction of captions whose image may be missing before the build fails",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    root = Path(args.root)
    files = [CaptionFile(Path(args.vi or root / "filtered_data.json"), "json_dict", "image_id")]
    if args.en != "":
        files.append(CaptionFile(Path(args.en or root / "english.jsonl"), "jsonl", "image"))
    out = Path(args.out or root / "corpus" / "pretrain_corpus.jsonl")

    sources: list[str] = []
    ids: set[int] = set()
    id_files: dict[str, str] = {}
    if not args.no_decontamination:
        sources = [s.strip() for s in args.exclude_sources.split(",") if s.strip()]
        for f in args.exclude_coco_ids or DEFAULT_EXCLUDE_IDS:
            lst = load_ids(f)
            ids.update(int(i) for i in lst)
            id_files[str(resolve_split_path(f))] = ids_sha1(lst)

    try:
        report = build_corpus(
            root,
            files,
            out,
            image_dirs=args.image_dir or DEFAULT_IMAGE_DIRS,
            exclude_sources=sources,
            exclude_coco_ids=ids,
            exclusion_files=id_files,
            max_missing=args.max_missing,
        )
    except CorpusBuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"{'source':12s} {'captions in':>11s} {'out':>9s} {'images in':>9s} {'out':>8s}  dropped")
    for src, row in report["per_source"].items():
        dropped = ", ".join(f"{k}={v}" for k, v in row["dropped"].items())
        print(
            f"{src:12s} {row['captions_in']:11d} {row['captions_out']:9d} "
            f"{row['images_in']:9d} {row['images_out']:8d}  {dropped}"
        )
    t, o = report["totals"], report["output"]
    print(
        f"total: {t['captions_out']} / {t['captions_in']} captions, {t['images_out']} / "
        f"{t['images_in']} images -> {o['path']} (sha1 {o['sha1'][:12]}); "
        f"report {out.with_name(out.stem + '.report.json')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
