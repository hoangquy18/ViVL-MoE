#!/usr/bin/env python
"""Push / pull the pretraining corpus to Hugging Face dataset repos as tar shards.

    python scripts/hf_corpus.py push --repo USER/vivlmoe-pretrain [--public] [--root dataset] \
        [--shard-gb 2] [--extras-repo USER/vivlmoe-pretrain-extras --extra NAME=DIR ...] [--dry-run]
    python scripts/hf_corpus.py pull --repo USER/vivlmoe-pretrain --root /data/vivlmoe/dataset \
        [--extras-repo USER/vivlmoe-pretrain-extras --extras-dir /data/vivlmoe] [--workers 8]

Log in once with ``hf auth login``; the token never passes through this script. The image repo is
private unless --public. Extra folders always go to a private repo (PhoMT forbids redistribution,
KTVIC / UIT-ViIC hold benchmark test sets): --extras-repo, which defaults to --repo when that one is
private. Extras are pushed first (small: a permission problem fails fast), then the images; re-running
the same command resumes. pull restores <root>/images, <root>/corpus and <extras-dir>/<NAME>.
See vivlmoe/data/corpus_shards.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vivlmoe.data.corpus_shards import (  # noqa: E402
    CORPUS_FILES,
    Shard,
    corpus_images,
    dir_items,
    plan_shards,
    pull,
    pull_extras,
    push,
    push_extras,
)

_PUBLIC_TERMS = """
## Sources and terms

Every image and caption comes from a third-party source and keeps that source's license and terms:
MS-COCO 2017 (Vietnamese translations of the captions; images belong to their Flickr owners),
UIT-ViIC (train split only), ViVQA, EVJVQA, Oxford-102 Flowers, Vietnamese web images and a
LAION-Aesthetics sample (images belong to their owners). Research use only; no rights are granted
beyond those of the original sources. Removal requests: open a discussion on this repository.
"""


def _readme(root: Path, shards: list[Shard], public: bool) -> str:
    report = json.loads((root / CORPUS_FILES[1]).read_text(encoding="utf-8"))
    t = report["totals"]
    gb = sum(s.bytes for s in shards) / 1e9
    lead = (
        "Temporary working copy of the image-caption corpus used to pretrain a re-implementation of\n"
        '"One Tower, Many Experts" (ViVL-MoE); it will be deleted once training is done.'
        if public
        else 'Private working copy for re-implementing "One Tower, Many Experts" (ViVL-MoE).'
    )
    return f"""---
license: other
pretty_name: ViVL-MoE pretraining corpus
language: [vi, en]
task_categories: [image-to-text]
---
# ViVL-MoE pretraining corpus

{lead}

* `corpus/pretrain_corpus.jsonl`: {t["captions_out"]:,} captions (Vietnamese and English) over
  {t["images_out"]:,} images, decontaminated against the ViMMU-Bench evaluation splits
  (`corpus/pretrain_corpus.report.json`)
* `images/shard-*.tar`: the images ({gb:.1f} GB in {len(shards)} uncompressed tars, members
  `images/<name>`); `shards.json` lists the files of every shard
{_PUBLIC_TERMS if public else ""}"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push", help="pack and upload (resumable)")
    p.add_argument("--repo", required=True, help="USER/NAME of the image dataset repo")
    p.add_argument("--public", action="store_true", help="make the image repo public")
    p.add_argument("--root", default="dataset", help="dataset root with images/ and corpus/")
    p.add_argument("--shard-gb", type=float, default=2.0, help="target shard size in GB")
    p.add_argument("--extra", action="append", default=[], metavar="NAME=DIR", help="repeatable")
    p.add_argument("--extras-repo", default=None, help="private repo for --extra (default --repo)")
    p.add_argument(
        "--work-dir", default=None, help="scratch for shards (default <root>/.hf_upload)"
    )
    p.add_argument("--dry-run", action="store_true", help="print the shard plan, upload nothing")
    q = sub.add_parser("pull", help="download and unpack")
    q.add_argument("--repo", required=True)
    q.add_argument("--root", required=True, help="dataset root to restore images/ and corpus/ into")
    q.add_argument("--extras-repo", default=None, help="repo holding extras/ (default --repo)")
    q.add_argument("--extras-dir", default=None, help="where extras/<NAME>.tar are unpacked")
    q.add_argument("--workers", type=int, default=4, help="shards downloaded in parallel")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.cmd == "pull":
        from huggingface_hub import HfApi

        api = HfApi()
        print(pull(api, args.repo, Path(args.root), args.workers))
        if args.extras_dir:
            print(pull_extras(api, args.extras_repo or args.repo, Path(args.extras_dir)))
        return 0

    root = Path(args.root)
    extras: dict[str, Path] = {}
    for spec in args.extra:
        name, _, d = spec.partition("=")
        if not name or not d or "/" in name or name.startswith(".") or not Path(d).is_dir():
            ap.error(f"--extra expects NAME=DIR with an existing DIR, got {spec!r}")
        extras[name] = Path(d)
    if extras and args.public and not args.extras_repo:
        ap.error("--extra with --public needs a separate private --extras-repo")
    shards = plan_shards(root, corpus_images(root / CORPUS_FILES[0]), int(args.shard_gb * 1e9))
    n_images = sum(len(s.files) for s in shards)
    vis = "public" if args.public else "private"
    print(
        f"{args.repo} ({vis}): {n_images} images in {len(shards)} shards, "
        f"{sum(s.bytes for s in shards) / 1e9:.1f} GB + corpus files"
    )
    for name, d in sorted(extras.items()):
        items = dir_items(d, name)
        size = sum(src.stat().st_size for src, _ in items) / 1e9
        print(
            f"{args.extras_repo or args.repo} (private): extra {name}, {len(items)} files, {size:.2f} GB from {d}"
        )
    if args.dry_run:
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception as exc:  # no token / invalid token
        print(
            f"ERROR: not logged in to Hugging Face ({exc}); run `hf auth login` first",
            file=sys.stderr,
        )
        return 1
    print(f"logged in as {user}")
    work_dir = Path(args.work_dir) if args.work_dir else root / ".hf_upload"
    if extras:
        print(push_extras(api, args.extras_repo or args.repo, extras, work_dir))
    done = push(
        api,
        args.repo,
        root,
        shards,
        work_dir,
        _readme(root, shards, args.public),
        private=not args.public,
    )
    if work_dir.is_dir() and not any(work_dir.iterdir()):
        work_dir.rmdir()
    print(done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
