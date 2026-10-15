#!/usr/bin/env python
"""One-off image validation of a corpus config, writing the manifest cache used at training time.

    python scripts/build_manifest.py --config configs/pretrain/stage2_contrastive.yaml \
        [--source train | --source evals.ktvic_dev] [--level decode] [--split train] [--out DIR]

The cache (gzip JSON of bad image keys) is keyed by store + level + number of unique images, so
``data.<source>.manifest_cache`` in the config must point at the same directory/file. The list of
dropped records is written next to it as ``<kind>_<split>_dropped.json``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vivlmoe.config import load_config  # noqa: E402
from vivlmoe.data.registry import build_adapter  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", default="train", help="'train' or 'evals.<name>'")
    ap.add_argument("--split", default=None, help="override the source's split")
    ap.add_argument(
        "--level", choices=["exists", "decode"], default=None, help="override image_check"
    )
    ap.add_argument("--policy", choices=["skip", "error"], default=None)
    ap.add_argument("--out", default=None, help="manifest cache directory (default: from config)")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config, args.overrides)
    if args.source == "train":
        src = cfg.data.train
    elif args.source.startswith("evals."):
        src = cfg.data.evals.get(args.source[len("evals.") :])
    else:
        src = None
    if src is None:
        raise SystemExit(f"no data source {args.source!r} in {args.config}")
    if args.level:
        src.image_check = args.level
    if args.policy:
        src.missing_image_policy = args.policy
    if args.out:
        src.manifest_cache = args.out
    if not src.manifest_cache:
        raise SystemExit("set data.<source>.manifest_cache in the config or pass --out")
    split = args.split or src.split
    out_dir = Path(src.manifest_cache)
    t0 = time.time()
    adapter = build_adapter(src, run_dir=out_dir)
    records = adapter.load_split(split)
    dropped = adapter.write_dropped(split, out_dir)
    print(
        f"{src.kind}/{split}: {len(records)} records kept, {len(adapter.dropped.get(split, []))} dropped "
        f"in {time.time() - t0:.1f}s; cache {adapter.manifest_cache_path(adapter.source_split(split))}; "
        f"dropped list {dropped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
