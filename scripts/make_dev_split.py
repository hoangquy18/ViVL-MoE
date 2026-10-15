#!/usr/bin/env python
"""Write the committed dev holdout id lists (Appendix A: 20 % of KTVIC / UIT-ViIC train by image id).

    python scripts/make_dev_split.py [--ktvic-root DIR] [--uitviic-root DIR] [--fraction 0.2] [--seed 0]
    python scripts/make_dev_split.py --only uitviic_official --uitviic-corpus dataset/filtered_data.json

Outputs ``configs/data/splits/{ktvic,uitviic}_dev_image_ids.json`` (sorted JSON lists) and prints
their sha1, which is what checkpoints record as ``dev_split_sha1``. With ``--uitviic-corpus`` it also
writes ``uitviic_{test,official_dev}_image_ids.json``: the official UIT-ViIC dev split is missing from
the local copy but is recovered from the corpus' ``vilc_*`` images (all 3,850 UIT-ViIC images). Both
lists are excluded from the pretraining corpus by ``scripts/build_pretrain_corpus.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vivlmoe.config.schema import DataSourceCfg  # noqa: E402
from vivlmoe.data.corpus_prep import (  # noqa: E402
    CaptionFile,
    iter_captions,
    uitviic_official_split_ids,
)
from vivlmoe.data.registry import build_adapter  # noqa: E402
from vivlmoe.data.splits import holdout_image_ids, ids_sha1, save_ids  # noqa: E402
from vivlmoe.utils.hf import repo_root  # noqa: E402

DEFAULT_KTVIC = "/Users/gos/Documents/Moe/moe/ktvic_dataset"
DEFAULT_UITVIIC = "/Users/gos/Documents/Moe/moe/uit-vilc/uitvic_dataset"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ktvic-root", default=DEFAULT_KTVIC)
    ap.add_argument("--uitviic-root", default=DEFAULT_UITVIIC)
    ap.add_argument("--fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(repo_root() / "configs" / "data" / "splits"))
    ap.add_argument("--only", choices=["ktvic", "uitviic", "uitviic_official"], default=None)
    ap.add_argument(
        "--uitviic-corpus",
        default=None,
        help="json_dict pretraining corpus with vilc_* images: also write the official test/dev ids",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.only == "uitviic_official" and not args.uitviic_corpus:
        ap.error("--only uitviic_official needs --uitviic-corpus")

    targets = {"ktvic": args.ktvic_root, "uitviic": args.uitviic_root}
    if args.only:
        targets = {k: v for k, v in targets.items() if k == args.only}
    for kind, root in targets.items():
        # image_check=none: only ids are needed here (the training pipeline validates images)
        cfg = DataSourceCfg(kind=kind, root=root, image_check="none")
        records = build_adapter(cfg).load_split("train")
        ids = holdout_image_ids(records.image_ids(), fraction=args.fraction, seed=args.seed)
        out = save_ids(Path(args.out_dir) / f"{kind}_dev_image_ids.json", ids)
        n_images = len(set(records.image_ids()))
        print(f"{kind}: {len(ids)} / {n_images} images held out -> {out} (sha1 {ids_sha1(ids)})")
    if args.uitviic_corpus and args.only in (None, "uitviic_official"):
        split_ids = {}
        for split in ("train", "test"):
            cfg = DataSourceCfg(kind="uitviic", root=args.uitviic_root, image_check="none")
            split_ids[split] = build_adapter(cfg).load_split(split).image_ids()
        names = (
            img
            for img, _ in iter_captions(
                CaptionFile(Path(args.uitviic_corpus), "json_dict", "image_id")
            )
        )
        official = uitviic_official_split_ids(split_ids["train"], split_ids["test"], names)
        for name, ids in official.items():
            out = save_ids(Path(args.out_dir) / f"uitviic_{name}_image_ids.json", ids)
            print(f"uitviic {name}: {len(ids)} images -> {out} (sha1 {ids_sha1(ids)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
