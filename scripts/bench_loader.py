#!/usr/bin/env python
"""Measure DataLoader throughput (samples/s) for a config's training source without a model.

    python scripts/bench_loader.py --config configs/pretrain/stage2_contrastive.yaml \
        [--batches 50] [--workers 8] [data.batch_size=64 ...]

Use it on the training box to size ``data.num_workers`` / ``prefetch_factor`` so the frozen ViT-H
is never starved. Supports contrastive corpora, captioning/VQA sources and ViMACSA.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vivlmoe.config import load_config  # noqa: E402
from vivlmoe.data.builders import (  # noqa: E402
    build_image_text_train,
    build_parallel_text,
    build_transform_for,
    build_vimacsa,
    build_vqa,
    draft_size_for,
)
from vivlmoe.data.collators import (  # noqa: E402
    ContrastiveCollator,
    MultiImageClsCollator,
    Seq2SeqCollator,
    TeacherCollator,
)
from vivlmoe.data.loaders import build_train_loader  # noqa: E402
from vivlmoe.data.tokenization import build_tokenizers  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config, args.overrides)
    if args.workers is not None:
        cfg.data.num_workers = args.workers
    src = cfg.data.train
    if src is None:
        raise SystemExit("config has no data.train")
    task = cfg.task.name
    spec, train_tf, _ = build_transform_for(cfg)
    draft = draft_size_for(cfg, spec)
    tk = cfg.data.tokenizer
    groups = None
    if task == "stage1_teacher":
        tokens = build_tokenizers(cfg.model, cfg.data, need_clip=True)
        dataset = build_parallel_text(src)
        collate = TeacherCollator(tokens.require_clip(), tokens.vi, tk.en_max_len, tk.vi_max_len)
    elif task == "contrastive":
        tokens = build_tokenizers(cfg.model, cfg.data)
        dataset, groups = build_image_text_train(src, train_tf, draft)
        collate = ContrastiveCollator(tokens.vi, tk.vi_max_len)
    elif task in ("captioning", "vqa"):
        tokens = build_tokenizers(cfg.model, cfg.data, need_decoder=True)
        if task == "vqa":
            dataset, template = build_vqa(
                src, train_tf, draft, expand_answers=cfg.task.train_on_all_answers
            )
        else:
            dataset, groups = build_image_text_train(src, train_tf, draft)
            template = cfg.task.prompt_template
        collate = Seq2SeqCollator(
            tokens.vi,
            tokens.require_decoder(),
            task == "vqa",
            True,
            template,
            cfg.task.max_source_len,
            cfg.task.max_target_len,
        )
    elif task == "vimacsa":
        tokens = build_tokenizers(cfg.model, cfg.data)
        dataset = build_vimacsa(
            src, train_tf, (spec.crop_size, spec.crop_size), cfg.task.max_images, draft
        )
        collate = MultiImageClsCollator(
            tokens.vi,
            cfg.task.aspects,
            cfg.task.use_image_tags,
            tk.cls_max_len,
            cfg.task.legacy_parity,
        )
    else:
        raise SystemExit(f"unsupported task {task!r}")

    loader = build_train_loader(dataset, collate, cfg.data, seed=cfg.run.seed, groups=groups)
    n_samples, t0 = 0, None
    for i, batch in enumerate(loader):
        if i == args.warmup:
            t0, n_samples = time.time(), 0
        first = next(v for v in batch.values() if hasattr(v, "shape"))
        n_samples += int(first.shape[0])
        if i + 1 >= args.warmup + args.batches:
            break
    if t0 is None:
        raise SystemExit("not enough batches for the requested warmup")
    dt = time.time() - t0
    print(
        f"{task}: {n_samples} samples in {dt:.1f}s = {n_samples / dt:.1f} samples/s "
        f"(batch {cfg.data.batch_size}, workers {cfg.data.num_workers}, transform {spec.crop_size}px)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
