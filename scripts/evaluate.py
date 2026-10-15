#!/usr/bin/env python
"""Evaluate a trained checkpoint on one or more eval splits (single process or ``torchrun``).

    python scripts/evaluate.py --config configs/downstream/ktvic_caption.yaml \\
        --checkpoint runs/ktvic_caption/checkpoints/best.pt --splits dev,test [--limit 8] [k=v ...]

The config may use ``model.from_checkpoint`` (the model section is inherited from that
checkpoint's sidecar); ``--checkpoint`` names the weights actually loaded, strictly, with
provenance-aware re-materialisation of excluded pristine towers. ``--limit N`` keeps the first N
records of every requested split (``data.evals.<split>.limit``). Outputs go to ``--out-dir``
(default ``<run.dir>/<run.name>/eval/<checkpoint stem>``): ``predictions/<split>_step<N>.jsonl``,
``metrics/<split>_step<N>.json`` and ``metrics/summary.json``; a table is printed on rank 0.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vivlmoe.config import load_config  # noqa: E402
from vivlmoe.training.distributed import (  # noqa: E402
    broadcast_object,
    destroy_distributed,
    init_distributed,
)
from vivlmoe.training.logging import setup_logging  # noqa: E402
from vivlmoe.utils.io import write_json  # noqa: E402
from vivlmoe.utils.misc import format_table  # noqa: E402

log = logging.getLogger("scripts.evaluate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, help="YAML config of the task (supports _include)")
    p.add_argument("--checkpoint", required=True, help="checkpoint file to evaluate (best.pt)")
    p.add_argument(
        "--splits", required=True, help="comma-separated names of data.evals entries, e.g. dev,test"
    )
    p.add_argument(
        "--limit", type=int, default=None, help="keep only the first N records per split"
    )
    p.add_argument(
        "--out-dir", default=None, help="output directory (default: <run_dir>/eval/<ckpt>)"
    )
    p.add_argument("overrides", nargs="*", help="key.path=value overrides (YAML-parsed values)")
    return p.parse_args(argv)


def _populate_registries() -> None:
    from vivlmoe.tasks import import_builtin_tasks

    import_builtin_tasks()
    import vivlmoe.data.adapters  # noqa: F401


def _prepare_model(
    cfg: Any, task: Any, ckpt: dict[str, Any], device: torch.device
) -> torch.nn.Module:
    """Build the task model, apply the run's freezing (dtype policy), load the weights strictly."""
    from vivlmoe.models.freezing import apply_frozen_dtype_policy
    from vivlmoe.models.towers import find_tower_paths
    from vivlmoe.training.checkpoint import ProvenanceTracker, load_model_state
    from vivlmoe.training.optim import apply_freezing, resolve_freeze_paths

    model = task.build_model()
    provenance = ProvenanceTracker.from_config(cfg.model, model, find_tower_paths(model))
    apply_freezing(model, resolve_freeze_paths(model, cfg.optim, task.always_frozen()))
    report = load_model_state(model, ckpt, provenance.as_dict())
    log.info(
        "loaded %d tensors from step %s (re-materialised: %s)",
        len(report.loaded_keys),
        ckpt.get("step"),
        report.rematerialised or "none",
    )
    model.to(device)
    amp = cfg.train.amp
    amp_dtype = None
    if amp.enabled and device.type == "cuda":
        amp_dtype = torch.bfloat16 if amp.dtype == "bf16" else torch.float16
    apply_frozen_dtype_policy(model, amp_dtype)
    model.eval()
    return model


def evaluate(
    cfg: Any, ctx: Any, checkpoint: str, splits: list[str], out_dir: Path
) -> dict[str, dict[str, float]]:
    from vivlmoe.tasks import build_task
    from vivlmoe.training.checkpoint import load_checkpoint_file
    from vivlmoe.training.state import EvalContext

    task = build_task(cfg, ctx)
    if not hasattr(task, "build_eval_loaders"):
        raise TypeError(
            f"task {cfg.task.name!r} does not expose build_eval_loaders() (evaluate.py needs it)"
        )
    ckpt = load_checkpoint_file(checkpoint)
    model = _prepare_model(cfg, task, ckpt, ctx.device)
    loaders = task.build_eval_loaders()
    evaluators = task.evaluators()
    missing = [s for s in splits if s not in loaders or s not in evaluators]
    if missing:
        raise KeyError(f"splits {missing} not in data.evals {sorted(loaders)}")
    step = int(ckpt.get("step", 0))
    results: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for split in splits:
            ec = EvalContext(
                ctx=ctx,
                step=step,
                run_dir=out_dir,
                split_name=split,
                writer=None,
                save_predictions=True,
            )
            local = evaluators[split].run(model, loaders[split], ec)
            if ctx.is_main and local is None:
                raise RuntimeError(f"evaluator '{split}' returned None on rank 0")
            res = broadcast_object(local, ctx)
            results[split] = {k: float(v) for k, v in res.items()}
            log.info("eval %s: %s", split, {k: round(v, 5) for k, v in results[split].items()})
    return results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise SystemExit("--splits must name at least one split")
    cfg = load_config(args.config, list(args.overrides))
    missing = [s for s in splits if s not in cfg.data.evals]
    if missing:
        raise KeyError(f"splits {missing} not in data.evals {sorted(cfg.data.evals)}")
    if args.limit is not None:  # re-load so the limit goes through the normal override path
        limits = [f"data.evals.{s}.limit={int(args.limit)}" for s in splits]
        cfg = load_config(args.config, [*args.overrides, *limits])
    ctx = init_distributed(cfg.dist)
    run_dir = Path(cfg.run.dir) / cfg.run.name
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "eval" / Path(args.checkpoint).stem
    setup_logging(ctx, None)
    log.info(
        "evaluating %s on %s -> %s (world %d)", args.checkpoint, splits, out_dir, ctx.world_size
    )
    _populate_registries()
    try:
        results = evaluate(cfg, ctx, args.checkpoint, splits, out_dir)
        if ctx.is_main:
            rows = [
                (split, k, f"{v:.5f}")
                for split, res in results.items()
                for k, v in sorted(res.items())
            ]
            print(format_table(["split", "metric", "value"], rows))
            summary = {
                "checkpoint": str(args.checkpoint),
                "config": str(args.config),
                "limit": args.limit,
                "overrides": list(args.overrides),
                "splits": results,
            }
            write_json(out_dir / "metrics" / "summary.json", summary)
            log.info("wrote %s", out_dir / "metrics" / "summary.json")
    finally:
        destroy_distributed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
