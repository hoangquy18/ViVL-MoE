#!/usr/bin/env python
"""Offline routing analysis of one checkpoint on one eval split (paper Sec. 6.5).

python scripts/analyze_routing.py --config configs/moe/sparse_train.yaml \\
    --checkpoint runs/moe_sparse/checkpoints/best.pt --split ktvic_dev [--max-batches 20] \\
    [--prev-snapshot runs/moe_sparse/routing/snapshot_step_0001000.pt] [--out-dir analysis/routing]

Runs the :class:`RoutingProbeEvaluator` once with the task's eval loader and writes
``<out-dir>/routing/step_<N>.json`` (+ PNG heatmap, + snapshot). A previous snapshot enables the
expert change rate. Single process or ``torchrun`` (ordered gather across ranks).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from vivlmoe.analysis.routing_stats import RoutingSnapshot  # noqa: E402
from vivlmoe.config import load_config  # noqa: E402
from vivlmoe.config.schema import ConfigError, RoutingProbeCfg  # noqa: E402
from vivlmoe.tasks.evaluators.routing import RoutingProbeEvaluator  # noqa: E402
from vivlmoe.training.checkpoint import (  # noqa: E402
    ProvenanceTracker,
    load_checkpoint_file,
    load_model_state,
)
from vivlmoe.training.distributed import destroy_distributed, init_distributed  # noqa: E402
from vivlmoe.training.logging import setup_logging  # noqa: E402
from vivlmoe.training.state import EvalContext  # noqa: E402

log = logging.getLogger("scripts.analyze_routing")

TASK_FORWARD_MODES = {
    "contrastive": "contrastive",
    "captioning": "joint_caption",
    "vqa": "joint_vqa",
    "vimacsa": "joint_cls",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", required=True, help="name of a data.evals entry")
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--prev-snapshot", default=None, help="snapshot_step_*.pt for the change rate")
    p.add_argument("--out-dir", default=None, help="default: <checkpoint dir>/../analysis")
    p.add_argument("overrides", nargs="*", help="key.path=value overrides")
    return p.parse_args(argv)


def _build_task(cfg, ctx):
    from vivlmoe.tasks import TASKS, build_task, import_builtin_tasks

    import_builtin_tasks()
    if cfg.task.name not in TASKS:
        raise ImportError(
            f"task '{cfg.task.name}' is not registered (milestone M5 provides 'contrastive', "
            f"M7 the downstream tasks); registered: {TASKS.keys()}"
        )
    return build_task(cfg, ctx)


def _eval_loader(task, split: str):
    builder = getattr(task, "build_eval_loaders", None)
    evals = builder() if callable(builder) else task.build_loaders().evals
    if split not in evals:
        raise ConfigError(f"split '{split}' not in the task's eval loaders {sorted(evals)}")
    return evals[split]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if not cfg.model.is_moe:
        raise ConfigError("analyze_routing needs an MoE model (model.moe is null)")
    if args.split not in cfg.data.evals:
        raise ConfigError(f"--split {args.split!r} is not in data.evals {sorted(cfg.data.evals)}")
    ctx = init_distributed(cfg.dist)
    setup_logging(ctx, None)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.resolve().parent.parent / "analysis"
    try:
        task = _build_task(cfg, ctx)
        model = task.build_model()
        ckpt = load_checkpoint_file(ckpt_path)
        tracker = ProvenanceTracker.from_config(cfg.model, model)
        load_model_state(model, ckpt, tracker.as_dict())
        model.to(ctx.device).eval()
        loader = _eval_loader(task, args.split)
        mode = getattr(task, "probe_forward_mode", TASK_FORWARD_MODES.get(cfg.task.name))
        if mode is None:
            raise ConfigError(f"no routing forward mode known for task '{cfg.task.name}'")
        step = int(ckpt.get("step", 0))
        probe = RoutingProbeEvaluator(
            "routing_probe",
            RoutingProbeCfg(
                split=args.split, every_steps=1, max_steps=None, max_batches=args.max_batches
            ),
            mode,
            cfg.model.encoder.num_layers,
        )
        if args.prev_snapshot:
            probe.previous = RoutingSnapshot.from_state_dict(
                torch.load(args.prev_snapshot, map_location="cpu", weights_only=False)
            )
        ec = EvalContext(ctx=ctx, step=step, run_dir=out_dir, split_name=args.split)
        with torch.no_grad():
            result = probe.run(model, loader, ec)
        if ctx.is_main:
            summary = {k: v for k, v in result.items() if "/layer" not in k}
            print(json.dumps(summary, indent=1, sort_keys=True))
            log.info("routing report written under %s", out_dir / "routing")
    finally:
        destroy_distributed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
