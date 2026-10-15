#!/usr/bin/env python
"""Train any task from a YAML config (single process or ``torchrun --nproc_per_node N``).

python scripts/train.py --config configs/pretrain/stage1_teacher.yaml run.name=s1 train.epochs=1
torchrun --nproc_per_node 4 scripts/train.py --config ... --resume auto
python scripts/train.py --config ... --dry-run     # build everything, 2 optimizer steps, report
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vivlmoe.config import load_config, save_resolved  # noqa: E402
from vivlmoe.training.distributed import destroy_distributed, init_distributed  # noqa: E402
from vivlmoe.training.logging import setup_logging  # noqa: E402
from vivlmoe.utils.io import write_json, write_text  # noqa: E402

log = logging.getLogger("scripts.train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, help="YAML config (supports _include)")
    p.add_argument(
        "--resume", default=None, help="'auto' (last.pt of the run dir) or a checkpoint path"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="build everything, run 2 optimizer steps, report"
    )
    p.add_argument("overrides", nargs="*", help="key.path=value overrides (YAML-parsed values)")
    return p.parse_args(argv)


def _populate_registries() -> None:
    """Import task and data adapter modules so their registries are filled (missing ones warn)."""
    from vivlmoe.tasks import import_builtin_tasks

    import_builtin_tasks()
    try:
        import vivlmoe.data.adapters  # noqa: F401  (registers dataset kinds when present)
    except ImportError as exc:
        log.warning("data adapters not importable yet: %s", exc)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *(argv if argv is not None else sys.argv[1:]),
    ]
    cfg = load_config(args.config, args.overrides)
    ctx = init_distributed(cfg.dist)
    run_dir = Path(cfg.run.dir) / cfg.run.name
    if ctx.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_resolved(cfg, run_dir / "config.resolved.yaml", argv=command)
        write_text(run_dir / "command.txt", " ".join(command) + "\n")
    setup_logging(ctx, run_dir if not args.dry_run else None)
    log.info("config %s -> run dir %s (world %d)", args.config, run_dir, ctx.world_size)
    _populate_registries()
    from vivlmoe.tasks import build_task
    from vivlmoe.training.trainer import Trainer

    try:
        task = build_task(cfg, ctx)
        trainer = Trainer(cfg, task, ctx, run_dir, resume=args.resume, dry_run=args.dry_run)
        result = trainer.fit()
        if ctx.is_main:
            name = "dry_run_report.json" if args.dry_run else "final_metrics.json"
            write_json(run_dir / name, result)
            log.info("wrote %s", run_dir / name)
    finally:
        destroy_distributed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
