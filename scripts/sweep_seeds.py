#!/usr/bin/env python
"""Run one config over several seeds (paper Sec. 6.1: 5 seeds) and aggregate the final metrics.

    python scripts/sweep_seeds.py --config configs/downstream/ktvic_caption.yaml --seeds 0 1 2 3 4 \\
        [--nproc 4] [--dry-run] [key.path=value ...]

Each seed is a sequential subprocess (``torchrun --nproc_per_node N scripts/train.py`` when
``--nproc > 1``, else ``python scripts/train.py``) with ``run.seed=<s> run.name=<name>_seed<s>``;
the first failure aborts the sweep. ``final_metrics.json`` of every run is aggregated into
``<run.dir>/<name>_sweep/sweep_summary.json`` (``{metric: {mean, std, values}}``; std uses
``ddof=1`` when n > 1) and a Markdown table next to it. ``--dry-run`` only prints the commands.

``--vary-upcycle-seed`` (plan decision 21, upcycling-seed ablation): before each training run
``scripts/upcycle.py --config <--upcycle-config> --seed <s> --out <run.dir>/<name>_upseed<s>/upcycled.pt``
re-upcycles the dense checkpoint with that seed and the training run points
``model.from_checkpoint`` / ``init_from.path`` at it (the config must inherit its model from an
upcycled checkpoint, e.g. ``configs/moe/sparse_train.yaml``).
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vivlmoe.config.loader import load_yaml_with_includes, parse_overrides  # noqa: E402
from vivlmoe.utils.io import write_json, write_text  # noqa: E402

log = logging.getLogger("scripts.sweep_seeds")
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"
UPCYCLE_SCRIPT = REPO_ROOT / "scripts" / "upcycle.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--nproc", type=int, default=1, help="GPUs per run (torchrun when > 1)")
    p.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    p.add_argument("--python", default=sys.executable, help="interpreter for single-process runs")
    p.add_argument(
        "--vary-upcycle-seed",
        action="store_true",
        help="re-run scripts/upcycle.py with --seed <s> per seed and train from that checkpoint",
    )
    p.add_argument(
        "--upcycle-config",
        default=str(REPO_ROOT / "configs" / "moe" / "upcycle.yaml"),
        help="upcycling config for --vary-upcycle-seed (default: configs/moe/upcycle.yaml)",
    )
    p.add_argument(
        "--upcycle-args",
        default="",
        help='extra arguments for scripts/upcycle.py as one string (e.g. "--noise-std-rel 0.02")',
    )
    p.add_argument("overrides", nargs="*", help="key.path=value overrides forwarded to train.py")
    return p.parse_args(argv)


def base_run(config: str, overrides: list[str]) -> tuple[str, Path]:
    """``(run name, run dir)`` from the raw YAML (+ overrides) without validating the config."""
    raw = load_yaml_with_includes(config)
    ov = parse_overrides(overrides)
    run = dict(raw.get("run") or {})
    run.update(ov.get("run") or {})
    name = str(run.get("name") or Path(config).stem)
    if name == "CHANGE_ME":
        name = Path(config).stem
    return name, Path(str(run.get("dir") or "runs"))


def build_command(
    config: str, seed: int, run_name: str, nproc: int, python: str, overrides: list[str]
) -> list[str]:
    if nproc > 1:
        launcher = ["torchrun", "--standalone", "--nproc_per_node", str(nproc), str(TRAIN_SCRIPT)]
    else:
        launcher = [python, str(TRAIN_SCRIPT)]
    return [*launcher, "--config", config, *overrides, f"run.seed={seed}", f"run.name={run_name}"]


def upcycled_path(run_root: Path, run_name: str, seed: int) -> Path:
    return run_root / f"{run_name}_upseed{seed}" / "upcycled.pt"


def build_upcycle_command(
    upcycle_config: str, seed: int, out: Path, python: str, extra: list[str]
) -> list[str]:
    """``python scripts/upcycle.py --config CFG --seed S --out OUT [extra]`` (single process)."""
    return [
        python,
        str(UPCYCLE_SCRIPT),
        "--config",
        upcycle_config,
        "--seed",
        str(seed),
        "--out",
        str(out),
        *extra,
    ]


def plan_runs(args: argparse.Namespace, name: str, run_root: Path) -> list[dict[str, Any]]:
    """Per seed: ``{seed, run_name, upcycle: cmd | None, train: cmd}``."""
    plan: list[dict[str, Any]] = []
    for seed in args.seeds:
        run_name = f"{name}_seed{seed}"
        overrides = list(args.overrides)
        upcycle_cmd = None
        if args.vary_upcycle_seed:
            out = upcycled_path(run_root, name, seed)
            upcycle_cmd = build_upcycle_command(
                args.upcycle_config, seed, out, args.python, shlex.split(args.upcycle_args)
            )
            overrides += [f"model.from_checkpoint={out}", f"init_from.path={out}"]
        train_cmd = build_command(args.config, seed, run_name, args.nproc, args.python, overrides)
        plan.append(
            {"seed": seed, "run_name": run_name, "upcycle": upcycle_cmd, "train": train_cmd}
        )
    return plan


def _run(cmd: list[str], what: str, seed: int) -> None:
    log.info("seed %d %s: %s", seed, what, " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise SystemExit(
            f"seed {seed}: {what} failed with exit code {proc.returncode}; aborting the sweep"
        )


def aggregate(results: dict[int, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """``{seed: final_metrics}`` -> ``{metric: {mean, std, values}}`` over the metrics common to all seeds."""
    flat: dict[int, dict[str, float]] = {}
    for seed, fm in results.items():
        d: dict[str, float] = {}
        for k, v in (fm.get("final") or {}).items():
            d[k] = float(v)
        for k, v in (fm.get("last_eval") or {}).items():
            d.setdefault(f"last/{k}", float(v))
        if fm.get("best_metric") is not None:
            d["best_metric"] = float(fm["best_metric"])
        flat[seed] = d
    keys = sorted(set.intersection(*(set(d) for d in flat.values()))) if flat else []
    seeds = sorted(flat)
    out: dict[str, dict[str, Any]] = {}
    for k in keys:
        values = [flat[s][k] for s in seeds]
        arr = np.asarray(values, dtype=np.float64)
        out[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "values": {str(s): v for s, v in zip(seeds, values, strict=True)},
        }
    return out


def markdown_table(summary: dict[str, dict[str, Any]], seeds: list[int]) -> str:
    header = "| metric | mean | std | " + " | ".join(f"seed {s}" for s in seeds) + " |"
    sep = "|" + "---|" * (3 + len(seeds))
    lines = [header, sep]
    for k, d in summary.items():
        vals = " | ".join(f"{d['values'][str(s)]:.4f}" for s in seeds)
        lines.append(f"| {k} | {d['mean']:.4f} | {d['std']:.4f} | {vals} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    name, run_root = base_run(args.config, args.overrides)
    plan = plan_runs(args, name, run_root)
    for item in plan:
        seed = item["seed"]
        if item["upcycle"] is not None:
            print(f"[upcycle seed {seed}] " + " ".join(shlex.quote(c) for c in item["upcycle"]))
        print(f"[seed {seed}] " + " ".join(shlex.quote(c) for c in item["train"]))
    if args.dry_run:
        print(f"[dry-run] {len(plan)} run(s); summary would go to {run_root / (name + '_sweep')}")
        return 0
    results: dict[int, dict[str, Any]] = {}
    for item in plan:
        seed = item["seed"]
        if item["upcycle"] is not None:
            _run(item["upcycle"], "upcycle", seed)
            out = upcycled_path(run_root, name, seed)
            if not out.exists():
                raise SystemExit(f"seed {seed}: upcycle.py did not write {out}")
        _run(item["train"], "train", seed)
        final = run_root / item["run_name"] / "final_metrics.json"
        if not final.exists():
            raise SystemExit(f"seed {seed}: {final} not found after a successful run")
        with final.open("r", encoding="utf-8") as fh:
            results[seed] = json.load(fh)
    summary = aggregate(results)
    sweep_dir = run_root / f"{name}_sweep"
    write_json(
        sweep_dir / "sweep_summary.json",
        {
            "config": args.config,
            "overrides": args.overrides,
            "seeds": args.seeds,
            "vary_upcycle_seed": bool(args.vary_upcycle_seed),
            "upcycle_config": args.upcycle_config if args.vary_upcycle_seed else None,
            "metrics": summary,
        },
    )
    table = markdown_table(summary, sorted(results))
    write_text(sweep_dir / "sweep_summary.md", table)
    print(table)
    log.info("wrote %s", sweep_dir / "sweep_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
