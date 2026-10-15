"""scripts/train.py end to end (single process) with the synthetic task: dry-run, fit, resume."""

from __future__ import annotations

import importlib.util
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

import tests.unit.training.synthetic_task  # noqa: F401  (registers the 'synthetic' task)

REPO = Path(__file__).resolve().parents[3]

CONFIG = """
_include: [configs/_base/train_defaults.yaml, configs/model/tiny.yaml]
run: {{name: cli, dir: "{run_root}", seed: 3}}
task: {{name: synthetic}}
data: {{batch_size: 4, eval_batch_size: 8, num_workers: 0, evals: {{dev: {{kind: synthetic, split: dev}}}}}}
train: {{max_steps: 6, amp: {{enabled: false}}, log_every: 2}}
schedule: {{kind: cosine, warmup_ratio: 0.2}}
eval: {{every_steps: 3, every_epochs: null, selection_metric: dev/score}}
checkpoint: {{every_steps: 3, every_epochs: null}}
logging: {{tensorboard: false}}
"""


@pytest.fixture()
def train_main() -> Iterator:
    spec = importlib.util.spec_from_file_location(
        "vivlmoe_train_script", REPO / "scripts" / "train.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    root = logging.getLogger()
    saved = list(root.handlers), root.level
    try:
        yield mod.main
    finally:  # setup_logging replaces the root handlers; restore pytest's
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved[0]:
            root.addHandler(h)
        root.setLevel(saved[1])


def test_dry_run_then_fit_then_resume(tmp_path: Path, train_main) -> None:
    cfg_path = tmp_path / "cli.yaml"
    cfg_path.write_text(CONFIG.format(run_root=tmp_path / "runs"), encoding="utf-8")
    run_dir = tmp_path / "runs" / "cli"

    assert train_main(["--config", str(cfg_path), "--dry-run"]) == 0
    report = json.loads((run_dir / "dry_run_report.json").read_text())
    assert report["dry_run"] and report["steps"] == 2 and report["effective_batch"] == 4
    assert (run_dir / "config.resolved.yaml").exists() and (run_dir / "command.txt").exists()
    assert not (run_dir / "checkpoints").exists()

    assert train_main(["--config", str(cfg_path), "train.max_steps=4"]) == 0
    final = json.loads((run_dir / "final_metrics.json").read_text())
    assert final["step"] == 4 and "dev/score" in final["last_eval"]
    names = sorted(p.name for p in (run_dir / "checkpoints").glob("step_*.pt"))
    assert names == ["step_0000003.pt", "step_0000004.pt"]
    assert "train.max_steps=4" in (run_dir / "command.txt").read_text()

    assert train_main(["--config", str(cfg_path), "--resume", "auto"]) == 0
    final2 = json.loads((run_dir / "final_metrics.json").read_text())
    assert final2["step"] == 6 and final2["best_step"] is not None
    assert (run_dir / "train.log").exists()
