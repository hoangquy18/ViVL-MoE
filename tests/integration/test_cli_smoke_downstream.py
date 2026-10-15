"""Subprocess chain: train.py caption_tiny -> evaluate.py on its best.pt -> sweep_seeds.py --dry-run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "configs" / "_smoke"

pytestmark = pytest.mark.slow


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    proc = subprocess.run(
        [sys.executable, *args], cwd=str(cwd), env=env, capture_output=True, text=True, timeout=900
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"{' '.join(args)} failed ({proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


@pytest.mark.parametrize("name", ["caption_tiny", "vqa_tiny", "vimacsa_tiny"])
def test_downstream_smoke_chain(tmp_path: Path, name: str) -> None:
    syn = tmp_path / "syn"
    common = [
        f"run.dir={tmp_path / 'runs'}",
        f"data.train.root={syn}",
        f"data.evals.dev.root={syn}",
    ]
    _run(["scripts/train.py", "--config", str(SMOKE / f"{name}.yaml"), *common], REPO)
    run_dir = tmp_path / "runs" / name
    final = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    assert final["step"] == 4 and final["best_step"] is not None
    assert any(k.startswith("final/dev/") for k in final["final"])
    best = run_dir / "checkpoints" / "best.pt"
    assert best.exists() and (run_dir / "checkpoints" / "best.pt.meta.json").exists()
    assert list((run_dir / "predictions").glob("dev_step*.jsonl"))

    out_dir = tmp_path / "eval"
    proc = _run(
        [
            "scripts/evaluate.py",
            "--config",
            str(SMOKE / f"{name}.yaml"),
            "--checkpoint",
            str(best),
            "--splits",
            "dev",
            "--limit",
            "4",
            "--out-dir",
            str(out_dir),
            *common,
        ],
        REPO,
    )
    assert "dev" in proc.stdout
    summary = json.loads((out_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert summary["limit"] == 4 and summary["splits"]["dev"]["num_samples"] >= 1
    assert list((out_dir / "predictions").glob("dev_step*.jsonl"))

    if name == "caption_tiny":
        proc = _run(
            [
                "scripts/sweep_seeds.py",
                "--config",
                str(SMOKE / f"{name}.yaml"),
                "--seeds",
                "0",
                "1",
                "2",
                "--dry-run",
                *common,
            ],
            REPO,
        )
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("[seed ")]
        assert len(lines) == 3 and "run.name=caption_tiny_seed2" in lines[2]
        assert not (tmp_path / "runs" / "caption_tiny_seed0").exists()
