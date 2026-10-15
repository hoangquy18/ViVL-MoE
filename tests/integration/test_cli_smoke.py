"""CLI smoke chain (plan M5 acceptance): stage1_tiny -> stage2_tiny -> stage3_tiny -> --resume auto.

Every stage is a ``scripts/train.py`` subprocess on CPU with synthetic data; the run-directory
layout and the vision-tower provenance handoff are asserted on disk.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import CONFIGS, REPO_ROOT
from vivlmoe.utils.ckpt_meta import read_meta

TRAIN = REPO_ROOT / "scripts" / "train.py"
PY = sys.executable
CHECKPOINT_FILES = ("last.pt", "best.pt", "last.pt.meta.json", "best.pt.meta.json")


def _run_train(config: Path, overrides: list[str], *args: str, cwd: Path) -> str:
    cmd = [PY, str(TRAIN), "--config", str(config), *args, *overrides]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise AssertionError(
            f"{' '.join(cmd)} failed ({proc.returncode})\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    return proc.stderr


def _assert_run_dir(run_dir: Path, expect_final: bool = True) -> dict:
    assert (run_dir / "config.resolved.yaml").exists(), run_dir
    assert (run_dir / "command.txt").exists()
    assert (run_dir / "train.log").exists()
    ck = run_dir / "checkpoints"
    for name in CHECKPOINT_FILES:
        assert (ck / name).exists(), ck / name
    assert list(ck.glob("step_*.pt")) and all(
        (p.parent / (p.name + ".meta.json")).exists() for p in ck.glob("step_*.pt")
    )
    assert (run_dir / "eval_history.jsonl").exists()
    metrics_dir = run_dir / "metrics"
    assert metrics_dir.is_dir() and list(metrics_dir.glob("*.json")), metrics_dir
    final = run_dir / "final_metrics.json"
    if expect_final:
        assert final.exists()
        return json.loads(final.read_text(encoding="utf-8"))
    return {}


@pytest.mark.slow
def test_pretraining_smoke_chain(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    syn = tmp_path / "synthetic"
    common = [f"run.dir={runs}", "data.num_workers=0", "train.amp.enabled=false"]

    # ---- stage 1 --------------------------------------------------------------------
    s1 = runs / "stage1_tiny"
    _run_train(
        CONFIGS / "_smoke" / "stage1_tiny.yaml",
        [*common, f"data.evals.ktvic_dev.root={syn}"],
        cwd=REPO_ROOT,
    )
    final1 = _assert_run_dir(s1)
    assert final1["step"] == 4 and "phomt_dev/mse" in final1["last_eval"]
    assert "ktvic_dev/rsum" in final1["last_eval"]
    assert read_meta(s1 / "checkpoints" / "best.pt")["tower_provenance"] == {
        "text_encoder": "trained"
    }

    # ---- stage 2: text tower from stage 1 --------------------------------------------
    s2 = runs / "stage2_tiny"
    roots = [
        f"data.train.root={syn}",
        f"data.evals.ktvic_dev.root={syn}",
        f"data.evals.uitviic_dev.root={syn}",
    ]
    _run_train(
        CONFIGS / "_smoke" / "stage2_tiny.yaml",
        [*common, *roots, f"init_from.path={s1 / 'checkpoints' / 'best.pt'}"],
        cwd=REPO_ROOT,
    )
    final2 = _assert_run_dir(s2)
    assert final2["step"] == 4 and "ktvic_dev/rsum" in final2["last_eval"]
    meta2 = read_meta(s2 / "checkpoints" / "best.pt")
    assert meta2["config"]["init_from"]["load"] == ["text_encoder"]

    # ---- stage 3: everything from stage 2, ViT trainable -----------------------------
    s3 = runs / "stage3_tiny"
    _run_train(
        CONFIGS / "_smoke" / "stage3_tiny.yaml",
        [*common, *roots, f"init_from.path={s2 / 'checkpoints' / 'best.pt'}", "train.max_steps=3"],
        cwd=REPO_ROOT,
    )
    final3 = _assert_run_dir(s3)
    assert final3["step"] == 3
    meta3 = read_meta(s3 / "checkpoints" / "last.pt")
    assert meta3["tower_provenance"]["vision_encoder"] == "trained"
    assert meta3["excluded_modules"] == []
    assert meta3["config"]["optim"]["freeze_preset"] == "stage3"
    assert [g["name"] for g in meta3["config"]["optim"]["groups"]] == ["encoders", "projections"]
    steps_before = sorted(p.name for p in (s3 / "checkpoints").glob("step_*.pt"))
    assert steps_before == ["step_0000002.pt", "step_0000003.pt"]

    # ---- resume: continues the same run to the full budget ---------------------------
    _run_train(
        CONFIGS / "_smoke" / "stage3_tiny.yaml",
        [*common, *roots, f"init_from.path={s2 / 'checkpoints' / 'best.pt'}"],
        "--resume",
        "auto",
        cwd=REPO_ROOT,
    )
    final3b = _assert_run_dir(s3)
    assert final3b["step"] == 4 and final3b["best_step"] is not None
    steps_after = sorted(p.name for p in (s3 / "checkpoints").glob("step_*.pt"))
    assert steps_after == ["step_0000003.pt", "step_0000004.pt"]  # keep_last_n 2
    history = [
        json.loads(line)
        for line in (s3 / "eval_history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [h["step"] for h in history] == [2, 4]  # step-3 checkpoint resumed without re-eval
    assert "resumed from" in (s3 / "train.log").read_text(encoding="utf-8")


@pytest.mark.hf_cache
def test_dry_run_stage1_small_hf_cache(tmp_path: Path) -> None:
    """``--dry-run`` of the real Stage-1 config with the locally cached small towers."""
    phomt = REPO_ROOT / "dataset" / "PhoMT"  # the config's root, resolved against cwd=REPO_ROOT
    if not (phomt / "detokenization" / "train" / "train.en").exists():
        pytest.skip(f"PhoMT not found under {phomt}")
    overrides = [
        f"run.dir={tmp_path / 'runs'}",
        "model.vision.name=openai/clip-vit-base-patch32",
        "model.text.name=FacebookAI/xlm-roberta-base",
        "data.train.options.max_train_samples=256",
        "data.num_workers=0",
        "data.batch_size=4",
        "train.grad_accum_steps=2",
        "train.amp.enabled=false",
    ]
    _run_train(CONFIGS / "pretrain" / "stage1_teacher.yaml", overrides, "--dry-run", cwd=REPO_ROOT)
    report = json.loads(
        (tmp_path / "runs" / "stage1_teacher" / "dry_run_report.json").read_text(encoding="utf-8")
    )
    assert report["dry_run"] and report["steps"] == 2 and report["effective_batch"] == 8
    assert all(abs(x) < float("inf") for x in report["losses"])
