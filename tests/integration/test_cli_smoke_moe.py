"""CLI smoke chain (plan M6 acceptance): dense tiny checkpoint -> upcycle -> moe_tiny -> analyze_routing.

The dense checkpoint is built programmatically (factory + ``build_checkpoint``, no Trainer) so the
chain does not depend on the Stage-3 smoke run; ``scripts/upcycle.py``, ``scripts/train.py`` and
``scripts/analyze_routing.py`` run as CPU subprocesses on synthetic data.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.conftest import CONFIGS, REPO_ROOT
from vivlmoe.config import load_config
from vivlmoe.config.loader import to_dict
from vivlmoe.config.schema import CheckpointCfg
from vivlmoe.models.factory import build_task_model
from vivlmoe.training.checkpoint import (
    CheckpointManager,
    ProvenanceTracker,
    build_checkpoint,
    load_checkpoint_file,
)
from vivlmoe.training.distributed import DistContext
from vivlmoe.training.state import TrainState
from vivlmoe.utils.ckpt_meta import read_meta

SCRIPTS = REPO_ROOT / "scripts"
PY = sys.executable
CTX = DistContext.single_process("cpu")


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    cmd = [PY, str(SCRIPTS / script), *args]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise AssertionError(
            f"{' '.join(cmd)} failed ({proc.returncode})\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return proc


def _dense_checkpoint(tmp_path: Path) -> Path:
    """A tiny dense 'Stage-3' checkpoint with a proper sidecar (config.model = tiny)."""
    cfg = load_config(
        CONFIGS / "_smoke" / "stage2_tiny.yaml", ["init_from=null", f"run.dir={tmp_path / 'runs'}"]
    )
    torch.manual_seed(0)
    model = build_task_model("backbone", cfg.model)
    prov = ProvenanceTracker.from_config(cfg.model, model).as_dict()
    ckpt = build_checkpoint(
        model=model,
        excluded=[],
        provenance=prov,
        config=to_dict(cfg),
        train_state=TrainState(step=4, epoch=1).state_dict(),
        task_state={},
        ctx=CTX,
        metrics={"ktvic_dev/rsum": 123.0},
    )
    path = tmp_path / "dense" / "best.pt"
    CheckpointManager(path.parent, CheckpointCfg(), CTX).save(path, ckpt)
    assert read_meta(path)["tower_provenance"] == {
        "vision_encoder": "trained",
        "text_encoder": "trained",
    }
    return path


@pytest.mark.slow
def test_upcycle_train_analyze_chain(tmp_path: Path) -> None:
    dense = _dense_checkpoint(tmp_path)
    syn = tmp_path / "synthetic"
    runs = tmp_path / "runs"

    # ---- upcycle (noise-0 equivalence check inside) ------------------------------------
    upcycled = tmp_path / "upcycle_tiny" / "upcycled.pt"
    proc = _run(
        "upcycle.py",
        "--config",
        str(CONFIGS / "_smoke" / "upcycle_tiny.yaml"),
        "--out",
        str(upcycled),
        f"model.from_checkpoint={dense}",
    )
    assert "equivalence check passed" in proc.stderr and "parameters: dense" in proc.stderr
    meta = read_meta(upcycled)
    assert meta["config"]["model"]["moe"]["num_routed"] == 4 and meta["upcycle_info"]["seed"] == 0
    assert meta["upcycle_info"]["param_count_moe"] > meta["upcycle_info"]["param_count_dense"]
    assert meta["tower_provenance"] == {"vision_encoder": "trained", "text_encoder": "trained"}
    ck = load_checkpoint_file(upcycled)
    assert ck["optimizer"] is None and ck["step"] == 0
    assert "encoder.layers.0.ffn.routed_experts.3.fc2.weight" in ck["model"]
    assert "encoder.layers.0.ffn.router.expert_bias" in ck["model"]
    assert not any(k.endswith("ffn.fc1.weight") for k in ck["model"])
    # nr1 needs top_k 1; the script must refuse a mismatching override
    bad = subprocess.run(
        [
            PY,
            str(SCRIPTS / "upcycle.py"),
            "--config",
            str(CONFIGS / "_smoke" / "upcycle_tiny.yaml"),
            "--out",
            str(tmp_path / "bad.pt"),
            f"model.from_checkpoint={dense}",
            "model_overrides.moe.num_routed=1",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert bad.returncode != 0 and "top_k" in bad.stderr and not (tmp_path / "bad.pt").exists()

    # ---- sparse training with the routing probe ----------------------------------------
    overrides = [
        f"model.from_checkpoint={upcycled}",
        f"init_from.path={upcycled}",
        f"run.dir={runs}",
        f"data.train.root={syn}",
        f"data.evals.dev.root={syn}",
        "data.num_workers=0",
        "train.amp.enabled=false",
    ]
    _run("train.py", "--config", str(CONFIGS / "_smoke" / "moe_tiny.yaml"), *overrides)
    run_dir = runs / "moe_tiny"
    final = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    assert final["step"] == 4
    last = final["last_eval"]
    assert "dev/rsum" in last and "dev/routing/sym_kl" in last and "dev/routing/margin_full" in last
    assert 0.0 <= last["dev/routing/change_rate"] <= 1.0  # step 4 vs the step-2 snapshot
    routing = run_dir / "routing"
    # eval.every_steps=4 but the probe runs on its own cadence (every 2): step 2 must exist
    assert (routing / "step_0000002.json").exists() and (
        routing / "snapshot_step_0000002.pt"
    ).exists()
    history = [
        json.loads(line)
        for line in (run_dir / "eval_history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_step = {h["step"]: h for h in history if not h.get("final")}
    assert "dev/routing/sym_kl" in by_step[2] and "dev/rsum" not in by_step[2]  # probe only
    assert "dev/routing/change_rate" in by_step[4] and "dev/rsum" in by_step[4]  # full eval
    step4 = json.loads((routing / "step_0000004.json").read_text(encoding="utf-8"))
    assert step4["previous_step"] == 2 and step4["change_rate"]["steps"] == [2, 4]
    assert len(step4["modality"]["heatmap"]) == 4 and step4["forward_mode"] == "contrastive"
    best = run_dir / "checkpoints" / "best.pt"
    assert best.exists() and read_meta(best)["config"]["model"]["moe"]["num_routed"] == 4
    assert "encoder.layers.1.ffn.router.expert_bias" in load_checkpoint_file(best)["model"]
    assert (run_dir / "eval_history.jsonl").exists()

    # ---- offline routing analysis of the trained checkpoint ----------------------------
    out_dir = tmp_path / "analysis"
    proc = _run(
        "analyze_routing.py",
        "--config",
        str(CONFIGS / "_smoke" / "moe_tiny.yaml"),
        "--checkpoint",
        str(best),
        "--split",
        "dev",
        "--out-dir",
        str(out_dir),
        "--prev-snapshot",
        str(routing / "snapshot_step_0000002.pt"),
        *overrides,
    )
    summary = json.loads(proc.stdout[proc.stdout.index("{") :])
    assert "routing/change_rate" in summary and "routing/sym_kl" in summary
    reports = sorted((out_dir / "routing").glob("step_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["change_rate"] is not None and report["split"] == "dev"
    assert (out_dir / "routing" / f"{reports[0].stem}_heatmap.png").exists()
