"""v2 CLI smoke chain on tiny random models (CPU, synthetic data), every step a subprocess:

distill (SigLIP teacher) -> align (sigmoid loss) -> upcycle -> sparse (routing probe) -> captioning with the
LLM head on the sparse checkpoint -> evaluate.py on its best checkpoint.
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

V2 = CONFIGS / "v2" / "_smoke"
PY = sys.executable

pytestmark = pytest.mark.slow


def _run(script: str, *args: str | Path) -> subprocess.CompletedProcess[str]:
    cmd = [PY, str(REPO_ROOT / "scripts" / script), *map(str, args)]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise AssertionError(
            f"{' '.join(cmd)} failed ({proc.returncode})\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    return proc


def _final(run_dir: Path) -> dict:
    return json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))


def test_v2_smoke_chain(tmp_path: Path) -> None:
    runs, syn = tmp_path / "runs", tmp_path / "synthetic"
    common = [f"run.dir={runs}", "data.num_workers=0", "train.amp.enabled=false"]

    # ---- distill: student text tower <- SigLIP text teacher -------------------------------
    _run(
        "train.py",
        "--config",
        V2 / "distill_tiny.yaml",
        *common,
        f"data.evals.ktvic_dev.root={syn}",
    )
    distill = runs / "distill_tiny" / "checkpoints" / "best.pt"
    last = _final(runs / "distill_tiny")["last_eval"]
    assert {"phomt_dev/mse", "ktvic_dev/rsum"} <= set(last)

    # ---- align: sigmoid loss, text tower from the distill run ----------------------------
    roots = [f"data.train.root={syn}", f"data.evals.ktvic_dev.root={syn}"]
    _run(
        "train.py", "--config", V2 / "align_tiny.yaml", *common, *roots, f"init_from.path={distill}"
    )
    align = runs / "align_tiny" / "checkpoints" / "best.pt"
    meta = read_meta(align)
    assert meta["config"]["model"]["contrastive_head"] == "sigmoid"
    assert meta["config"]["init_from"]["load"] == ["text_encoder"]
    assert "ktvic_dev/rsum" in _final(runs / "align_tiny")["last_eval"]

    # ---- upcycle (noise-0 equivalence check inside) -------------------------------------
    upcycled = tmp_path / "upcycle_tiny" / "upcycled.pt"
    proc = _run(
        "upcycle.py",
        "--config",
        V2 / "upcycle_tiny.yaml",
        "--out",
        upcycled,
        f"model.from_checkpoint={align}",
    )
    assert "equivalence check passed" in proc.stderr
    assert read_meta(upcycled)["config"]["model"]["moe"]["num_routed"] == 4

    # ---- sparse training with the routing probe -----------------------------------------
    sparse_overrides = [
        f"model.from_checkpoint={upcycled}",
        f"init_from.path={upcycled}",
        f"data.train.root={syn}",
        f"data.evals.dev.root={syn}",
    ]
    _run("train.py", "--config", V2 / "sparse_tiny.yaml", *common, *sparse_overrides)
    sparse = runs / "sparse_tiny" / "checkpoints" / "best.pt"
    last = _final(runs / "sparse_tiny")["last_eval"]
    assert "dev/rsum" in last and "dev/routing/sym_kl" in last

    # ---- captioning with the LLM head on the sparse checkpoint, then evaluate.py ----------
    caption = [
        f"model.from_checkpoint={sparse}",
        f"init_from.path={sparse}",
        f"data.train.root={syn}",
        f"data.evals.dev.root={syn}",
    ]
    _run("train.py", "--config", V2 / "caption_moe_tiny.yaml", *common, *caption)
    run_dir = runs / "caption_moe_tiny"
    final = _final(run_dir)
    assert final["step"] == 4 and any(k.startswith("final/dev/") for k in final["final"])
    cfg = read_meta(run_dir / "checkpoints" / "best.pt")["config"]
    assert cfg["model"]["generation"]["kind"] == "causal_lm" and cfg["model"]["moe"] is not None
    out_dir = tmp_path / "eval"
    _run(
        "evaluate.py",
        "--config",
        V2 / "caption_moe_tiny.yaml",
        "--checkpoint",
        run_dir / "checkpoints" / "best.pt",
        "--splits",
        "dev",
        "--limit",
        "4",
        "--out-dir",
        out_dir,
        *common,
        *caption,
    )
    summary = json.loads((out_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert summary["splits"]["dev"]["num_samples"] >= 1
