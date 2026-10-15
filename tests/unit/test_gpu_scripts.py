"""scripts/gpu/*.sh: shell syntax, and the downstream step's dataset selection in DRY_RUN mode."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = sorted((REPO / "scripts" / "gpu").glob("*.sh")) + [REPO / "scripts" / "run_pretrain.sh"]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_shell_syntax(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_downstream_runs_only_datasets_with_data(tmp_path: Path) -> None:
    (tmp_path / "ktvic").mkdir()
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "VENV": str(tmp_path / "no-venv"),
        "LOG_DIR": str(tmp_path / "logs"),
        "KTVIC_ROOT": str(tmp_path / "ktvic"),
        "UITVIIC_ROOT": str(tmp_path / "missing"),
        "OPENVIVQA_ROOT": "",
        "SEEDS": "0 1",
    }
    out = subprocess.run(
        ["bash", str(REPO / "scripts" / "gpu" / "6_downstream.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    commands = [ln for ln in out.splitlines() if ln.startswith("+ ")]
    assert len(commands) == 1
    cmd = commands[0]
    assert "configs/v2/downstream/ktvic_caption.yaml --seeds 0 1 --nproc 1" in cmd
    assert "model.from_checkpoint=runs/v2_moe_sparse/checkpoints/best.pt" in cmd
    for key in ("data.train.root", "data.evals.dev.root", "data.evals.test.root"):
        assert f"{key}={tmp_path / 'ktvic'}" in cmd
    assert "bỏ qua uitviic_caption" in out and "bỏ qua openvivqa" in out


@pytest.mark.parametrize(
    ("ngpu", "mem", "expected"),
    [
        (1, 44, "data.batch_size=256 train.grad_accum_steps=2"),  # 1x A40: the config itself
        (2, 24, "data.batch_size=128 train.grad_accum_steps=2"),  # 2x 3090: 256 pairs, 512 per step
        (1, 24, "data.batch_size=128 train.grad_accum_steps=4"),  # 1x 3090: 128 pairs, 512 per step
        (4, 44, "data.batch_size=64 train.grad_accum_steps=2"),  # 4x A40
    ],
)
def test_pretrain_batch_follows_the_hardware(
    ngpu: int, mem: int, expected: str, tmp_path: Path
) -> None:
    script = f'source "{REPO}/scripts/gpu/env.sh" && pretrain_overrides'
    env = {**os.environ, "NGPU": str(ngpu), "GPU_MEM_GB": str(mem), "LOG_DIR": str(tmp_path)}
    out = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == expected
