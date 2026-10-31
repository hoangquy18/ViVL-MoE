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


FP16 = " train.amp.dtype=fp16 train.amp.teacher_autocast=true train.max_consecutive_nonfinite=16"


@pytest.mark.parametrize(
    ("ngpu", "mem", "cc", "expected"),
    [
        (1, 44, "8.6", "data.batch_size=256 train.grad_accum_steps=2"),  # 1x A40: the config itself
        (
            2,
            24,
            "8.6",
            "data.batch_size=128 train.grad_accum_steps=2",
        ),  # 2x 3090: 256 pairs, 512/step
        (
            1,
            24,
            "8.9",
            "data.batch_size=128 train.grad_accum_steps=4",
        ),  # 1x 4090: 128 pairs, 512/step
        (4, 44, "", "data.batch_size=64 train.grad_accum_steps=2"),  # 4x A40, capability unknown
        (1, 47, "7.5", "data.batch_size=256 train.grad_accum_steps=2" + FP16),  # Quadro RTX 8000
    ],
)
def test_pretrain_batch_follows_the_hardware(
    ngpu: int, mem: int, cc: str, expected: str, tmp_path: Path
) -> None:
    script = f'source "{REPO}/scripts/gpu/env.sh" && pretrain_overrides'
    env = {
        **os.environ,
        "NGPU": str(ngpu),
        "GPU_MEM_GB": str(mem),
        "GPU_CC": cc,
        "LOG_DIR": str(tmp_path),
    }
    out = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == expected


def _dry_run(script: str, tmp_path: Path, **env: str) -> list[str]:
    full = {
        **os.environ,
        "DRY_RUN": "1",
        "VENV": str(tmp_path / "no-venv"),
        "LOG_DIR": str(tmp_path / "logs"),
        "RUNS": str(tmp_path / "runs"),
        "NGPU": "1",
        "GPU_MEM_GB": "48",
        **env,
    }
    out = subprocess.run(
        ["bash", str(REPO / "scripts" / "gpu" / script)],
        env=full,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [ln for ln in out.splitlines() if ln.startswith("+ ")]


def test_pretrain_checks_the_backup_login_before_training(tmp_path: Path) -> None:
    commands = _dry_run("5_pretrain.sh", tmp_path)
    assert len(commands) == 2
    assert "hf_checkpoints.py check --repo nhq188/vivlmoe-v2-checkpoints" in commands[0]
    assert "HF_HUB_OFFLINE=0" in commands[0] and "scripts/run_pretrain.sh" in commands[1]
    no_backup = _dry_run("5_pretrain.sh", tmp_path, AUTO_BACKUP="0")
    assert len(no_backup) == 1 and "scripts/run_pretrain.sh" in no_backup[0]


def test_autobackup_worker_watches_the_runs(tmp_path: Path) -> None:
    commands = _dry_run("5c_autobackup.sh", tmp_path, BACKUP_CHECK="0", BACKUP_INTERVAL="60")
    assert commands == [
        "+ exec nice -n 10 python scripts/hf_checkpoints.py watch --repo "
        f"nhq188/vivlmoe-v2-checkpoints --runs {tmp_path / 'runs'} --interval 60"
    ]
    assert "check --repo" in _dry_run("5c_autobackup.sh", tmp_path)[0]
