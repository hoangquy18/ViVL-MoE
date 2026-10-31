"""scripts/bench_vision.py: end to end on CPU with the tiny v2 (RADIO stub) tower."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def test_bench_vision_times_the_tower_on_cpu(capsys: pytest.CaptureFixture[str]) -> None:
    spec = importlib.util.spec_from_file_location(
        "bench_vision_under_test", REPO / "scripts" / "bench_vision.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    config = str(REPO / "configs" / "v2" / "_smoke" / "align_tiny.yaml")
    argv = ["--config", config, "--device", "cpu", "--batch", "2", "--iters", "1"]
    assert mod.main([*argv, "--profile-rows", "3"]) == 0
    out = capsys.readouterr().out
    assert "2 images per forward" in out and "images/s" in out and "Self CPU" in out
