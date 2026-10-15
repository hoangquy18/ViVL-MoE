"""scripts/fetch_models.py v2 path: safetensors (sharded or not), else pytorch_model.bin (no network)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fetch_models.py"


def _load(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # the script sets it to 0; restored after the test
    spec = importlib.util.spec_from_file_location("fetch_models_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_hub(monkeypatch: pytest.MonkeyPatch, snap: Path, on_bin: list[str]) -> list[Any]:
    calls: list[Any] = []

    def snapshot_download(
        repo_id: str, revision: str | None = None, allow_patterns: Any = None
    ) -> str:
        calls.append(allow_patterns)
        if allow_patterns and "pytorch_model*.bin" in allow_patterns:
            for name in on_bin:
                (snap / name).write_bytes(b"w")
        return str(snap)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    return calls


def test_bin_only_repo_falls_back_to_pytorch_model_bin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load(monkeypatch)
    snap = tmp_path / "abc123"
    snap.mkdir()
    calls = _fake_hub(monkeypatch, snap, ["pytorch_model.bin"])
    assert mod.fetch_v2("org/bin-only", "text", None) == "abc123"
    assert calls == [mod.V2_PATTERNS, mod.BIN_PATTERNS]


def test_sharded_safetensors_need_every_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load(monkeypatch)
    snap = tmp_path / "def456"
    snap.mkdir()
    index = {
        "weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        }
    }
    (snap / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"w")
    _fake_hub(monkeypatch, snap, [])
    with pytest.raises(SystemExit, match="missing weight shards"):
        mod.fetch_v2("org/sharded", "siglip", None)
    (snap / "model-00002-of-00002.safetensors").write_bytes(b"w")
    assert mod.fetch_v2("org/sharded", "siglip", None) == "def456"


def test_repo_without_weights_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load(monkeypatch)
    snap = tmp_path / "0000"
    snap.mkdir()
    _fake_hub(monkeypatch, snap, [])
    with pytest.raises(SystemExit, match="no safetensors or pytorch_model.bin"):
        mod.fetch_v2("org/empty", "text", None)


def test_pretraining_group_is_part_of_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load(monkeypatch)
    pretrain = {m[0] for m in mod.MODELS["v2-pretrain"]}
    assert pretrain == {
        "nvidia/C-RADIOv4-SO400M",
        "jhu-clsp/mmBERT-base",
        "google/siglip2-giant-opt-patch16-384",
    }
    assert pretrain < {m[0] for m in mod.MODELS["v2"]}


def test_pin_revision_writes_the_fetched_commit_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load(monkeypatch)
    cfg = tmp_path / "v2.yaml"
    name = "nvidia/C-RADIOv4-SO400M"
    cfg.write_text(
        f"model:\n  vision: {{kind: radio, name: {name}, revision: null, image_size: 512}}\n",
        encoding="utf-8",
    )
    assert mod.pin_revision(cfg, name, "c0457f5").startswith("pinned")
    text = cfg.read_text(encoding="utf-8")
    assert f"name: {name}, revision: c0457f5, image_size: 512" in text
    assert mod.pin_revision(cfg, name, "c0457f5").endswith("already pinned at c0457f5")
    assert "pins" in mod.pin_revision(
        cfg, name, "ffffff0"
    )  # a different fetch: reported, not rewritten
    assert cfg.read_text(encoding="utf-8") == text
