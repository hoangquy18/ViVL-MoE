"""``scripts/probe_vision.py`` end to end on synthetic dev splits with a tiny SigLIP saved to disk (the
real SigLIP 2 / C-RADIOv4 checkpoints are only loaded on the training box)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from transformers import SiglipConfig, SiglipModel

from vivlmoe.data.tokenization import tiny_wordlevel_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE = REPO_ROOT / "configs" / "_smoke" / "stage2_tiny.yaml"


def _load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        f"vivlmoe_script_{name}", REPO_ROOT / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


def _tiny_siglip(path: Path) -> None:
    text = dict(vocab_size=1000, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, max_position_embeddings=64)  # fmt: skip
    vision = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                  image_size=32, patch_size=16)  # fmt: skip
    SiglipModel(SiglipConfig(text_config=text, vision_config=vision)).save_pretrained(path)
    tiny_wordlevel_tokenizer("clip", vocab_size=1000, model_max_length=64).save_pretrained(path)


def test_probe_vision_scores_every_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _load_script("probe_vision")
    model_dir = tmp_path / "tiny-siglip"
    _tiny_siglip(model_dir)
    monkeypatch.setattr(
        probe,
        "ENCODER_BUILDERS",
        {"tiny-siglip": lambda args, cache: probe.build_siglip(str(model_dir), 32, args, cache)},
    )
    syn, out = tmp_path / "syn", tmp_path / "probe.json"
    rc = probe.main(
        [
            "--config", str(SMOKE), "--encoders", "tiny-siglip", "--device", "cpu",
            "--workers", "0", "--batch-size", "4", "--out", str(out),
            f"data.evals.ktvic_dev.root={syn}", f"data.evals.uitviic_dev.root={syn}",
        ]
    )  # fmt: skip
    assert rc == 0
    results = json.loads(out.read_text(encoding="utf-8"))
    assert set(results) == {"ktvic_dev", "uitviic_dev"}
    for split in results.values():
        metrics = split["tiny-siglip"]
        assert {"i2t_r1", "t2i_r10", "rsum"} <= set(metrics)
        assert 0.0 <= metrics["rsum"] <= 600.0
