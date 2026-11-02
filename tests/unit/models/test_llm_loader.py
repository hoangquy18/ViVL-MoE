from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from vivlmoe.models.heads.llm_loader import MissingWeightsError, load_causal_lm

REPO_ROOT = Path(__file__).resolve().parents[3]
TINY = dict(
    vocab_size=1000, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
    pad_token_id=0, eos_token_id=1, bos_token_id=1, tie_word_embeddings=False,
)  # fmt: skip


def _load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        f"vivlmoe_script_{name}", REPO_ROOT / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


def test_pretrained_load_is_fp32_and_refuses_missing_weights(tmp_path: Path) -> None:
    Qwen3ForCausalLM(Qwen3Config(**TINY)).to(torch.bfloat16).save_pretrained(tmp_path)
    model = load_causal_lm(str(tmp_path))
    assert {p.dtype for p in model.parameters()} == {torch.float32}

    weights = tmp_path / "model.safetensors"
    state = load_file(weights)
    state.pop("model.norm.weight")  # would be randomly initialised by from_pretrained
    save_file(state, weights, metadata={"format": "pt"})
    with pytest.raises(MissingWeightsError, match="missing 1 weights"):
        load_causal_lm(str(tmp_path))


def test_unknown_llm_class_points_to_the_v2_environment() -> None:
    with pytest.raises(ImportError, match="requirements-v2"):
        load_causal_lm("Qwen/Qwen3.5-0.8B", llm_class="NoSuchModelForCausalLM")


@pytest.mark.parametrize("arch", ["qwen3", "qwen3_next"])
def test_check_llm_head_passes_on_tiny_models(
    arch: str, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _load_script("check_llm_head")
    assert script.main(["--tiny", arch, "--device", "cpu", "--amp-dtype", "fp16"]) == 0
    out = capsys.readouterr().out
    assert (
        "PASS gradients: every trainable parameter received a finite gradient" in out
        and "PASS batched generation: left-padded batched generation matches" in out
    )


def test_check_llm_head_flags_non_finite_gradients() -> None:
    script = _load_script("check_llm_head")
    model = load_causal_lm(None, tiny_config=script.TINY_CONFIGS["qwen3"], tiny_arch="qwen3")
    with torch.no_grad():
        model.model.norm.weight.fill_(float("nan"))  # nan activations, as an fp16 overflow gives
    ok, msg = script.check_gradients(model, torch.device("cpu"), "fp16")
    assert not ok and "non-finite gradient" in msg


@pytest.mark.parametrize(
    ("ppl", "fp32_ppl", "ok"),
    [
        (12.0, 11.9, True),
        (12.0, None, True),  # no autocast (CPU): fp32 only
        (13.0, 11.9, False),  # 9% drift from fp32
        (float("inf"), 11.9, False),
        (float("nan"), 11.9, False),
        (300.0, None, False),  # above --max-ppl: random lm_head
    ],
)
def test_perplexity_verdict(ppl: float, fp32_ppl: float | None, ok: bool) -> None:
    script = _load_script("check_llm_head")
    assert script.perplexity_verdict(ppl, fp32_ppl, "fp16", 200.0, 0.05)[0] is ok
