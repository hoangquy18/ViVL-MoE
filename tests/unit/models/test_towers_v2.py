"""v2 towers: SigLIP 2 / C-RADIO vision towers and the mmBERT (ModernBERT) text tower."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import ModernBertConfig, ModernBertModel, SiglipVisionConfig, SiglipVisionModel

from vivlmoe.config.schema import TextCfg, VisionCfg
from vivlmoe.models.factory import build_task_model, load_model_cfg
from vivlmoe.models.towers import (
    RadioVisionTower,
    SiglipVisionTower,
    build_vision_tower,
    find_tower_paths,
)
from vivlmoe.models.towers.radio_stub import RadioStub
from vivlmoe.models.towers.text import TextTower
from vivlmoe.training import checkpoint as ckpt

REPO_ROOT = Path(__file__).resolve().parents[3]
V2_TINY = REPO_ROOT / "configs" / "model" / "v2_tiny.yaml"
SIGLIP_TINY = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                   image_size=32, patch_size=16)  # fmt: skip
RADIO_TINY = {"embed_dim": 32, "patch_size": 16, "preferred_resolution": 32}
MODERNBERT_TINY = dict(vocab_size=1000, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=4, max_position_embeddings=130, pad_token_id=1,
                       bos_token_id=0, eos_token_id=2, cls_token_id=0, sep_token_id=2,
                       global_attn_every_n_layers=1)  # fmt: skip


def test_siglip_tower_has_no_cls_no_head_and_frozen_output_is_fp32() -> None:
    tower = build_vision_tower(VisionCfg(kind="siglip", pretrained=False, tiny_config=SIGLIP_TINY))
    assert isinstance(tower, SiglipVisionTower)
    assert getattr(tower.model.vision_model, "head", None) is None  # vision_use_head=False
    tower.set_trainable(False)
    out = tower(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 4, 32) and out.dtype == torch.float32 and not out.requires_grad


def test_siglip_pretrained_drops_the_map_head_and_interpolates(tmp_path: Path) -> None:
    SiglipVisionModel(SiglipVisionConfig(**SIGLIP_TINY)).save_pretrained(tmp_path)  # with head
    tower = build_vision_tower(VisionCfg(kind="siglip", name=str(tmp_path), image_size=48))
    assert tower.num_tokens == 9 and tower.provenance == f"hf:{tmp_path}"
    assert tower(torch.randn(1, 3, 48, 48)).shape == (1, 9, 32)


def test_siglip_rejects_post_layernorm() -> None:
    cfg = VisionCfg(
        kind="siglip", pretrained=False, tiny_config=SIGLIP_TINY, apply_post_layernorm=True
    )
    with pytest.raises(ValueError, match="post-LayerNorm"):
        build_vision_tower(cfg)


def test_radio_tower_is_frozen_only() -> None:
    tower = build_vision_tower(VisionCfg(kind="radio", pretrained=False, tiny_config=RADIO_TINY))
    assert isinstance(tower, RadioVisionTower) and not tower.trainable
    assert (tower.num_tokens, tower.hidden_size, tower.image_size) == (4, 32, 32)
    out = tower(torch.rand(2, 3, 32, 32))
    assert out.shape == (2, 4, 32) and out.dtype == torch.float32 and not out.requires_grad
    tower.set_trainable(False)  # what the freeze presets call
    with pytest.raises(ValueError, match="frozen-only"):
        tower.set_trainable(True)


def test_radio_tower_runs_reduced_precision_weights_without_autocast() -> None:
    # On CUDA the frozen tower holds bf16 weights and evaluation runs without autocast, while RADIO's input
    # conditioner casts the pixels to fp32: the first layer then met fp32 inputs and bf16 weights.
    torch.manual_seed(0)
    tower = build_vision_tower(VisionCfg(kind="radio", pretrained=False, tiny_config=RADIO_TINY))
    px = torch.rand(2, 3, 32, 32)
    ref = tower(px)
    tower.model.to(torch.bfloat16)  # what apply_dtype_policy does on CUDA
    out = tower(px)
    assert out.dtype == torch.float32 and out.shape == ref.shape
    assert torch.allclose(out, ref, atol=0.1)


def test_radio_pretrained_runs_pinned_remote_code(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict = {}

    def fake_from_pretrained(path: str, **kw: object) -> tuple[RadioStub, dict]:
        calls.update(path=path, **kw)
        stub = RadioStub(embed_dim=24, patch_size=16, preferred_resolution=64)
        return stub, {"missing_keys": calls.get("missing", [])}

    monkeypatch.setattr("transformers.AutoModel.from_pretrained", fake_from_pretrained)
    tower = build_vision_tower(
        VisionCfg(kind="radio", name="nvidia/C-RADIOv4-SO400M", revision="abc123")
    )
    assert calls["trust_remote_code"] is True and calls["revision"] == "abc123"
    assert tower.provenance == "hf:nvidia/C-RADIOv4-SO400M@abc123"
    # no image_size: the model's preferred resolution
    assert (tower.image_size, tower.num_tokens, tower.hidden_size) == (64, 16, 24)
    calls["missing"] = ["radio_model.model.norm.weight"]  # a weight the checkpoint does not provide
    with pytest.raises(RuntimeError, match="lacks 1 weights"):
        build_vision_tower(VisionCfg(kind="radio", name="nvidia/C-RADIOv4-SO400M"))


def test_modernbert_text_tower_tiny_and_pretrained(tmp_path: Path) -> None:
    tower = TextTower(TextCfg(arch="modernbert", pretrained=False, tiny_config=MODERNBERT_TINY))
    ids = torch.tensor([[0, 5, 6, 2, 1, 1]])
    assert tower(ids, (ids != 1).long()).shape == (1, 6, 32) and tower.pad_token_id == 1
    ModernBertModel(ModernBertConfig(**MODERNBERT_TINY)).to(torch.bfloat16).save_pretrained(
        tmp_path
    )
    loaded = TextTower(TextCfg(name=str(tmp_path)))  # built from its own config (no pooling kwarg)
    assert getattr(loaded.model.config, "reference_compile", False) is False  # removed in 5.x
    assert {p.dtype for p in loaded.parameters()} == {torch.float32}


def test_v2_tiny_backbone_contrastive_forward() -> None:
    model = build_task_model("backbone", load_model_cfg(V2_TINY))
    ids = torch.tensor([[0, 5, 6, 7, 2], [0, 8, 2, 1, 1]])
    batch = {
        "pixel_values": torch.rand(2, 3, 32, 32),
        "input_ids": ids,
        "attention_mask": (ids != 1).long(),
    }
    out = model(batch, "contrastive")
    assert out.image_embeds.shape == (2, 32) and out.text_embeds.shape == (2, 32)
    assert isinstance(model.vision_encoder, RadioVisionTower)


def test_tower_lookups_do_not_descend_into_a_tower() -> None:
    model = build_task_model("backbone", load_model_cfg(V2_TINY))
    model.vision_encoder.model.text_encoder = nn.Linear(2, 2)  # remote code may reuse the name
    expected = {"vision_encoder": "vision_encoder", "text_encoder": "text_encoder"}
    assert find_tower_paths(model) == expected
    assert ckpt.find_tower_paths(model) == expected


def test_frozen_tower_forwards_in_chunks_with_identical_features() -> None:
    torch.manual_seed(0)
    whole = build_vision_tower(VisionCfg(kind="radio", pretrained=False, tiny_config=RADIO_TINY))
    chunked = build_vision_tower(
        VisionCfg(kind="radio", pretrained=False, tiny_config=RADIO_TINY, frozen_batch_chunk=2)
    )
    chunked.load_state_dict(whole.state_dict())
    calls: list[int] = []
    inner = chunked._features
    chunked._features = lambda px: calls.append(px.shape[0]) or inner(px)  # type: ignore[method-assign]
    px = torch.rand(5, 3, 32, 32)
    assert torch.allclose(chunked(px), whole(px), atol=1e-6)
    assert calls == [2, 2, 1]
