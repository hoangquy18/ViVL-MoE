from __future__ import annotations

import dataclasses

import pytest
import torch

from vivlmoe.config.schema import ModelCfg, VisionCfg
from vivlmoe.models.towers.clip_teacher import CLIPImageTeacher, CLIPTextTeacher, teacher_output_dim
from vivlmoe.models.towers.text import TextTower
from vivlmoe.models.towers.vision import VisionTower


def test_frozen_tower_no_graph_eval_and_fp32(tiny_cfg: ModelCfg) -> None:
    tower = VisionTower(tiny_cfg.vision)
    assert tower.trainable  # HF default: everything trainable until a freeze plan runs
    tower.set_trainable(False)
    tower.train()
    assert tower.training and not tower.model.training  # train() keeps a frozen tower in eval
    assert not tower.trainable
    out = tower(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 5, 32)
    assert not out.requires_grad and out.dtype == torch.float32
    assert next(tower.model.parameters()).dtype == torch.float32  # auto policy: fp32 on CPU


def test_trainable_tower_gets_grads_and_grad_ckpt(tiny_cfg: ModelCfg) -> None:
    cfg = dataclasses.replace(tiny_cfg.vision, gradient_checkpointing=True)
    tower = VisionTower(cfg)
    tower.set_trainable(True)
    tower.train()
    assert tower.model.training
    out = tower(torch.randn(2, 3, 32, 32))
    assert out.requires_grad
    out.sum().backward()
    post_ln = {id(p) for p in tower.model.vision_model.post_layernorm.parameters()}
    assert all(p.grad is not None for p in tower.model.parameters() if id(p) not in post_ln)
    assert all(not p.requires_grad for p in tower.model.vision_model.post_layernorm.parameters())


def test_external_requires_grad_false_is_respected(tiny_cfg: ModelCfg) -> None:
    tower = VisionTower(tiny_cfg.vision)
    for p in tower.model.parameters():
        p.requires_grad_(False)  # a generic freezer that bypasses set_trainable
    tower.train()
    assert not tower.model.training
    assert not tower(torch.randn(1, 3, 32, 32)).requires_grad


def test_explicit_bf16_policy_on_cpu(tiny_cfg: ModelCfg) -> None:
    cfg = dataclasses.replace(tiny_cfg.vision, frozen_dtype="bf16")
    tower = VisionTower(cfg)
    tower.set_trainable(False)
    assert next(tower.model.parameters()).dtype == torch.bfloat16
    out = tower(torch.randn(1, 3, 32, 32))
    assert out.dtype == torch.float32
    tower.set_trainable(True)
    assert next(tower.model.parameters()).dtype == torch.float32


def test_post_layernorm_option(tiny_cfg: ModelCfg) -> None:
    torch.manual_seed(0)
    raw = VisionTower(tiny_cfg.vision)
    torch.manual_seed(0)
    ln = VisionTower(dataclasses.replace(tiny_cfg.vision, apply_post_layernorm=True))
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        a, b = raw(x), ln(x)
    assert torch.allclose(b, raw.model.vision_model.post_layernorm(a), atol=1e-6)
    assert not torch.allclose(a, b)
    ln.set_trainable(True)
    assert all(p.requires_grad for p in ln.model.vision_model.post_layernorm.parameters())


def test_assert_dims_and_paper_dims_from_config() -> None:
    cfg = VisionCfg(
        pretrained=False,
        tiny_config={
            "hidden_size": 1280,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 16,
            "image_size": 378,
            "patch_size": 14,
        },
    )
    tower = VisionTower(cfg)
    assert tower.num_tokens == 730 and tower.hidden_size == 1280
    tower.assert_dims(730, 1280)
    with pytest.raises(ValueError, match="dims mismatch"):
        tower.assert_dims(50, 768)
    with pytest.raises(ValueError, match="pixel_values must be"):
        tower(torch.randn(1, 3, 32))


def test_text_tower(tiny_cfg: ModelCfg) -> None:
    tower = TextTower(tiny_cfg.text)
    assert tower.pad_token_id == 1
    assert not hasattr(tower.model, "pooler") or tower.model.pooler is None
    ids = torch.randint(3, 1000, (2, 6))
    h = tower(ids, torch.ones(2, 6, dtype=torch.long))
    assert h.shape == (2, 6, 32)
    with pytest.raises(ValueError):
        tower(ids, torch.ones(2, 5, dtype=torch.long))


def test_clip_teachers_frozen_eval_fp32(tiny_cfg: ModelCfg) -> None:
    teacher = CLIPTextTeacher(tiny_cfg)
    teacher.train()
    assert not teacher.model.training
    assert all(not p.requires_grad for p in teacher.parameters())
    ids = torch.randint(3, 1000, (2, 77))
    z = teacher(ids, torch.ones(2, 77, dtype=torch.long))
    assert z.shape == (2, 16) and z.dtype == torch.float32 and not z.requires_grad
    assert teacher_output_dim(tiny_cfg) == 16
    eos_cfg = dataclasses.replace(
        tiny_cfg, stage1=dataclasses.replace(tiny_cfg.stage1, teacher_target="eos_hidden")
    )
    assert teacher_output_dim(eos_cfg) == 32
    z2 = CLIPTextTeacher(eos_cfg)(ids, torch.ones(2, 77, dtype=torch.long))
    assert z2.shape == (2, 32)
    img = CLIPImageTeacher(tiny_cfg)(torch.randn(2, 3, 32, 32))
    assert img.shape == (2, 16) and not img.requires_grad


def test_forward_never_casts_policy_is_explicit(tiny_cfg: ModelCfg) -> None:
    """Review fix: the weight dtype (checkpoint fingerprint) must not change inside ``forward``."""
    cfg = dataclasses.replace(tiny_cfg.vision, frozen_dtype="bf16")
    tower = VisionTower(cfg)
    for p in tower.model.parameters():
        p.requires_grad_(False)  # Trainer path: optim.apply_freezing touches requires_grad only
    layout_before = [(k, tuple(v.shape), v.dtype) for k, v in tower.state_dict().items()]
    tower(torch.randn(1, 3, 32, 32))
    assert [(k, tuple(v.shape), v.dtype) for k, v in tower.state_dict().items()] == layout_before
    assert next(tower.model.parameters()).dtype == torch.float32
    assert tower.apply_dtype_policy() == torch.bfloat16
    assert next(tower.model.parameters()).dtype == torch.bfloat16
    out = tower(torch.randn(1, 3, 32, 32))
    assert out.dtype == torch.float32 and not out.requires_grad
    assert tower.apply_dtype_policy() == torch.bfloat16  # idempotent


def test_target_dtype_honours_amp_dtype(tiny_cfg: ModelCfg) -> None:
    tower = VisionTower(tiny_cfg.vision)  # auto
    tower.set_trainable(False)
    assert tower.target_dtype(torch.float16) == torch.float32  # CPU: never reduced precision
    assert tower.target_dtype(None) == torch.float32
    tower.set_trainable(True)
    assert tower.target_dtype(torch.bfloat16) == torch.float32  # trainable towers are fp32
    bf16 = VisionTower(dataclasses.replace(tiny_cfg.vision, frozen_dtype="bf16"))
    bf16.set_trainable(False)
    with pytest.raises(ValueError, match="fp16 AMP"):
        bf16.target_dtype(torch.float16)
    assert bf16.target_dtype(torch.bfloat16) == torch.bfloat16


def test_teacher_output_dim_rejects_vision_only_repo(
    tiny_cfg: ModelCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transformers import AutoConfig, CLIPVisionConfig

    from vivlmoe.models.towers import clip_teacher

    monkeypatch.setattr(clip_teacher, "resolve_model_path", lambda name: "/nonexistent")
    monkeypatch.setattr(
        AutoConfig, "from_pretrained", classmethod(lambda cls, path: CLIPVisionConfig())
    )
    cfg = dataclasses.replace(
        tiny_cfg, vision=dataclasses.replace(tiny_cfg.vision, pretrained=True)
    )
    with pytest.raises(ValueError, match="full CLIP repo"):
        teacher_output_dim(cfg)
