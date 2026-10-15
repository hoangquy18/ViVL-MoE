"""Config rules for the v2 vision towers (fail at load time, not mid-run)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vivlmoe.config import ConfigError, load_config

SMOKE = Path(__file__).resolve().parents[3] / "configs" / "_smoke" / "stage2_tiny.yaml"


def test_v1_defaults_are_unchanged() -> None:
    cfg = load_config(SMOKE)
    assert cfg.model.vision.kind == "clip" and cfg.model.text.arch == "xlm-roberta"
    assert cfg.data.transform.source == "clip_processor"


def test_v2_tower_with_the_clip_processor_transform_is_rejected() -> None:
    with pytest.raises(ConfigError, match="source: tower"):
        load_config(SMOKE, ["model.vision.kind=siglip"])


def test_radio_tower_cannot_use_gradient_checkpointing() -> None:
    overrides = ["model.vision.kind=radio", "data.transform.source=tower"]
    load_config(SMOKE, overrides)
    with pytest.raises(ConfigError, match="frozen-only"):
        load_config(SMOKE, [*overrides, "model.vision.gradient_checkpointing=true"])


def test_post_layernorm_is_clip_only() -> None:
    overrides = ["model.vision.kind=siglip", "data.transform.source=tower"]
    with pytest.raises(ConfigError, match="CLIP towers"):
        load_config(SMOKE, [*overrides, "model.vision.apply_post_layernorm=true"])


DISTILL = SMOKE.parents[1] / "v2" / "_smoke" / "distill_tiny.yaml"


def test_siglip_teacher_rules() -> None:
    cfg = load_config(DISTILL)
    assert cfg.model.stage1.teacher_kind == "siglip" and cfg.data.tokenizer.en_max_len == 64
    with pytest.raises(ConfigError, match="en_max_len must be 64"):
        load_config(DISTILL, ["data.tokenizer.en_max_len=77"])
    with pytest.raises(ConfigError, match="teacher_target must be projected"):
        load_config(DISTILL, ["model.stage1.teacher_target=eos_hidden"])
    with pytest.raises(ConfigError, match="needs teacher_name"):
        load_config(DISTILL, ["model.stage1.teacher_tiny_config=null"])
    with pytest.raises(ConfigError, match="only applies to teacher_kind=siglip"):
        load_config(SMOKE, ["model.stage1.teacher_name=org/siglip"])
    # the teacher length only binds the distillation task (later stages keep the CLIP default)
    later = ["model.stage1.teacher_kind=siglip", "model.stage1.teacher_name=org/siglip"]
    assert load_config(SMOKE, later).data.tokenizer.en_max_len == 77
