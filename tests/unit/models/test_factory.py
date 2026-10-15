from __future__ import annotations

from pathlib import Path

import pytest

from vivlmoe.config.schema import ModelCfg
from vivlmoe.models.backbone import ViVLBackbone
from vivlmoe.models.factory import (
    MODEL_KINDS,
    build_image_teacher,
    build_task_model,
    build_teacher,
    load_model_cfg,
)
from vivlmoe.models.heads.classifier import MultiImageClassifier
from vivlmoe.models.heads.seq2seq import Seq2SeqVLModel
from vivlmoe.models.stage1 import Stage1Student

EXPECTED = {
    "backbone": ViVLBackbone,
    "stage1": Stage1Student,
    "seq2seq": Seq2SeqVLModel,
    "classifier": MultiImageClassifier,
}


def test_load_model_cfg_resolves_include(configs_dir: Path) -> None:
    cfg = load_model_cfg(configs_dir / "model" / "tiny.yaml")
    assert isinstance(cfg, ModelCfg)
    assert cfg.vision.pretrained is False and cfg.encoder.dim == 32
    assert cfg.generation.tiny_config["decoder_start_token_id"] == 0
    with pytest.raises(ValueError, match="no 'model' section"):
        load_model_cfg(configs_dir / "_base" / "train_defaults.yaml")


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_build_every_kind_from_tiny(tiny_cfg: ModelCfg, kind: str) -> None:
    model = build_task_model(kind, tiny_cfg)
    assert isinstance(model, EXPECTED[kind])


def test_unknown_kind(tiny_cfg: ModelCfg) -> None:
    with pytest.raises(ValueError, match="unknown model kind"):
        build_task_model("router", tiny_cfg)


def test_teachers(tiny_cfg: ModelCfg) -> None:
    assert build_teacher(tiny_cfg).output_dim == 16
    assert build_image_teacher(tiny_cfg).output_dim == 16
    assert build_task_model("stage1", tiny_cfg).teacher_dim == 16
