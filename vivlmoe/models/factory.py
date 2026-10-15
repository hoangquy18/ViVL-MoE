"""Model factory: one entry point for tasks, ``evaluate.py`` and ``upcycle.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from torch import nn

from vivlmoe.config.loader import from_dict, load_yaml_with_includes
from vivlmoe.config.schema import ModelCfg
from vivlmoe.models.backbone import ViVLBackbone
from vivlmoe.models.heads.classifier import MultiImageClassifier
from vivlmoe.models.heads.seq2seq import Seq2SeqVLModel
from vivlmoe.models.stage1 import Stage1Student
from vivlmoe.models.towers.clip_teacher import CLIPImageTeacher, CLIPTextTeacher, teacher_output_dim

ModelKind = Literal["backbone", "stage1", "seq2seq", "classifier"]
MODEL_KINDS: tuple[str, ...] = ("backbone", "stage1", "seq2seq", "classifier")


def build_task_model(kind: str, cfg: ModelCfg) -> nn.Module:
    """``backbone`` (contrastive stages), ``stage1`` student, ``seq2seq`` (captioning/VQA), ``classifier`` (ViMACSA)."""
    if kind == "backbone":
        return ViVLBackbone(cfg)
    if kind == "stage1":
        return Stage1Student(cfg, teacher_dim=teacher_output_dim(cfg))
    if kind == "seq2seq":
        return Seq2SeqVLModel(ViVLBackbone(cfg), cfg)
    if kind == "classifier":
        return MultiImageClassifier(ViVLBackbone(cfg), cfg)
    raise ValueError(f"unknown model kind {kind!r}; expected one of {MODEL_KINDS}")


def build_teacher(cfg: ModelCfg) -> CLIPTextTeacher:
    return CLIPTextTeacher(cfg)


def build_image_teacher(cfg: ModelCfg) -> CLIPImageTeacher:
    return CLIPImageTeacher(cfg)


def load_model_cfg(path: str | Path) -> ModelCfg:
    """Read the ``model:`` section of a (possibly ``_include``-ing) YAML file, e.g. ``configs/model/tiny.yaml``."""
    raw = load_yaml_with_includes(path)
    if "model" not in raw:
        raise ValueError(f"{path}: no 'model' section")
    return from_dict(ModelCfg, raw["model"], "model")
