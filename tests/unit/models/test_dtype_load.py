"""transformers 5 keeps a checkpoint's dtype when loading; trainable modules and teacher targets must
still come out fp32 (``vivlmoe.utils.hf.load_fp32``)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
from transformers import T5Config, T5ForConditionalGeneration, XLMRobertaConfig, XLMRobertaModel

from vivlmoe.config.schema import ModelCfg
from vivlmoe.models.factory import build_task_model
from vivlmoe.models.towers.text import TextTower
from vivlmoe.utils.hf import load_fp32


def _dtypes(module: torch.nn.Module) -> set[torch.dtype]:
    return {p.dtype for p in module.parameters()}


def _save_bf16(model: torch.nn.Module, path: Path) -> None:
    model.to(torch.bfloat16).save_pretrained(path)


def test_load_fp32_casts_a_half_precision_load(tiny_cfg: ModelCfg, tmp_path: Path) -> None:
    XLMRobertaModel(
        XLMRobertaConfig(**tiny_cfg.text.tiny_config), add_pooling_layer=False
    ).save_pretrained(tmp_path)
    # what transformers 5 does by default for a bf16 repo
    loaded = load_fp32(XLMRobertaModel, str(tmp_path), torch_dtype=torch.bfloat16)
    assert _dtypes(loaded) == {torch.float32}


def test_pretrained_text_tower_is_fp32_from_a_bf16_checkpoint(
    tiny_cfg: ModelCfg, tmp_path: Path
) -> None:
    _save_bf16(
        XLMRobertaModel(XLMRobertaConfig(**tiny_cfg.text.tiny_config), add_pooling_layer=False),
        tmp_path,
    )
    tower = TextTower(dataclasses.replace(tiny_cfg.text, name=str(tmp_path), pretrained=True))
    assert _dtypes(tower) == {torch.float32}


def test_pretrained_t5_decoder_is_fp32_from_a_bf16_checkpoint(
    tiny_cfg: ModelCfg, tmp_path: Path
) -> None:
    _save_bf16(T5ForConditionalGeneration(T5Config(**tiny_cfg.generation.tiny_config)), tmp_path)
    gen = dataclasses.replace(tiny_cfg.generation, name=str(tmp_path), pretrained=True)
    model = build_task_model("seq2seq", dataclasses.replace(tiny_cfg, generation=gen))
    assert _dtypes(model.decoder) == {torch.float32}
