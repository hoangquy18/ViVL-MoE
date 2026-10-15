"""Tiny-config fixtures shared by the model tests (no downloads: configs/model/tiny.yaml)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vivlmoe.config.schema import ModelCfg
from vivlmoe.models.factory import load_model_cfg

TINY_YAML = Path(__file__).resolve().parents[3] / "configs" / "model" / "tiny.yaml"
VOCAB = 1000


@pytest.fixture(scope="session")
def tiny_cfg() -> ModelCfg:
    return load_model_cfg(TINY_YAML)


def make_text_batch(b: int, length: int, pad_from: list[int] | None = None) -> dict:
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(3, VOCAB, (b, length), generator=g)
    mask = torch.ones(b, length, dtype=torch.long)
    if pad_from is not None:
        for i, start in enumerate(pad_from):
            mask[i, start:] = 0
            ids[i, start:] = 1  # XLM-R pad id
    return {"input_ids": ids, "attention_mask": mask}


def make_contrastive_batch(b: int = 3, length: int = 7, size: int = 32) -> dict:
    g = torch.Generator().manual_seed(2)
    batch = make_text_batch(b, length, pad_from=[length, 4, 5][:b])
    batch["pixel_values"] = torch.randn(b, 3, size, size, generator=g)
    batch["uids"] = list(range(b))
    batch["image_ids"] = [f"img{i}" for i in range(b)]
    return batch
