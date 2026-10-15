"""XLM-RoBERTa text tower ``f_t`` (paper Sec. 4.2, ``T = f_t(c)``): token-level hidden states, no pooler."""

from __future__ import annotations

import logging

import torch
from torch import nn
from transformers import AutoModel, XLMRobertaConfig, XLMRobertaModel

from vivlmoe.config.schema import TextCfg
from vivlmoe.utils.hf import resolve_model_path

log = logging.getLogger(__name__)


class TextTower(nn.Module):
    """``forward(input_ids [B,m], attention_mask [B,m]) -> [B, m, d_t]``."""

    def __init__(self, cfg: TextCfg) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.pretrained:
            path = resolve_model_path(cfg.name)
            self.model = AutoModel.from_pretrained(path, add_pooling_layer=False)
            self.provenance = f"hf:{cfg.name}"
        else:
            if cfg.tiny_config is None:
                raise ValueError("model.text.tiny_config is required when pretrained=false")
            self.model = XLMRobertaModel(
                XLMRobertaConfig(**cfg.tiny_config), add_pooling_layer=False
            )
            self.provenance = "random"
        self.hidden_size: int = int(self.model.config.hidden_size)
        if cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        log.info(
            "TextTower %s: hidden=%d, grad_ckpt=%s",
            cfg.name if cfg.pretrained else "random-tiny",
            self.hidden_size,
            cfg.gradient_checkpointing,
        )

    @property
    def pad_token_id(self) -> int:
        pad = self.model.config.pad_token_id
        if pad is None:
            raise ValueError("text tower config has no pad_token_id")
        return int(pad)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"input_ids/attention_mask must be [B,m] with equal shapes, got "
                f"{tuple(input_ids.shape)} / {tuple(attention_mask.shape)}"
            )
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state
