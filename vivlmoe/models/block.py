"""Post-LN Transformer block of ``f_enc`` (paper Fig. 3): MHSA -> Add&LN -> FFN -> Add&LN."""

from __future__ import annotations

import torch
from torch import nn

from vivlmoe.config.schema import EncoderCfg
from vivlmoe.models.attention import MultiHeadSelfAttention
from vivlmoe.models.outputs import RouterOutput


class TransformerBlock(nn.Module):
    """``ffn`` is a ``DenseFFN`` or (M6) an ``MoEFFN``; both expose ``forward_block``."""

    def __init__(self, cfg: EncoderCfg, ffn: nn.Module) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(cfg.dim, cfg.heads, cfg.attn_dropout)
        self.ln1 = nn.LayerNorm(cfg.dim, eps=cfg.ln_eps)
        self.ffn = ffn
        self.ln2 = nn.LayerNorm(cfg.dim, eps=cfg.ln_eps)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        modality_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, RouterOutput | None]:
        x = self.ln1(x + self.drop(self.attn(x, key_padding_mask)))
        h, router_out = self.ffn.forward_block(x, modality_ids, key_padding_mask)
        x = self.ln2(x + self.drop(h))
        return x, router_out
