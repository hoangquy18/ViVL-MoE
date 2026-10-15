"""Multi-head self-attention of the shared encoder ``f_enc`` (paper Fig. 3): fused QKV + SDPA."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, attn_dropout: float = 0.0) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.heads, self.dim_head = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = attn_dropout

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``x [B,S,d]``; ``key_padding_mask [B,S]`` bool, True = keep -> ``[B,S,d]``."""
        b, s, _ = x.shape
        q, k, v = self.qkv(x).view(b, s, 3, self.heads, self.dim_head).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B,h,S,dh]
        attn_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool or key_padding_mask.shape != (b, s):
                raise ValueError(
                    f"key_padding_mask must be bool [B,S]={b, s}, got {key_padding_mask.dtype} "
                    f"{tuple(key_padding_mask.shape)}"
                )
            attn_mask = key_padding_mask[:, None, None, :]  # broadcast over heads and queries
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout if self.training else 0.0
        )
        return self.out(out.transpose(1, 2).reshape(b, s, self.heads * self.dim_head))
