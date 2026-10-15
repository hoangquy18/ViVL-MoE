"""Perceiver resampler ``R_v`` (paper Sec. 4.2 / 5.2): 64 learned queries cross-attend to projected patch tokens.

Layout per plan 8.3: ``latents [N, d]``; ``depth`` x { ``PerceiverAttention`` (pre-norm, ``to_q d->inner``,
``to_kv d->2*inner``, ``to_out inner->d``, SDPA over ``heads x dim_head``), ``ResamplerFFN`` (pre-norm,
``d -> ff_mult*d`` GELU ``-> d``) } with residual connections, then a final LayerNorm. Keys/values are the
projected patch tokens only (Flamingo-style ``concat_latents_to_kv`` is opt-in).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vivlmoe.config.schema import ResamplerCfg


class PerceiverAttention(nn.Module):
    def __init__(
        self, dim: int, heads: int, dim_head: int, concat_latents_to_kv: bool = False
    ) -> None:
        super().__init__()
        inner = heads * dim_head
        self.heads, self.dim_head = heads, dim_head
        self.concat_latents_to_kv = concat_latents_to_kv
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, 2 * inner, bias=False)
        self.to_out = nn.Linear(inner, dim)

    def forward(self, media: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """``media [B,n,d]``, ``latents [B,N,d]`` -> ``[B,N,d]``."""
        media = self.norm_media(media)
        latents = self.norm_latents(latents)
        kv_input = torch.cat([media, latents], dim=1) if self.concat_latents_to_kv else media
        b, n_lat, _ = latents.shape
        q = self.to_q(latents).view(b, n_lat, self.heads, self.dim_head).transpose(1, 2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        k = k.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)  # [B,h,N,dh]
        out = out.transpose(1, 2).reshape(b, n_lat, self.heads * self.dim_head)
        return self.to_out(out)


class ResamplerFFN(nn.Module):
    def __init__(self, dim: int, ff_mult: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ff_mult * dim)
        self.fc2 = nn.Linear(ff_mult * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(self.norm(x))))


class PerceiverResampler(nn.Module):
    """``forward(tokens [B,n_raw,d]) -> [B,num_latents,d]``."""

    def __init__(self, dim: int, cfg: ResamplerCfg) -> None:
        super().__init__()
        self.dim, self.num_latents = dim, cfg.num_latents
        self.latents = nn.Parameter(torch.randn(cfg.num_latents, dim))
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "attn": PerceiverAttention(
                        dim, cfg.heads, cfg.dim_head, cfg.concat_latents_to_kv
                    ),
                    "ff": ResamplerFFN(dim, cfg.ff_mult),
                }
            )
            for _ in range(cfg.depth)
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError(f"resampler expects [B,n,{self.dim}], got {tuple(tokens.shape)}")
        x = self.latents.to(tokens.dtype).unsqueeze(0).expand(tokens.shape[0], -1, -1)
        for layer in self.layers:
            x = x + self.dropout(layer["attn"](tokens, x))
            x = x + self.dropout(layer["ff"](x))
        return self.norm(x)
