"""MoE feed-forward layer replacing the dense FFN of ``f_enc`` (paper Sec. 5.3, Eq. 18).

``h = 1/2 [ (1/N_s) sum_s E^sh_s(x) + sum_{i in S(x)} p~_i E^rt_i(x) ]`` (Eq. 18); with ``N_s = 0`` the
output is half the routed path. Every expert is a :class:`DenseFFN` (same class as the dense FFN,
plan decision 4) so sparse upcycling is an exact copy. Only the valid (non-padded) tokens are
routed; padded positions receive zeros. State-dict keys:
``ffn.{shared_experts.{s},routed_experts.{e}}.{fc1,fc2}.{weight,bias}``, ``ffn.router.{proj.weight,expert_bias}``.
"""

from __future__ import annotations

import torch
from torch import nn

from vivlmoe.config.schema import MoECfg
from vivlmoe.models.ffn import DenseFFN
from vivlmoe.models.moe.dispatch import dispatch
from vivlmoe.models.moe.router import Router
from vivlmoe.models.outputs import RouterOutput


class MoEFFN(nn.Module):
    def __init__(
        self, dim: int, ffn_dim: int, dropout: float, moe_cfg: MoECfg, layer_idx: int
    ) -> None:
        super().__init__()
        self.moe_cfg, self.layer_idx, self.dim = moe_cfg, layer_idx, dim
        self.shared_experts = nn.ModuleList(
            DenseFFN(dim, ffn_dim, dropout) for _ in range(moe_cfg.num_shared)
        )
        self.routed_experts = nn.ModuleList(
            DenseFFN(dim, ffn_dim, dropout) for _ in range(moe_cfg.num_routed)
        )
        self.router = Router(
            dim,
            moe_cfg.num_routed,
            moe_cfg.top_k,
            aux_free_bias=moe_cfg.aux_free_bias,
            init_std=moe_cfg.router_init_std,
        )

    @property
    def num_shared(self) -> int:
        return len(self.shared_experts)

    @property
    def num_routed(self) -> int:
        return len(self.routed_experts)

    def forward_tokens(
        self, flat: torch.Tensor, modality_ids: torch.Tensor
    ) -> tuple[torch.Tensor, RouterOutput]:
        """Eq. 18 on valid tokens ``flat [T, d]`` -> ``(h fp32 [T, d], RouterOutput)``."""
        ro = self.router(flat, modality_ids, self.layer_idx)
        routed = dispatch(flat, list(self.routed_experts), ro.topk_idx, ro.topk_probs)
        if self.num_shared == 0:
            return 0.5 * routed, ro
        shared = torch.stack([e(flat).to(routed.dtype) for e in self.shared_experts]).mean(dim=0)
        return 0.5 * (shared + routed), ro

    def forward_block(
        self, x: torch.Tensor, modality_ids: torch.Tensor, token_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, RouterOutput]:
        """``x [B,S,d]``, ``modality_ids [B,S]``, ``token_mask [B,S]`` bool True=keep (None = all valid)."""
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"MoEFFN expects [B,S,{self.dim}], got {tuple(x.shape)}")
        if token_mask is None:
            flat, mids = x.reshape(-1, self.dim), modality_ids.reshape(-1)
            h, ro = self.forward_tokens(flat, mids)
            return h.to(x.dtype).view_as(x), ro
        mask = token_mask.bool()
        if mask.shape != x.shape[:2]:
            raise ValueError(
                f"token_mask must be [B,S]={tuple(x.shape[:2])}, got {tuple(mask.shape)}"
            )
        h, ro = self.forward_tokens(x[mask], modality_ids[mask])
        out = torch.zeros_like(x)
        out[mask] = h.to(x.dtype)
        return out, ro


__all__ = ["MoEFFN"]
