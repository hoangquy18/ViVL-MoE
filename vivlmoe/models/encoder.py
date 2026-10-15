"""Shared multimodal encoder ``f_enc`` (paper Sec. 4.2, Fig. 3) with the MoE seam (Sec. 5).

Dense: every layer's FFN is a :class:`DenseFFN`. Sparse (``cfg.moe`` set): the FFN is
``vivlmoe.models.moe.moe_ffn.MoEFFN`` (milestone M6), whose ``router`` exposes ``expert_bias`` (persistent
buffer ``[N_r]``), ``dispatch_counts`` (int64 ``[N_r]``) and ``token_count`` (int64 scalar). The encoder
performs no collectives; the Task all-reduces ``dispatch_stats()`` and calls ``apply_router_bias_update``.
"""

from __future__ import annotations

import torch
from torch import nn

from vivlmoe.config.schema import EncoderCfg, MoECfg
from vivlmoe.models.block import TransformerBlock
from vivlmoe.models.ffn import DenseFFN
from vivlmoe.models.outputs import EncoderOutput, RouterOutput


class MultimodalEncoder(nn.Module):
    def __init__(self, cfg: EncoderCfg, moe: MoECfg | None = None) -> None:
        super().__init__()
        self.cfg, self.moe_cfg = cfg, moe
        self.layers = nn.ModuleList(
            TransformerBlock(cfg, self._build_ffn(i)) for i in range(cfg.num_layers)
        )

    def _build_ffn(self, layer_idx: int) -> nn.Module:
        if self.moe_cfg is None:
            return DenseFFN(self.cfg.dim, self.cfg.ffn_dim, self.cfg.ffn_inner_dropout)
        from vivlmoe.models.moe.moe_ffn import MoEFFN  # lazy: M6 seam

        return MoEFFN(
            dim=self.cfg.dim,
            ffn_dim=self.cfg.ffn_dim,
            dropout=self.cfg.ffn_inner_dropout,
            moe_cfg=self.moe_cfg,
            layer_idx=layer_idx,
        )

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        modality_ids: torch.Tensor,
    ) -> EncoderOutput:
        """``x [B,S,d]``, ``key_padding_mask [B,S]`` bool True=keep or None, ``modality_ids [B,S]`` long."""
        if x.ndim != 3 or x.shape[-1] != self.cfg.dim:
            raise ValueError(f"encoder expects [B,S,{self.cfg.dim}], got {tuple(x.shape)}")
        if modality_ids.shape != x.shape[:2]:
            raise ValueError(
                f"modality_ids must be [B,S]={tuple(x.shape[:2])}, got {tuple(modality_ids.shape)}"
            )
        router_outputs: list[RouterOutput] = []
        for layer in self.layers:
            x, ro = layer(x, key_padding_mask, modality_ids)
            if ro is not None:
                router_outputs.append(ro)
        return EncoderOutput(last_hidden_state=x, router_outputs=router_outputs)

    # ---------------------------------------------------------------- MoE API (no collectives)
    @property
    def is_moe(self) -> bool:
        return self.moe_cfg is not None

    def moe_layers(self) -> list[tuple[int, nn.Module]]:
        """``[(layer_idx, ffn)]`` for MoE layers (empty for the dense encoder)."""
        if not self.is_moe:
            return []
        return [
            (i, layer.ffn) for i, layer in enumerate(self.layers) if hasattr(layer.ffn, "router")
        ]

    def dispatch_stats(self) -> torch.Tensor:
        """``int64 [L, N_r + 1]``: per-layer routed-token counts per expert and the total token count."""
        rows = [
            torch.cat([ffn.router.dispatch_counts, ffn.router.token_count.reshape(1)])
            for _, ffn in self.moe_layers()
        ]
        if not rows:
            return torch.zeros(0, 0, dtype=torch.int64)
        return torch.stack(rows).to(torch.int64)

    @torch.no_grad()
    def apply_router_bias_update(self, f: torch.Tensor, eta: float) -> None:
        """Eq. 22: ``b_i += eta * sign(k / N_r - f_i)`` per layer; ``f [L, N_r]`` (all-reduced fractions)."""
        layers = self.moe_layers()
        if not layers:
            raise RuntimeError("apply_router_bias_update called on a dense encoder")
        if self.moe_cfg is None:
            raise RuntimeError("unreachable: moe_cfg missing")
        if f.shape != (len(layers), self.moe_cfg.num_routed):
            raise ValueError(
                f"f must be [L={len(layers)}, N_r={self.moe_cfg.num_routed}], got {tuple(f.shape)}"
            )
        target = self.moe_cfg.top_k / self.moe_cfg.num_routed
        for (_, ffn), f_l in zip(layers, f, strict=True):
            bias = ffn.router.expert_bias
            bias.add_(eta * torch.sign(target - f_l.to(bias.device, bias.dtype)))

    @torch.no_grad()
    def reset_dispatch_stats(self) -> None:
        for _, ffn in self.moe_layers():
            ffn.router.dispatch_counts.zero_()
            ffn.router.token_count.zero_()

    def router_bias_state(self) -> list[torch.Tensor]:
        """Detached copies of every layer's ``expert_bias`` (empty list for dense)."""
        return [ffn.router.expert_bias.detach().clone() for _, ffn in self.moe_layers()]
