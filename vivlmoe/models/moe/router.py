"""Top-k router of one MoE layer (paper Sec. 5.3, Eq. 15 and Eq. 21; plan decisions 5 and 7).

* ``z = W_r x`` is computed in fp32 with autocast disabled.
* Selection ``S(x) = Top-k(z + b)`` uses the aux-free bias ``b`` (Eq. 21, a non-gradient buffer);
  the renormalised gates ``p~ = softmax(z[S(x)])`` (Eq. 15) come from ``z`` alone.
* ``dispatch_counts`` / ``token_count`` (int64, non-persistent) accumulate only in training mode; the
  bias is **never** updated here (Task -> :func:`vivlmoe.models.moe.balancing.aux_free_bias_step`).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vivlmoe.models.outputs import RouterOutput


class Router(nn.Module):
    def __init__(
        self,
        dim: int,
        num_routed: int,
        top_k: int,
        aux_free_bias: bool = True,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= num_routed:
            raise ValueError(f"top_k must be in [1, num_routed={num_routed}], got {top_k}")
        self.num_routed, self.top_k, self.aux_free_bias = num_routed, top_k, aux_free_bias
        self.proj = nn.Linear(dim, num_routed, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=init_std)
        self.register_buffer("expert_bias", torch.zeros(num_routed, dtype=torch.float32))
        self.register_buffer(
            "dispatch_counts", torch.zeros(num_routed, dtype=torch.int64), persistent=False
        )
        self.register_buffer("token_count", torch.zeros((), dtype=torch.int64), persistent=False)

    def forward(self, x: torch.Tensor, modality_ids: torch.Tensor, layer_idx: int) -> RouterOutput:
        """``x [T, d]`` (valid tokens only), ``modality_ids [T]`` -> :class:`RouterOutput` (fp32)."""
        if x.ndim != 2:
            raise ValueError(f"router expects [T, d], got {tuple(x.shape)}")
        if modality_ids.shape != x.shape[:1]:
            raise ValueError(
                f"modality_ids must be [T={x.shape[0]}], got {tuple(modality_ids.shape)}"
            )
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(x.float(), self.proj.weight.float())  # [T, N_r] fp32
            selection = logits + self.expert_bias if self.aux_free_bias else logits  # Eq. 21
            topk_idx = selection.topk(self.top_k, dim=-1).indices  # [T, k]
            topk_probs = torch.softmax(logits.gather(1, topk_idx), dim=-1)  # Eq. 15 (from z)
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(topk_idx, self.num_routed)  # [T, k, N_r]
                self.dispatch_counts += one_hot.sum(dim=(0, 1)).to(torch.int64)
                self.token_count += x.shape[0]
        return RouterOutput(
            layer_idx=layer_idx,
            logits=logits,
            topk_idx=topk_idx,
            topk_probs=topk_probs,
            modality_ids=modality_ids,
        )

    @torch.no_grad()
    def apply_bias_update(self, f: torch.Tensor, eta: float) -> None:
        """Eq. 22 for this layer: ``b_i += eta * sign(k / N_r - f_i)`` (``f [N_r]`` all-reduced)."""
        if f.shape != self.expert_bias.shape:
            raise ValueError(f"f must be [{self.num_routed}], got {tuple(f.shape)}")
        target = self.top_k / self.num_routed
        f = f.to(self.expert_bias.device, self.expert_bias.dtype)
        self.expert_bias.add_(eta * torch.sign(target - f))

    @torch.no_grad()
    def reset_counts(self) -> None:
        self.dispatch_counts.zero_()
        self.token_count.zero_()

    def extra_repr(self) -> str:
        return (
            f"num_routed={self.num_routed}, top_k={self.top_k}, aux_free_bias={self.aux_free_bias}"
        )


__all__ = ["Router"]
