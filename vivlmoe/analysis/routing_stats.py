"""Routing analysis primitives (paper Sec. 6.5): expert change rate, modality heatmap, margins, load.

All collectors are pure accumulators over :class:`RouterOutput` s; they hold no model state and do
no collectives. Their ``sums`` tensors are all-reduced by the evaluator before ``result()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vivlmoe.losses.router_losses import EPS, dense_gates, sym_kl
from vivlmoe.models.outputs import TEXT, VISUAL, RouterOutput

MODALITY_NAMES = {VISUAL: "visual", TEXT: "text"}


# --------------------------------------------------------------------------------------------
# snapshots and change rate
# --------------------------------------------------------------------------------------------
@dataclass
class RoutingSnapshot:
    """Sorted top-k expert sets of a fixed token stream: ``topk [L, T, k]``, ``modality [T]``.

    Stored compactly (``int16`` expert ids, ``int8`` modality ids: ``N_r <= 8``, modality in {0, 1})
    because the previous snapshot lives in every checkpoint's ``evaluator_state``; the change rate
    upcasts on the fly.
    """

    step: int
    topk: torch.Tensor
    modality: torch.Tensor

    def __post_init__(self) -> None:
        if self.topk.ndim != 3:
            raise ValueError(f"topk must be [L, T, k], got {tuple(self.topk.shape)}")
        if self.modality.shape != self.topk.shape[1:2]:
            raise ValueError(
                f"modality must be [T={self.topk.shape[1]}], got {tuple(self.modality.shape)}"
            )
        topk = self.topk.to(torch.int64)
        if topk.numel() and (int(topk.min()) < 0 or int(topk.max()) > torch.iinfo(torch.int16).max):
            raise ValueError("expert ids must fit in int16")
        self.topk = topk.sort(dim=-1).values.to(torch.int16).cpu()
        self.modality = self.modality.to(torch.int8).cpu()

    @property
    def num_layers(self) -> int:
        return int(self.topk.shape[0])

    @property
    def num_tokens(self) -> int:
        return int(self.topk.shape[1])

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "topk": self.topk, "modality": self.modality}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> RoutingSnapshot:
        return cls(int(state["step"]), state["topk"], state["modality"])


def expert_change_rate(prev: RoutingSnapshot, cur: RoutingSnapshot) -> dict[str, Any]:
    """Fraction of tokens whose top-k *set* changed between two snapshots of the same token stream.

    Returns ``{'mean', 'per_layer': [L], 'per_modality': {'visual', 'text'}, 'steps': (prev, cur)}``.
    """
    if prev.topk.shape != cur.topk.shape:
        raise ValueError(
            f"snapshots cover different token streams: {tuple(prev.topk.shape)} vs {tuple(cur.topk.shape)}"
        )
    if not torch.equal(prev.modality.to(torch.int64), cur.modality.to(torch.int64)):
        raise ValueError("snapshots have different modality streams")
    changed = (prev.topk.to(torch.int64) != cur.topk.to(torch.int64)).any(dim=-1).float()  # [L, T]
    per_layer = changed.mean(dim=1) if cur.num_tokens else torch.zeros(cur.num_layers)
    per_modality = {}
    for mid, name in MODALITY_NAMES.items():
        sel = cur.modality.to(torch.int64) == mid
        per_modality[name] = float(changed[:, sel].mean()) if bool(sel.any()) else None
    return {
        "mean": float(per_layer.mean()) if cur.num_layers else 0.0,
        "per_layer": [float(v) for v in per_layer],
        "per_modality": per_modality,
        "steps": (prev.step, cur.step),
    }


# --------------------------------------------------------------------------------------------
# collectors
# --------------------------------------------------------------------------------------------
class ModalityRoutingCollector:
    """Eq. 23 numerators per layer -> heatmap ``[N_r, 2]`` (columns sum to 1) + sym-KL (Eq. 24)."""

    def __init__(self, num_layers: int, num_experts: int) -> None:
        self.sums = torch.zeros(num_layers, num_experts, 2, dtype=torch.float64)

    def add(self, ro: RouterOutput) -> None:
        gates = dense_gates(ro.detach()).double()
        for mid in (VISUAL, TEXT):
            sel = ro.modality_ids == mid
            if bool(sel.any()):
                self.sums[ro.layer_idx, :, mid] += gates[sel].sum(dim=0).cpu()

    def result(self) -> dict[str, Any]:
        mass = self.sums + EPS
        per_layer = mass / mass.sum(dim=1, keepdim=True)  # [L, N_r, 2]
        heatmap = per_layer.mean(dim=0)  # averaged over layers, columns sum to 1
        present = [bool(self.sums[:, :, m].sum() > 0) for m in (VISUAL, TEXT)]
        both = all(present)
        sym_layers = [
            float(sym_kl(per_layer[layer, :, VISUAL], per_layer[layer, :, TEXT])) if both else None
            for layer in range(per_layer.shape[0])
        ]
        return {
            "heatmap": heatmap.tolist(),  # [N_r][visual, text]
            "heatmap_per_layer": per_layer.tolist(),
            "sym_kl": float(sym_kl(heatmap[:, VISUAL], heatmap[:, TEXT])) if both else None,
            "sym_kl_per_layer": sym_layers,
            "sym_kl_layer_mean": (
                float(sum(sym_layers) / len(sym_layers)) if both and sym_layers else None
            ),
            "modalities_present": {"visual": present[0], "text": present[1]},
        }


class RouterMarginCollector:
    """Per-layer router margins by modality.

    ``margin_full`` = top1 - top2 of ``softmax(z)`` over all experts (bias-free; the reported default),
    ``margin_topk`` = ``p~_1 - p~_2`` over the selected set (0 when ``k = 1``).
    """

    def __init__(self, num_layers: int) -> None:
        # [L, 2 modalities, 3] = (sum margin_full, sum margin_topk, count)
        self.sums = torch.zeros(num_layers, 2, 3, dtype=torch.float64)

    def add(self, ro: RouterOutput) -> None:
        ro = ro.detach()
        if ro.num_tokens == 0:
            return
        probs = torch.softmax(ro.logits.float(), dim=-1)
        if ro.num_experts >= 2:
            top2 = probs.topk(2, dim=-1).values
            margin_full = top2[:, 0] - top2[:, 1]
        else:
            margin_full = torch.ones(ro.num_tokens, device=probs.device)
        if ro.top_k >= 2:
            sorted_probs = ro.topk_probs.float().sort(dim=-1, descending=True).values
            margin_topk = sorted_probs[:, 0] - sorted_probs[:, 1]
        else:
            margin_topk = torch.zeros(ro.num_tokens, device=probs.device)
        for mid in (VISUAL, TEXT):
            sel = ro.modality_ids == mid
            if bool(sel.any()):
                row = self.sums[ro.layer_idx, mid]
                row[0] += float(margin_full[sel].double().sum())
                row[1] += float(margin_topk[sel].double().sum())
                row[2] += float(sel.sum())

    def result(self) -> dict[str, Any]:
        total = self.sums.sum(dim=1)  # [L, 3]
        cnt = total[:, 2].clamp(min=1.0)
        out: dict[str, Any] = {
            "margin_full_per_layer": (total[:, 0] / cnt).tolist(),
            "margin_topk_per_layer": (total[:, 1] / cnt).tolist(),
            "tokens_per_layer": total[:, 2].tolist(),
        }
        any_tokens = float(total[:, 2].sum()) > 0
        out["margin_full"] = float(total[:, 0].sum() / total[:, 2].sum()) if any_tokens else None
        out["margin_topk"] = float(total[:, 1].sum() / total[:, 2].sum()) if any_tokens else None
        for mid, name in MODALITY_NAMES.items():
            m = self.sums[:, mid].sum(dim=0)
            n = float(m[2])
            out[f"margin_full_{name}"] = float(m[0] / n) if n > 0 else None
            out[f"margin_topk_{name}"] = float(m[1] / n) if n > 0 else None
        return out


class ExpertLoadCollector:
    """Assignment counts per expert and layer (``[L, N_r]``) plus the current bias snapshot."""

    def __init__(self, num_layers: int, num_experts: int) -> None:
        self.sums = torch.zeros(num_layers, num_experts + 1, dtype=torch.float64)

    def add(self, ro: RouterOutput) -> None:
        if ro.num_tokens == 0:
            return
        counts = torch.nn.functional.one_hot(ro.topk_idx.reshape(-1), ro.num_experts).sum(dim=0)
        self.sums[ro.layer_idx, :-1] += counts.double().cpu()
        self.sums[ro.layer_idx, -1] += ro.num_tokens

    def result(self, biases: list[torch.Tensor] | None = None) -> dict[str, Any]:
        tokens = self.sums[:, -1:].clamp(min=1.0)
        frac = self.sums[:, :-1] / tokens  # fraction of tokens routed to each expert
        out: dict[str, Any] = {
            "expert_frac_per_layer": frac.tolist(),
            "expert_frac_max": float(frac.max()) if frac.numel() else None,
            "expert_frac_min": float(frac.min()) if frac.numel() else None,
            "tokens_per_layer": self.sums[:, -1].tolist(),
        }
        if biases is not None and biases:
            b = torch.stack([x.detach().float().cpu() for x in biases])
            out["expert_bias_per_layer"] = b.tolist()
            out["bias_range"] = float((b.max(dim=1).values - b.min(dim=1).values).max())
        return out


__all__ = [
    "MODALITY_NAMES",
    "ExpertLoadCollector",
    "ModalityRoutingCollector",
    "RouterMarginCollector",
    "RoutingSnapshot",
    "expert_change_rate",
]
