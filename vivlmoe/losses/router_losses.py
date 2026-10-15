"""Router regularisers (paper Sec. 5.4: Eq. 19, 20, 23, 24) and the single :class:`RouterLossComputer`.

Per MoE layer ``l`` the :class:`RouterOutput` s of one forward are grouped by ``layer_idx``
(contrastive mode: the visual pass and the text pass of the same layer; joint mode: one output with
mixed modality ids), so ``L_mi`` is identical for both modes given the same tokens. All math is fp32.

* z-loss (Eq. 20): ``L_z = mean_t logsumexp(z_t)^2``.
* modality-importance loss (Eq. 23-24): ``P_m,i ∝ sum_{t: M(t)=m} p~_i(x_t)`` (``p~ = 0`` for unselected
  experts), ``L_mi = 1/2 [KL(P_v || P_t) + KL(P_t || P_v)]``; ``0`` when a modality is absent.
* Switch load-balancing loss (Eq. 19, ablation arm, computed only under ``use_lb``):
  ``N_r * sum_i f_i P_i`` with ``f_i`` the (no-grad) fraction of tokens dispatched to expert ``i``
  (``sum_i f_i = k``) and ``P_i`` the mean softmax probability (``= k`` under uniform routing).
* ``aux = lambda_z L_z + lambda_mi L_mi + lambda_lb L_lb``, each term reduced over layers by
  ``layer_reduction`` (``mean`` default, plan decision 8) and gated by ``use_*``.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F

from vivlmoe.config.schema import RouterLossCfg
from vivlmoe.models.outputs import TEXT, VISUAL, RouterOutput

EPS = 1e-8


def z_loss(logits: torch.Tensor) -> torch.Tensor:
    """Eq. 20 on ``logits [T, N_r]``; zero for an empty token set."""
    if logits.shape[0] == 0:
        return logits.new_zeros(())
    return torch.logsumexp(logits.float(), dim=-1).pow(2).mean()


def dense_gates(ro: RouterOutput) -> torch.Tensor:
    """``[T, N_r]`` renormalised gates ``p~`` with zeros for unselected experts (differentiable)."""
    gates = torch.zeros_like(ro.logits, dtype=torch.float32)
    return gates.scatter(1, ro.topk_idx, ro.topk_probs.float())


def modality_distributions(
    gates: torch.Tensor, modality_ids: torch.Tensor, eps: float = EPS
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Eq. 23: ``(P_v, P_t)`` over experts (each ``[N_r]``, sums to 1); ``None`` for an absent modality."""

    def dist(mask: torch.Tensor) -> torch.Tensor | None:
        if not bool(mask.any()):
            return None
        mass = gates[mask].sum(dim=0) + eps
        return mass / mass.sum()

    return dist(modality_ids == VISUAL), dist(modality_ids == TEXT)


def sym_kl(p: torch.Tensor, q: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Eq. 24: ``1/2 [KL(p || q) + KL(q || p)]`` for two distributions over experts."""
    p, q = p.float().clamp(min=eps), q.float().clamp(min=eps)
    return 0.5 * ((p * (p / q).log()).sum() + (q * (q / p).log()).sum())


def switch_lb_loss(logits: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
    """Eq. 19 (Switch Transformer): ``N_r * sum_i f_i * P_i``.

    ``f_i`` is the (no-grad) fraction of *tokens* dispatched to expert ``i`` (``count_i / T``, so
    ``sum_i f_i = k``) and ``P_i`` the mean softmax probability, hence the loss equals ``k`` under
    perfectly uniform routing (1 for the original top-1 Switch router).
    """
    t, n = logits.shape
    if t == 0:
        return logits.new_zeros(())
    with torch.no_grad():
        f = F.one_hot(topk_idx, n).float().sum(dim=(0, 1)) / t  # tokens per expert / T, sum k
    p = torch.softmax(logits.float(), dim=-1).mean(dim=0)
    return n * (f * p).sum()


def group_by_layer(router_outputs: list[RouterOutput]) -> OrderedDict[int, RouterOutput]:
    """Concatenate the outputs of every layer (first-seen order) into one :class:`RouterOutput`."""
    groups: OrderedDict[int, list[RouterOutput]] = OrderedDict()
    for ro in router_outputs:
        groups.setdefault(ro.layer_idx, []).append(ro)
    merged: OrderedDict[int, RouterOutput] = OrderedDict()
    for layer, ros in groups.items():
        if len(ros) == 1:
            merged[layer] = ros[0]
            continue
        merged[layer] = RouterOutput(
            layer_idx=layer,
            logits=torch.cat([r.logits for r in ros]),
            topk_idx=torch.cat([r.topk_idx for r in ros]),
            topk_probs=torch.cat([r.topk_probs for r in ros]),
            modality_ids=torch.cat([r.modality_ids for r in ros]),
        )
    return merged


class RouterLossComputer:
    """``(router_outputs) -> (aux, logs)``; ``aux`` is a 0-tensor for dense models (empty list)."""

    def __init__(self, cfg: RouterLossCfg) -> None:
        if cfg.layer_reduction not in ("mean", "sum"):
            raise ValueError(f"unknown layer_reduction {cfg.layer_reduction!r}")
        self.cfg = cfg

    def _reduce(self, values: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(values)
        return stacked.mean() if self.cfg.layer_reduction == "mean" else stacked.sum()

    def per_layer(self, router_outputs: list[RouterOutput]) -> dict[int, dict[str, torch.Tensor]]:
        """Unweighted ``{layer: {'z', 'mi', 'lb'}}`` (differentiable fp32 scalars).

        ``'lb'`` is only computed when ``cfg.use_lb`` (a 0-tensor otherwise: the Switch loss is an
        ablation arm and would cost a one-hot + softmax per layer on the training critical path).
        """
        out: dict[int, dict[str, torch.Tensor]] = {}
        for layer, ro in group_by_layer(router_outputs).items():
            gates = dense_gates(ro)
            p_v, p_t = modality_distributions(gates, ro.modality_ids)
            mi = (
                sym_kl(p_v, p_t) if p_v is not None and p_t is not None else ro.logits.new_zeros(())
            )
            lb = (
                switch_lb_loss(ro.logits, ro.topk_idx)
                if self.cfg.use_lb
                else ro.logits.new_zeros((), dtype=torch.float32)
            )
            out[layer] = {"z": z_loss(ro.logits), "mi": mi, "lb": lb}
        return out

    def __call__(self, router_outputs: list[RouterOutput]) -> tuple[torch.Tensor, dict[str, float]]:
        if not router_outputs:
            zero = torch.zeros(())
            return zero, {"loss_z": 0.0, "loss_mi": 0.0, "loss_lb": 0.0}
        device = router_outputs[0].logits.device
        per_layer = self.per_layer(router_outputs)
        lz = self._reduce([d["z"] for d in per_layer.values()])
        lmi = self._reduce([d["mi"] for d in per_layer.values()])
        llb = self._reduce([d["lb"] for d in per_layer.values()])
        cfg = self.cfg
        aux = torch.zeros((), dtype=torch.float32, device=device)
        if cfg.use_z:
            aux = aux + cfg.lambda_z * lz
        if cfg.use_mi:
            aux = aux + cfg.lambda_mi * lmi
        if cfg.use_lb:
            aux = aux + cfg.lambda_lb * llb
        logs = {
            "loss_z": float(lz.detach()),
            "loss_mi": float(lmi.detach()),
            "loss_lb": float(llb.detach()),
        }
        return aux, logs


__all__ = [
    "RouterLossComputer",
    "dense_gates",
    "group_by_layer",
    "modality_distributions",
    "switch_lb_loss",
    "sym_kl",
    "z_loss",
]
