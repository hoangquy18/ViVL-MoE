"""State-dict key mapping between the dense FFN and the MoE FFN of ``f_enc`` (plan Sec. 8.5).

Dense: ``encoder.layers.{i}.ffn.{fc1,fc2}.{weight,bias}``.
MoE:   ``encoder.layers.{i}.ffn.{shared_experts.{s},routed_experts.{e}}.{fc1,fc2}.{weight,bias}`` and
``encoder.layers.{i}.ffn.router.{proj.weight,expert_bias}``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vivlmoe.config.schema import MoECfg

DENSE_FFN_RE = re.compile(r"^encoder\.layers\.(\d+)\.ffn\.(fc1|fc2)\.(weight|bias)$")
FFN_TENSORS: tuple[str, ...] = ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")


@dataclass(frozen=True)
class DenseFFNKey:
    layer: int
    tensor: str  # 'fc1.weight' | 'fc1.bias' | 'fc2.weight' | 'fc2.bias'

    @property
    def is_weight(self) -> bool:
        return self.tensor.endswith(".weight")


def parse_dense_ffn_key(key: str) -> DenseFFNKey | None:
    m = DENSE_FFN_RE.match(key)
    if m is None:
        return None
    return DenseFFNKey(layer=int(m.group(1)), tensor=f"{m.group(2)}.{m.group(3)}")


def dense_ffn_key(layer: int, tensor: str) -> str:
    return f"encoder.layers.{layer}.ffn.{tensor}"


def shared_expert_key(layer: int, s: int, tensor: str) -> str:
    return f"encoder.layers.{layer}.ffn.shared_experts.{s}.{tensor}"


def routed_expert_key(layer: int, e: int, tensor: str) -> str:
    return f"encoder.layers.{layer}.ffn.routed_experts.{e}.{tensor}"


def router_proj_key(layer: int) -> str:
    return f"encoder.layers.{layer}.ffn.router.proj.weight"


def expert_bias_key(layer: int) -> str:
    return f"encoder.layers.{layer}.ffn.router.expert_bias"


def moe_ffn_keys(layer: int, moe_cfg: MoECfg) -> list[str]:
    """Every MoE FFN key of one layer in state-dict order."""
    keys = [shared_expert_key(layer, s, t) for s in range(moe_cfg.num_shared) for t in FFN_TENSORS]
    keys += [routed_expert_key(layer, e, t) for e in range(moe_cfg.num_routed) for t in FFN_TENSORS]
    keys += [expert_bias_key(layer), router_proj_key(layer)]
    return keys


def dense_ffn_layers(keys: list[str] | set[str]) -> list[int]:
    """Sorted layer indices that carry a complete dense FFN (all four tensors)."""
    seen: dict[int, set[str]] = {}
    for k in keys:
        parsed = parse_dense_ffn_key(k)
        if parsed is not None:
            seen.setdefault(parsed.layer, set()).add(parsed.tensor)
    incomplete = sorted(layer for layer, ts in seen.items() if ts != set(FFN_TENSORS))
    if incomplete:
        raise KeyError(
            f"layers {incomplete} carry an incomplete dense FFN (expected {FFN_TENSORS})"
        )
    return sorted(seen)


__all__ = [
    "DENSE_FFN_RE",
    "FFN_TENSORS",
    "DenseFFNKey",
    "dense_ffn_key",
    "dense_ffn_layers",
    "expert_bias_key",
    "moe_ffn_keys",
    "parse_dense_ffn_key",
    "routed_expert_key",
    "router_proj_key",
    "shared_expert_key",
]
