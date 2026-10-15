"""Sparse upcycling (paper Sec. 5.3): dense ``ViVLBackbone`` -> MoE ``ViVLBackbone``."""

from vivlmoe.upcycling.key_mapping import DENSE_FFN_RE, moe_ffn_keys, parse_dense_ffn_key
from vivlmoe.upcycling.upcycle import (
    UpcycleError,
    noise_seed,
    upcycle_info,
    upcycle_model,
    upcycle_state_dict,
    verify_equivalence,
)

__all__ = [
    "DENSE_FFN_RE",
    "UpcycleError",
    "moe_ffn_keys",
    "noise_seed",
    "parse_dense_ffn_key",
    "upcycle_info",
    "upcycle_model",
    "upcycle_state_dict",
    "verify_equivalence",
]
