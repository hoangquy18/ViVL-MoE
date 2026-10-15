"""Sparse MoE layers of ``f_enc`` (paper Sec. 5): router, dispatch, MoE FFN and the bias update."""

from vivlmoe.models.moe.balancing import aux_free_bias_step, expert_fractions
from vivlmoe.models.moe.dispatch import dispatch, dispatch_naive
from vivlmoe.models.moe.moe_ffn import MoEFFN
from vivlmoe.models.moe.router import Router

__all__ = [
    "MoEFFN",
    "Router",
    "aux_free_bias_step",
    "dispatch",
    "dispatch_naive",
    "expert_fractions",
]
