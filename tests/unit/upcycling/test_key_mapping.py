"""Dense <-> MoE FFN key mapping (plan 8.5): one strict regex plus the MoE key templates."""

from __future__ import annotations

import pytest

from vivlmoe.config.schema import MoECfg
from vivlmoe.upcycling.key_mapping import (
    DENSE_FFN_RE,
    FFN_TENSORS,
    DenseFFNKey,
    dense_ffn_key,
    dense_ffn_layers,
    expert_bias_key,
    moe_ffn_keys,
    parse_dense_ffn_key,
    routed_expert_key,
    router_proj_key,
    shared_expert_key,
)


def test_regex_matches_exactly_the_dense_ffn_tensors() -> None:
    assert parse_dense_ffn_key("encoder.layers.11.ffn.fc1.weight") == DenseFFNKey(11, "fc1.weight")
    assert parse_dense_ffn_key("encoder.layers.0.ffn.fc2.bias") == DenseFFNKey(0, "fc2.bias")
    assert parse_dense_ffn_key("encoder.layers.0.ffn.fc2.bias").is_weight is False
    for key in (
        "encoder.layers.0.ffn.shared_experts.0.fc1.weight",
        "encoder.layers.0.ffn.routed_experts.3.fc2.bias",
        "encoder.layers.0.ffn.router.proj.weight",
        "encoder.layers.0.attn.qkv.weight",
        "encoder.layers.0.ln1.weight",
        "backbone.encoder.layers.0.ffn.fc1.weight",
        "encoder.layers.x.ffn.fc1.weight",
        "encoder.layers.0.ffn.fc3.weight",
        "resampler.layers.0.ffn.fc1.weight",
    ):
        assert parse_dense_ffn_key(key) is None, key
        assert DENSE_FFN_RE.match(key) is None
    assert dense_ffn_key(2, "fc1.bias") == "encoder.layers.2.ffn.fc1.bias"


def test_moe_key_templates() -> None:
    cfg = MoECfg(num_shared=2, num_routed=3, top_k=2)
    keys = moe_ffn_keys(5, cfg)
    assert len(keys) == 4 * (2 + 3) + 2 and len(set(keys)) == len(keys)
    assert (
        shared_expert_key(5, 1, "fc2.weight") == "encoder.layers.5.ffn.shared_experts.1.fc2.weight"
    )
    assert routed_expert_key(5, 2, "fc1.bias") == "encoder.layers.5.ffn.routed_experts.2.fc1.bias"
    assert router_proj_key(5) == "encoder.layers.5.ffn.router.proj.weight"
    assert expert_bias_key(5) == "encoder.layers.5.ffn.router.expert_bias"
    assert all(k in keys for k in (shared_expert_key(5, 1, "fc2.weight"), router_proj_key(5)))
    assert moe_ffn_keys(0, MoECfg(num_shared=0, num_routed=1, top_k=1)) == [
        *[routed_expert_key(0, 0, t) for t in FFN_TENSORS],
        expert_bias_key(0),
        router_proj_key(0),
    ]


def test_dense_ffn_layers_is_strict() -> None:
    keys = [dense_ffn_key(layer, t) for layer in (3, 0, 1) for t in FFN_TENSORS] + [
        "encoder.layers.0.ln1.weight"
    ]
    assert dense_ffn_layers(keys) == [0, 1, 3]
    assert dense_ffn_layers(["text_proj.weight"]) == []
    with pytest.raises(KeyError, match="incomplete"):
        dense_ffn_layers(keys + [dense_ffn_key(7, "fc1.weight")])
