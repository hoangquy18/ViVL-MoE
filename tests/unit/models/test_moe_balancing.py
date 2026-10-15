"""Aux-free bias update (Eq. 21-22) through ``aux_free_bias_step`` in a single process."""

from __future__ import annotations

import pytest
import torch

from vivlmoe.config.schema import EncoderCfg, MoECfg
from vivlmoe.models.encoder import MultimodalEncoder
from vivlmoe.models.moe.balancing import aux_free_bias_step, expert_fractions
from vivlmoe.models.outputs import TEXT, VISUAL
from vivlmoe.training.distributed import DistContext

CTX = DistContext.single_process("cpu")
ENC = EncoderCfg(
    dim=16, num_layers=2, heads=2, ffn_dim=32, dropout=0.0, attn_dropout=0.0, ffn_inner_dropout=0.0
)
MOE = MoECfg(num_shared=1, num_routed=4, top_k=2, bias_update_rate=0.1)


def _encoder(seed: int = 0) -> MultimodalEncoder:
    torch.manual_seed(seed)
    return MultimodalEncoder(ENC, MOE)


def _inputs(
    b: int = 3, s: int = 6, seed: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, s, ENC.dim, generator=g)
    mask = torch.ones(b, s, dtype=torch.bool)
    mask[0, 4:] = False
    mids = torch.full((b, s), VISUAL)
    mids[1:] = TEXT
    return x, mask, mids


def test_bias_sign_rule_and_reset() -> None:
    enc = _encoder().train()
    x, mask, mids = _inputs()
    enc(x, mask, mids)
    stats = enc.dispatch_stats().clone()
    assert (stats[:, -1] == int(mask.sum())).all()
    f = expert_fractions(stats)
    logs = aux_free_bias_step(enc, CTX, eta=0.1)
    target = MOE.top_k / MOE.num_routed
    for (_, ffn), f_l in zip(enc.moe_layers(), f, strict=True):
        assert torch.allclose(ffn.router.expert_bias, 0.1 * torch.sign(target - f_l))
    assert int(enc.dispatch_stats().sum()) == 0
    assert set(logs) == {
        "moe/expert_frac_max",
        "moe/expert_frac_min",
        "moe/bias_range",
        "moe/tokens",
    }
    assert logs["moe/expert_frac_max"] == pytest.approx(float(f.max()))
    assert logs["moe/expert_frac_min"] == pytest.approx(float(f.min()))
    assert logs["moe/tokens"] == float(mask.sum())
    assert 0.0 <= logs["moe/bias_range"] <= 0.2 + 1e-6
    per_layer = aux_free_bias_step(enc, CTX, eta=0.0, per_layer_logs=True)
    assert "moe/layer0/expert_frac_max" in per_layer and "moe/layer1/expert_frac_min" in per_layer


def test_bias_accumulates_and_moves_load_towards_uniform() -> None:
    enc = _encoder().train()
    x, mask, mids = _inputs(b=8, s=8)
    ffn = enc.moe_layers()[0][1]
    with (
        torch.no_grad()
    ):  # a large bias on expert 0 of layer 0 makes it win every selection (Eq. 21)
        ffn.router.expert_bias[0] = 5.0
    enc(x, mask, mids)
    f0 = expert_fractions(enc.dispatch_stats())[0]
    assert float(f0[0]) == 1.0  # every token selects expert 0
    for _ in range(200):  # Eq. 22 keeps pushing under-used experts up until the load spreads
        aux_free_bias_step(enc, CTX, eta=0.2)
        enc(x, mask, mids)
    f1 = expert_fractions(enc.dispatch_stats())[0]
    assert float(f1[0]) < 1.0 and float(f1.min()) > 0.0  # bias redistributed the load
    assert float(ffn.router.expert_bias[0]) < 5.0


def test_bias_step_edge_cases() -> None:
    enc = _encoder()
    with pytest.raises(RuntimeError, match="dense"):
        aux_free_bias_step(MultimodalEncoder(ENC, None), CTX, 0.1)
    # no routed tokens: f == 0 is not a load signal -> biases untouched (no uniform +eta drift)
    logs = aux_free_bias_step(enc, CTX, eta=0.05)
    assert logs["moe/tokens"] == 0.0 and logs["moe/expert_frac_max"] == 0.0
    assert all(torch.equal(b, torch.zeros(4)) for b in enc.router_bias_state())
    assert torch.equal(
        expert_fractions(torch.tensor([[1, 2, 3, 4, 0]])), torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    )
    with pytest.raises(ValueError, match="f must be"):
        enc.apply_router_bias_update(torch.zeros(2, 3), 0.1)
