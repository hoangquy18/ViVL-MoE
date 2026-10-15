"""Router losses (Eq. 19, 20, 23, 24) and the RouterLossComputer, incl. the optimizer path."""

from __future__ import annotations

import math

import pytest
import torch

from vivlmoe.config.schema import EncoderCfg, MoECfg, RouterLossCfg
from vivlmoe.losses.router_losses import (
    RouterLossComputer,
    dense_gates,
    group_by_layer,
    modality_distributions,
    switch_lb_loss,
    sym_kl,
    z_loss,
)
from vivlmoe.models.encoder import MultimodalEncoder
from vivlmoe.models.outputs import TEXT, VISUAL, RouterOutput


def _ro(
    logits: torch.Tensor, topk_idx: torch.Tensor, mids: torch.Tensor, layer: int = 0
) -> RouterOutput:
    probs = torch.softmax(logits.gather(1, topk_idx), dim=-1)
    return RouterOutput(layer, logits.float(), topk_idx, probs, mids)


def _random_ro(
    t: int, n: int = 4, k: int = 2, layer: int = 0, seed: int = 0, modality: int | None = None
) -> RouterOutput:
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(t, n, generator=g)
    idx = logits.topk(k, dim=-1).indices
    mids = (
        torch.randint(0, 2, (t,), generator=g) if modality is None else torch.full((t,), modality)
    )
    return _ro(logits, idx, mids, layer)


def test_z_loss_value() -> None:
    logits = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
    expected = (math.log(2.0) ** 2 + math.log(math.e + math.exp(-1.0)) ** 2) / 2
    assert float(z_loss(logits)) == pytest.approx(expected, rel=1e-6)
    assert float(z_loss(torch.zeros(0, 3))) == 0.0


def test_dense_gates_and_modality_distributions() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0], [1.0, 3.0, 0.0]])
    idx = torch.tensor([[0, 1], [2, 1], [1, 0]])
    mids = torch.tensor([VISUAL, VISUAL, TEXT])
    ro = _ro(logits, idx, mids)
    gates = dense_gates(ro)
    assert gates.shape == (3, 3) and torch.allclose(gates.sum(dim=1), torch.ones(3))
    assert float(gates[0, 2]) == 0.0 and float(gates[1, 0]) == 0.0  # unselected -> p~ = 0
    p_v, p_t = modality_distributions(gates, mids)
    assert p_v is not None and p_t is not None
    assert float(p_v.sum()) == pytest.approx(1.0) and float(p_t.sum()) == pytest.approx(1.0)
    expected_v = gates[:2].sum(dim=0) + 1e-8
    assert torch.allclose(p_v, expected_v / expected_v.sum())
    only_v, absent = modality_distributions(gates, torch.zeros(3, dtype=torch.long))
    assert only_v is not None and absent is None


def test_sym_kl_properties() -> None:
    p = torch.tensor([0.2, 0.3, 0.5])
    q = torch.tensor([0.5, 0.3, 0.2])
    assert float(sym_kl(p, p)) == pytest.approx(0.0, abs=1e-9)
    assert float(sym_kl(p, q)) == pytest.approx(float(sym_kl(q, p)))
    kl = lambda a, b: float((a * (a / b).log()).sum())  # noqa: E731
    assert float(sym_kl(p, q)) == pytest.approx(0.5 * (kl(p, q) + kl(q, p)), rel=1e-6)
    assert float(sym_kl(p, q)) > 0.0


def test_switch_lb_loss_uniform_equals_k() -> None:
    n = 4
    logits = torch.zeros(n, n)  # uniform softmax -> P_i = 1/N
    idx = torch.arange(n).unsqueeze(1)  # each expert assigned exactly once -> f_i = 1/N
    assert float(switch_lb_loss(logits, idx)) == pytest.approx(1.0)  # k = 1: the Switch value
    # top-2 uniform: f_i = count_i / T = 2/N (fraction of TOKENS, sums to k), so N * sum f_i P_i = k
    idx2 = torch.stack([torch.arange(n), (torch.arange(n) + 1) % n], dim=1)
    assert float(switch_lb_loss(logits, idx2)) == pytest.approx(2.0)
    skewed = torch.zeros(n, 1, dtype=torch.long)  # everything to expert 0
    logits2 = torch.tensor([[5.0, 0.0, 0.0, 0.0]] * n)
    assert float(switch_lb_loss(logits2, skewed)) > 1.0
    assert float(switch_lb_loss(torch.zeros(0, n), torch.zeros(0, 1, dtype=torch.long))) == 0.0


def test_mi_contrastive_equals_joint_and_grouping() -> None:
    visual = _random_ro(9, layer=0, seed=1, modality=VISUAL)
    text = _random_ro(5, layer=0, seed=2, modality=TEXT)
    joint = RouterOutput(
        0,
        torch.cat([visual.logits, text.logits]),
        torch.cat([visual.topk_idx, text.topk_idx]),
        torch.cat([visual.topk_probs, text.topk_probs]),
        torch.cat([visual.modality_ids, text.modality_ids]),
    )
    computer = RouterLossComputer(RouterLossCfg(use_lb=True))
    a = computer.per_layer([visual, text])[0]
    b = computer.per_layer([joint])[0]
    for key in ("z", "mi", "lb"):
        assert float(a[key]) == pytest.approx(float(b[key]), rel=1e-6)
    assert float(a["mi"]) > 0.0 and float(a["lb"]) > 0.0
    # the Switch loss is an ablation arm: not computed (0-tensor) unless use_lb
    off = RouterLossComputer(RouterLossCfg()).per_layer([joint])[0]
    assert float(off["lb"]) == 0.0 and off["lb"].dtype == torch.float32
    groups = group_by_layer([visual, _random_ro(3, layer=1), text])
    assert list(groups) == [0, 1] and groups[0].num_tokens == 14 and groups[1].num_tokens == 3


def test_mi_zero_when_a_modality_is_absent() -> None:
    computer = RouterLossComputer(RouterLossCfg())
    aux, logs = computer([_random_ro(6, modality=VISUAL)])
    assert logs["loss_mi"] == 0.0 and math.isfinite(float(aux)) and logs["loss_z"] > 0.0


def test_layer_reduction_flags_and_lambdas() -> None:
    ros = [_random_ro(8, layer=0, seed=3), _random_ro(6, layer=1, seed=4)]
    mean_c = RouterLossComputer(RouterLossCfg(layer_reduction="mean", use_lb=True))
    sum_c = RouterLossComputer(RouterLossCfg(layer_reduction="sum", use_lb=True))
    aux_mean, logs_mean = mean_c(ros)
    aux_sum, logs_sum = sum_c(ros)
    assert float(aux_sum) == pytest.approx(2 * float(aux_mean), rel=1e-5)
    for k in ("loss_z", "loss_mi", "loss_lb"):
        assert logs_sum[k] == pytest.approx(2 * logs_mean[k], rel=1e-5)
    cfg = RouterLossCfg(lambda_z=0.1, lambda_mi=0.2, lambda_lb=0.3, use_lb=True)
    aux, logs = RouterLossComputer(cfg)(ros)
    assert float(aux) == pytest.approx(
        0.1 * logs["loss_z"] + 0.2 * logs["loss_mi"] + 0.3 * logs["loss_lb"], rel=1e-5
    )
    aux_no_mi, logs2 = RouterLossComputer(RouterLossCfg(lambda_z=0.1, lambda_mi=0.2, use_mi=False))(
        ros
    )
    assert float(aux_no_mi) == pytest.approx(0.1 * logs2["loss_z"], rel=1e-5)
    assert logs2["loss_mi"] == logs["loss_mi"]  # logs stay unweighted and unaffected by the flags
    aux_none, _ = RouterLossComputer(RouterLossCfg(use_z=False, use_mi=False, use_lb=False))(ros)
    assert float(aux_none) == 0.0
    assert aux.dtype == torch.float32


def test_empty_router_outputs_give_zero() -> None:
    aux, logs = RouterLossComputer(RouterLossCfg())([])
    assert isinstance(aux, torch.Tensor) and float(aux) == 0.0
    assert logs == {"loss_z": 0.0, "loss_mi": 0.0, "loss_lb": 0.0}
    with pytest.raises(ValueError, match="layer_reduction"):
        RouterLossComputer(RouterLossCfg(layer_reduction="max"))  # type: ignore[arg-type]


def _encoder_and_inputs():
    torch.manual_seed(0)
    enc = MultimodalEncoder(
        EncoderCfg(
            dim=16,
            num_layers=2,
            heads=2,
            ffn_dim=32,
            dropout=0.0,
            attn_dropout=0.0,
            ffn_inner_dropout=0.0,
        ),
        MoECfg(num_shared=1, num_routed=4, top_k=2),
    )
    g = torch.Generator().manual_seed(5)
    x = torch.randn(4, 6, 16, generator=g)
    mids = torch.zeros(4, 6, dtype=torch.long)
    mids[2:] = TEXT
    return enc, x, mids


def _router_grads(lambda_mi: float) -> tuple[torch.Tensor, float, dict[str, float]]:
    enc, x, mids = _encoder_and_inputs()
    enc.train()
    out = enc(x, None, mids)
    task = out.last_hidden_state.pow(2).mean()
    aux, logs = RouterLossComputer(RouterLossCfg(lambda_z=1e-3, lambda_mi=lambda_mi))(
        out.router_outputs
    )
    total = task + aux
    total.backward()
    grads = torch.cat([ffn.router.proj.weight.grad.flatten() for _, ffn in enc.moe_layers()])
    assert float(total) == pytest.approx(
        float(task) + 1e-3 * logs["loss_z"] + lambda_mi * logs["loss_mi"], rel=1e-5
    )
    return grads, float(task), logs


def test_router_losses_reach_the_optimizer() -> None:
    g0, task0, _ = _router_grads(0.0)
    g1, task1, logs = _router_grads(5e-3)
    assert task0 == pytest.approx(task1)
    assert torch.count_nonzero(g0) > 0  # z-loss (and the gates p~) already reach router.proj
    assert not torch.allclose(g0, g1), "lambda_mi did not change the router gradients"
    assert logs["loss_mi"] > 0.0
