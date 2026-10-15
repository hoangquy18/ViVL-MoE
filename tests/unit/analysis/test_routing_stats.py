"""Routing analysis primitives (paper Sec. 6.5): change rate, heatmap / sym-KL, margins, load."""

from __future__ import annotations

import pytest
import torch

from vivlmoe.analysis.routing_stats import (
    ExpertLoadCollector,
    ModalityRoutingCollector,
    RouterMarginCollector,
    RoutingSnapshot,
    expert_change_rate,
)
from vivlmoe.models.moe.router import Router
from vivlmoe.models.outputs import TEXT, VISUAL, RouterOutput


def _snapshot(step: int, topk: list[list[list[int]]], modality: list[int]) -> RoutingSnapshot:
    return RoutingSnapshot(step, torch.tensor(topk), torch.tensor(modality))


def test_snapshot_sorts_and_roundtrips() -> None:
    snap = _snapshot(5, [[[3, 1], [0, 2]]], [VISUAL, TEXT])
    assert snap.topk.tolist() == [[[1, 3], [0, 2]]] and snap.topk.dtype == torch.int16  # compact
    assert snap.modality.dtype == torch.int8
    assert snap.num_layers == 1 and snap.num_tokens == 2
    again = RoutingSnapshot.from_state_dict(snap.state_dict())
    assert (
        again.step == 5
        and torch.equal(again.topk, snap.topk)
        and torch.equal(again.modality, snap.modality)
    )
    with pytest.raises(ValueError, match="L, T, k"):
        RoutingSnapshot(0, torch.zeros(2, 2), torch.zeros(2))
    with pytest.raises(ValueError, match="int16"):
        RoutingSnapshot(0, torch.full((1, 2, 2), 70000), torch.zeros(2))
    with pytest.raises(ValueError, match="modality"):
        RoutingSnapshot(0, torch.zeros(1, 2, 2), torch.zeros(3))


def test_change_rate_identical_permuted_and_all_different() -> None:
    prev = _snapshot(
        0, [[[0, 1], [2, 3], [1, 2]], [[0, 3], [1, 2], [0, 1]]], [VISUAL, VISUAL, TEXT]
    )
    same = _snapshot(
        1, [[[1, 0], [3, 2], [2, 1]], [[3, 0], [2, 1], [1, 0]]], [VISUAL, VISUAL, TEXT]
    )
    res = expert_change_rate(prev, same)  # permuted top-k = same set -> nothing changed
    assert res["mean"] == 0.0 and res["per_layer"] == [0.0, 0.0]
    assert res["per_modality"] == {"visual": 0.0, "text": 0.0} and res["steps"] == (0, 1)
    diff = _snapshot(
        2, [[[2, 3], [0, 1], [0, 3]], [[1, 2], [0, 3], [2, 3]]], [VISUAL, VISUAL, TEXT]
    )
    res = expert_change_rate(prev, diff)
    assert res["mean"] == 1.0 and res["per_modality"] == {"visual": 1.0, "text": 1.0}
    partial = _snapshot(
        3, [[[0, 1], [0, 1], [1, 2]], [[0, 3], [1, 2], [2, 3]]], [VISUAL, VISUAL, TEXT]
    )
    res = expert_change_rate(prev, partial)  # layer 0: token 1 changed; layer 1: token 2 changed
    assert res["per_layer"] == pytest.approx([1 / 3, 1 / 3]) and res["mean"] == pytest.approx(1 / 3)
    assert res["per_modality"]["visual"] == pytest.approx(0.25) and res["per_modality"][
        "text"
    ] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="different token streams"):
        expert_change_rate(prev, _snapshot(4, [[[0, 1]]], [VISUAL]))
    with pytest.raises(ValueError, match="modality"):
        expert_change_rate(prev, _snapshot(4, prev.topk.tolist(), [TEXT, TEXT, TEXT]))


def _ro(logits: torch.Tensor, mids: torch.Tensor, k: int = 2, layer: int = 0) -> RouterOutput:
    idx = logits.topk(k, dim=-1).indices
    return RouterOutput(layer, logits, idx, torch.softmax(logits.gather(1, idx), dim=-1), mids)


def test_modality_heatmap_columns_sum_to_one_and_sym_kl_zero_when_identical() -> None:
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(6, 4, generator=g)
    same = ModalityRoutingCollector(num_layers=2, num_experts=4)
    for layer in range(2):  # identical token stream for both modalities
        same.add(_ro(logits, torch.full((6,), VISUAL), layer=layer))
        same.add(_ro(logits, torch.full((6,), TEXT), layer=layer))
    res = same.result()
    heat = torch.tensor(res["heatmap"])
    assert heat.shape == (4, 2) and torch.allclose(heat.sum(dim=0), torch.ones(2))
    assert torch.allclose(torch.tensor(res["heatmap_per_layer"]).sum(dim=1), torch.ones(2, 2))
    assert res["sym_kl"] == pytest.approx(0.0, abs=1e-9) and res[
        "sym_kl_layer_mean"
    ] == pytest.approx(0.0, abs=1e-9)
    assert res["modalities_present"] == {"visual": True, "text": True}
    skew = ModalityRoutingCollector(1, 4)
    skew.add(_ro(torch.tensor([[9.0, 8.0, 0.0, 0.0]] * 3), torch.full((3,), VISUAL)))
    skew.add(_ro(torch.tensor([[0.0, 0.0, 9.0, 8.0]] * 3), torch.full((3,), TEXT)))
    res = skew.result()
    assert res["sym_kl"] > 1.0 and torch.tensor(res["heatmap"])[:2, VISUAL].sum() > 0.99
    absent = ModalityRoutingCollector(1, 4)
    absent.add(_ro(logits, torch.full((6,), VISUAL)))
    res = absent.result()
    assert (
        res["sym_kl"] is None
        and res["sym_kl_per_layer"] == [None]
        and res["modalities_present"]["text"] is False
    )


def test_router_margin_ignores_bias_and_topk_margin_zero_for_k1() -> None:
    torch.manual_seed(0)
    router = Router(8, 4, 2, aux_free_bias=True, init_std=0.5).eval()
    x = torch.randn(10, 8)
    mids = torch.tensor([VISUAL] * 6 + [TEXT] * 4)
    plain = router(x, mids, layer_idx=0)
    router.expert_bias.copy_(torch.tensor([50.0, -50.0, 0.0, 25.0]))
    biased = router(x, mids, layer_idx=0)
    assert not torch.equal(plain.topk_idx.sort(dim=-1).values, biased.topk_idx.sort(dim=-1).values)
    a, b = RouterMarginCollector(1), RouterMarginCollector(1)
    a.add(plain)
    b.add(biased)
    ra, rb = a.result(), b.result()
    assert ra["margin_full"] == pytest.approx(rb["margin_full"])  # bias-free softmax(z) margin
    assert ra["margin_full_visual"] == pytest.approx(rb["margin_full_visual"])
    probs = torch.softmax(plain.logits, dim=-1).topk(2, dim=-1).values
    assert ra["margin_full"] == pytest.approx(float((probs[:, 0] - probs[:, 1]).mean()), rel=1e-6)
    assert ra["margin_topk"] != pytest.approx(rb["margin_topk"])
    assert 0.0 <= ra["margin_topk_text"] <= 1.0 and ra["tokens_per_layer"] == [10.0]
    k1 = Router(8, 4, 1, aux_free_bias=False).eval()
    c = RouterMarginCollector(1)
    c.add(k1(x, mids, layer_idx=0))
    assert c.result()["margin_topk"] == 0.0 and c.result()["margin_full"] > 0.0
    empty = RouterMarginCollector(1).result()
    assert empty["margin_full"] is None and empty["margin_full_visual"] is None


def test_expert_load_collector_fractions_and_bias_range() -> None:
    load = ExpertLoadCollector(num_layers=2, num_experts=3)
    ro = RouterOutput(
        1,
        torch.zeros(4, 3),
        torch.tensor([[0, 1], [0, 1], [0, 2], [1, 2]]),
        torch.full((4, 2), 0.5),
        torch.zeros(4, dtype=torch.long),
    )
    load.add(ro)
    res = load.result([torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.5, -0.25, 0.0])])
    assert res["expert_frac_per_layer"][1] == pytest.approx([0.75, 0.75, 0.5])
    assert res["expert_frac_per_layer"][0] == [0.0, 0.0, 0.0] and res["tokens_per_layer"] == [
        0.0,
        4.0,
    ]
    assert res["expert_frac_max"] == 0.75 and res["expert_frac_min"] == 0.0
    assert res["bias_range"] == pytest.approx(0.75)
    assert "bias_range" not in load.result()
