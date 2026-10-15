"""Token dispatch (plan decision 6): naive-loop parity, gradcheck, idle experts stay in the graph."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vivlmoe.models.ffn import DenseFFN
from vivlmoe.models.moe.dispatch import dispatch, dispatch_naive

D, H, N = 6, 10, 4


def _experts(n: int = N, dtype: torch.dtype = torch.float64, seed: int = 0) -> nn.ModuleList:
    torch.manual_seed(seed)
    return nn.ModuleList(DenseFFN(D, H, 0.0).to(dtype) for _ in range(n))


def _assignments(
    t: int, k: int = 2, n: int = N, seed: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand(t, n, generator=g, dtype=torch.float64)
    idx = scores.topk(k, dim=-1).indices
    gates = torch.softmax(scores.gather(1, idx), dim=-1)
    return idx, gates


def test_dispatch_matches_naive_loop() -> None:
    experts = _experts()
    x = torch.randn(9, D, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    idx, gates = _assignments(9)
    out = dispatch(x, experts, idx, gates)
    ref = dispatch_naive(x, experts, idx, gates)
    assert out.shape == (9, D) and out.dtype == torch.float64
    assert torch.allclose(out, ref, atol=1e-12)


def test_dispatch_gradcheck_double() -> None:
    experts = _experts()
    x = torch.randn(5, D, dtype=torch.float64, requires_grad=True)
    idx, gates = _assignments(5)
    gates = gates.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda a, g: dispatch(a, experts, idx, g), (x, gates))


def test_idle_expert_still_receives_a_gradient() -> None:
    experts = _experts(dtype=torch.float32)
    x = torch.randn(6, D, requires_grad=True)
    idx = torch.tensor([[0, 1]] * 6)  # experts 2 and 3 are idle
    gates = torch.full((6, 2), 0.5)
    out = dispatch(x, experts, idx, gates)
    assert out.dtype == torch.float32
    out.sum().backward()
    for e, expert in enumerate(experts):
        for name, p in expert.named_parameters():
            assert p.grad is not None, f"expert {e} {name} has no grad"
            if e >= 2:
                assert torch.count_nonzero(p.grad) == 0
    assert x.grad is not None and torch.count_nonzero(x.grad) > 0


def test_dispatch_empty_input_and_fp32_promotion() -> None:
    experts = _experts(dtype=torch.float32)
    out = dispatch(
        torch.zeros(0, D), experts, torch.zeros(0, 2, dtype=torch.long), torch.zeros(0, 2)
    )
    assert out.shape == (0, D) and out.dtype == torch.float32
    with torch.autocast("cpu", dtype=torch.bfloat16):
        x = torch.randn(4, D)
        idx, gates = _assignments(4)
        out = dispatch(x, experts, idx, gates.float())
    assert out.dtype == torch.float32


def test_dispatch_rejects_bad_inputs() -> None:
    experts = _experts(dtype=torch.float32)
    x = torch.randn(3, D)
    with pytest.raises(
        RuntimeError
    ):  # out-of-range expert id: caught by one_hot, no extra host sync
        dispatch(x, experts, torch.tensor([[0, 9]] * 3), torch.ones(3, 2))
    with pytest.raises(ValueError, match="T="):
        dispatch(x, experts, torch.zeros(2, 2, dtype=torch.long), torch.ones(2, 2))
    with pytest.raises(ValueError, match="x \\[T, d\\]"):
        dispatch(x.view(1, 3, D), experts, torch.zeros(3, 2, dtype=torch.long), torch.ones(3, 2))
