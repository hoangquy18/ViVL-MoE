from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vivlmoe.losses.teacher import TeacherMSELoss


def test_eq12_is_mean_squared_l2_distance() -> None:
    g = torch.Generator().manual_seed(0)
    s = torch.randn(4, 16, generator=g, requires_grad=True)
    t = torch.randn(4, 16, generator=g)
    loss = TeacherMSELoss("eq12")(s, t)
    ref = ((s - t) ** 2).sum(-1).mean()
    assert torch.allclose(loss, ref)
    loss.backward()
    assert torch.allclose(s.grad, 2 * (s - t) / 4)
    assert torch.allclose(TeacherMSELoss("eq12")(t, t), torch.zeros(()))


def test_legacy_differs_and_is_normalised() -> None:
    g = torch.Generator().manual_seed(1)
    s, t = torch.randn(4, 16, generator=g), torch.randn(4, 16, generator=g)
    legacy = TeacherMSELoss("legacy")(s, t)
    assert torch.allclose(legacy, F.mse_loss(F.normalize(s, dim=-1), F.normalize(t, dim=-1)))
    assert not torch.allclose(legacy, TeacherMSELoss("eq12")(s, t))
    assert torch.allclose(TeacherMSELoss("legacy")(s, 3 * s), torch.zeros(()), atol=1e-7)


def test_teacher_grad_is_detached_and_fp32() -> None:
    s = torch.randn(2, 8, dtype=torch.bfloat16, requires_grad=True)
    t = torch.randn(2, 8, requires_grad=True)
    loss = TeacherMSELoss()(s, t)
    assert loss.dtype == torch.float32
    loss.backward()
    assert t.grad is None and s.grad is not None


def test_validation() -> None:
    with pytest.raises(ValueError):
        TeacherMSELoss("other")
    with pytest.raises(ValueError, match="equal shapes"):
        TeacherMSELoss()(torch.zeros(2, 3), torch.zeros(2, 4))
