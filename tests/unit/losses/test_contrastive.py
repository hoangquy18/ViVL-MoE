from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from vivlmoe.losses.contrastive import ContrastiveLoss

WORLD = 2
B_GLOBAL = 8
D_IN, D_OUT = 6, 5


def _features(b: int = 4, d: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    img = F.normalize(torch.randn(b, d, generator=g), dim=-1).requires_grad_(True)
    txt = F.normalize(torch.randn(b, d, generator=g), dim=-1).requires_grad_(True)
    return img, txt


def test_matches_manual_formula_mean_and_sum() -> None:
    img, txt = _features()
    scale = torch.tensor(1 / 0.07).exp().clamp(max=100.0)
    loss_mean, stats = ContrastiveLoss(gather=True, symmetric_reduction="mean")(img, txt, scale)
    loss_sum, _ = ContrastiveLoss(gather=False, symmetric_reduction="sum")(img, txt, scale)
    logits = scale * img @ txt.t()
    labels = torch.arange(4)
    ref = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    assert torch.allclose(loss_sum, ref)
    assert torch.allclose(loss_mean, 0.5 * ref)
    assert set(stats) == {"acc_i2t", "acc_t2i"} and 0.0 <= stats["acc_i2t"] <= 1.0
    loss_mean.backward()
    assert img.grad is not None and txt.grad is not None


def test_fp32_and_clamped_scale() -> None:
    img, txt = _features()
    logit_scale = torch.tensor(math.log(1000.0), requires_grad=True)
    scale = logit_scale.clamp(max=math.log(100.0)).exp()
    assert math.isclose(scale.item(), 100.0, rel_tol=1e-6)
    loss, _ = ContrastiveLoss()(
        img.to(torch.bfloat16), txt.to(torch.bfloat16), scale.to(torch.bfloat16)
    )
    assert loss.dtype == torch.float32
    loss.backward()
    assert logit_scale.grad is not None and logit_scale.grad.item() == 0.0  # clamped: no gradient


def test_perfect_alignment_has_zero_loss_limit() -> None:
    img = F.normalize(torch.eye(4, 8), dim=-1)
    loss, stats = ContrastiveLoss()(img, img, torch.tensor(100.0))
    assert loss.item() < 1e-3 and stats["acc_i2t"] == 1.0


def test_validation() -> None:
    with pytest.raises(ValueError):
        ContrastiveLoss(symmetric_reduction="avg")
    with pytest.raises(ValueError, match="equal shapes"):
        ContrastiveLoss()(torch.zeros(2, 3), torch.zeros(3, 3), torch.tensor(1.0))


# ------------------------------------------------------------------------------- gloo
def _global_batch() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(123)
    return torch.randn(B_GLOBAL, D_IN, generator=g), torch.randn(B_GLOBAL, D_IN, generator=g)


def _model() -> torch.nn.Module:
    torch.manual_seed(7)
    m = torch.nn.Module()
    m.img = torch.nn.Linear(D_IN, D_OUT)
    m.txt = torch.nn.Linear(D_IN, D_OUT)
    m.logit_scale = torch.nn.Parameter(torch.tensor(math.log(1 / 0.07)))
    return m


def _run(
    m: torch.nn.Module, x_img: torch.Tensor, x_txt: torch.Tensor, gather: bool
) -> torch.Tensor:
    img = F.normalize(m.img(x_img), dim=-1)
    txt = F.normalize(m.txt(x_txt), dim=-1)
    loss, _ = ContrastiveLoss(gather=gather, debug_check_shapes=gather)(
        img, txt, m.logit_scale.exp()
    )
    loss.backward()
    return loss.detach()


def _worker(rank: int, init_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD)
    try:
        torch.set_num_threads(1)
        x_img, x_txt = _global_batch()
        per = B_GLOBAL // WORLD
        sl = slice(rank * per, (rank + 1) * per)
        m = _model()
        loss = _run(m, x_img[sl], x_txt[sl], gather=True)
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss = loss / WORLD  # what DDP-style logging reports
        grads = [p.grad.clone() for p in m.parameters()]
        for g in grads:  # DDP averages gradients across ranks
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
            g.div_(WORLD)
        if rank == 0:
            ref = _model()
            ref_loss = _run(ref, x_img, x_txt, gather=False)
            assert torch.allclose(loss, ref_loss, atol=1e-6), (loss, ref_loss)
            for g, p in zip(grads, ref.parameters(), strict=True):
                assert torch.allclose(g, p.grad, atol=1e-6), (g, p.grad)
    finally:
        dist.barrier()
        dist.destroy_process_group()


@pytest.mark.gloo
def test_gloo_gather_matches_single_process(tmp_path: Path) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    init_file = tmp_path / "init"
    mp.spawn(_worker, args=(str(init_file),), nprocs=WORLD, join=True)
