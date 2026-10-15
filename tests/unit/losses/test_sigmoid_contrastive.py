from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from vivlmoe.losses.contrastive import SigmoidContrastiveLoss

WORLD = 2
B_GLOBAL = 8
D_IN, D_OUT = 6, 5


def _features(b: int = 4, d: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    img = F.normalize(torch.randn(b, d, generator=g), dim=-1)
    txt = F.normalize(torch.randn(b, d, generator=g), dim=-1)
    return img, txt


def test_matches_the_siglip_formula() -> None:
    img, txt = _features()
    t, b = torch.tensor(10.0), torch.tensor(-10.0)
    loss, stats = SigmoidContrastiveLoss()(img, txt, t, b)
    expected = 0.0
    for i in range(4):
        for j in range(4):
            z = 1.0 if i == j else -1.0
            logit = float(t * img[i] @ txt[j] + b)
            expected -= math.log(1.0 / (1.0 + math.exp(-z * logit)))
    assert float(loss) == pytest.approx(expected / 4, rel=1e-5)
    assert set(stats) == {"acc_i2t", "acc_t2i"}


def test_temperature_and_bias_get_gradients() -> None:
    img, txt = _features()
    scale = torch.tensor(math.log(10.0), requires_grad=True)
    bias = torch.tensor(-10.0, requires_grad=True)
    loss, _ = SigmoidContrastiveLoss()(img, txt, scale.exp(), bias)
    loss.backward()
    assert scale.grad is not None and bias.grad is not None and float(bias.grad) != 0.0


def test_validation() -> None:
    with pytest.raises(ValueError, match="equal shapes"):
        SigmoidContrastiveLoss()(
            torch.zeros(2, 3), torch.zeros(3, 3), torch.tensor(1.0), torch.tensor(0.0)
        )


# ------------------------------------------------------------------------------- gloo
def _global_batch() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(123)
    return torch.randn(B_GLOBAL, D_IN, generator=g), torch.randn(B_GLOBAL, D_IN, generator=g)


def _model() -> torch.nn.Module:
    torch.manual_seed(7)
    m = torch.nn.Module()
    m.img = torch.nn.Linear(D_IN, D_OUT)
    m.txt = torch.nn.Linear(D_IN, D_OUT)
    m.logit_scale = torch.nn.Parameter(torch.tensor(math.log(10.0)))
    m.logit_bias = torch.nn.Parameter(torch.tensor(-10.0))
    return m


def _run(
    m: torch.nn.Module, x_img: torch.Tensor, x_txt: torch.Tensor, gather: bool
) -> torch.Tensor:
    img = F.normalize(m.img(x_img), dim=-1)
    txt = F.normalize(m.txt(x_txt), dim=-1)
    loss, _ = SigmoidContrastiveLoss(gather=gather)(img, txt, m.logit_scale.exp(), m.logit_bias)
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
        loss = loss / WORLD
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
def test_gloo_text_gather_matches_single_process(tmp_path: Path) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    init_file = tmp_path / "init"
    mp.spawn(_worker, args=(str(init_file),), nprocs=WORLD, join=True)
