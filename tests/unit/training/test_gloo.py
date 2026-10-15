"""2-process gloo tests: collectives, ordered gather (incl. empty shard), DDP + no_sync grads."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from vivlmoe.config.schema import DistCfg
from vivlmoe.tasks.evaluators.base import gather_ordered, gather_ordered_tensors
from vivlmoe.training.ddp import (
    check_all_trainable_params_received_grad,
    sync_context,
    unwrap,
    wrap_ddp,
)
from vivlmoe.training.distributed import (
    all_gather_object,
    all_gather_varlen,
    all_reduce_mean,
    all_reduce_sum_,
    barrier,
    broadcast_object,
    destroy_distributed,
    init_distributed,
    seed_everything,
)

pytestmark = pytest.mark.gloo
WORLD = 2


def _init(rank: int, tmpdir: str, name: str):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(WORLD)
    return init_distributed(
        DistCfg(backend="gloo", timeout_minutes=1), init_method=f"file://{tmpdir}/{name}"
    )


def _collectives_worker(rank: int, tmpdir: str) -> None:
    ctx = _init(rank, tmpdir, "coll")
    try:
        assert ctx.is_distributed and ctx.world_size == WORLD and ctx.is_main == (rank == 0)
        t = torch.tensor([rank + 1.0])
        assert all_reduce_sum_(t, ctx).item() == 3.0
        assert all_reduce_mean(torch.tensor([rank + 1.0]), ctx).item() == 1.5
        parts = all_gather_varlen(torch.full((2 * rank + 1, 3), float(rank)), ctx)
        assert [p.shape[0] for p in parts] == [1, 3] and parts[1].eq(1.0).all()
        empty = all_gather_varlen(torch.zeros(0, 2), ctx)
        assert [p.shape for p in empty] == [(0, 2), (0, 2)]
        assert all_gather_object({"r": rank}, ctx) == [{"r": 0}, {"r": 1}]
        assert broadcast_object({"from": rank}, ctx) == {"from": 0}
        # ordered gather: rank 0 holds uids [2, 0], rank 1 holds [1]
        ids = [2, 0] if rank == 0 else [1]
        payload = [f"p{i}" for i in ids]
        assert gather_ordered(ids, payload, ctx) == ["p0", "p1", "p2"]
        vals = torch.tensor([[20.0], [0.0]]) if rank == 0 else torch.tensor([[10.0]])
        got = gather_ordered_tensors(torch.tensor(ids), vals, ctx)
        assert torch.equal(got, torch.tensor([[0.0], [10.0], [20.0]]))
        # empty shard on rank 1 (N = 1): both ranks must still call the collective
        ids1 = [0] if rank == 0 else []
        assert gather_ordered(ids1, ["only"] if rank == 0 else [], ctx) == ["only"]
        got1 = gather_ordered_tensors(
            torch.tensor(ids1, dtype=torch.int64), torch.ones(len(ids1), 2), ctx
        )
        assert got1.shape == (1, 2)
        # duplicated / missing ids raise on every rank
        with pytest.raises(RuntimeError, match="not 0..1"):
            gather_ordered([0] if rank == 0 else [0], ["a"], ctx)
        barrier(ctx)
    finally:
        destroy_distributed()


def _ddp_worker(rank: int, tmpdir: str) -> None:
    ctx = _init(rank, tmpdir, "ddp")
    try:
        seed_everything(0)
        model = nn.Linear(4, 1)
        ddp = wrap_ddp(model, ctx, DistCfg(backend="gloo"))
        assert unwrap(ddp) is model
        g = torch.Generator().manual_seed(123)
        x, y = torch.randn(8, 4, generator=g), torch.randn(8, generator=g)
        xr, yr = x[rank * 4 : (rank + 1) * 4], y[rank * 4 : (rank + 1) * 4]
        for micro in range(2):  # two accumulation micro-batches, no_sync on the first
            with sync_context(ddp, sync=micro == 1):
                pred = ddp(xr[micro * 2 : micro * 2 + 2]).squeeze(-1)
                loss = ((pred - yr[micro * 2 : micro * 2 + 2]) ** 2).mean() / 2
                loss.backward()
        check_all_trainable_params_received_grad(ddp)
        grads = torch.cat([p.grad.flatten() for p in model.parameters()])
        ref = nn.Linear(4, 1)
        ref.load_state_dict(model.state_dict())
        ((ref(x).squeeze(-1) - y) ** 2).mean().backward()
        ref_grads = torch.cat([p.grad.flatten() for p in ref.parameters()])
        assert torch.allclose(grads, ref_grads, atol=1e-6), (grads, ref_grads)
        gathered = all_gather_varlen(grads.unsqueeze(0), ctx)
        assert torch.equal(gathered[0], gathered[1])
        # no_sync alone must NOT reduce: local grads differ between ranks
        ddp.zero_grad(set_to_none=True)
        with sync_context(ddp, sync=False):
            ((ddp(xr).squeeze(-1) - yr) ** 2).mean().backward()
        local = torch.cat([p.grad.flatten() for p in model.parameters()]).unsqueeze(0)
        both = all_gather_varlen(local, ctx)
        assert not torch.allclose(both[0], both[1])
        barrier(ctx)
    finally:
        destroy_distributed()


def _spawn(fn, tmp_path: Path) -> None:
    mp.spawn(fn, args=(str(tmp_path),), nprocs=WORLD, join=True)


@pytest.mark.timeout(120)
def test_collectives_and_ordered_gather(tmp_path: Path) -> None:
    _spawn(_collectives_worker, tmp_path)


@pytest.mark.timeout(120)
def test_ddp_wrap_no_sync_and_identical_grads(tmp_path: Path) -> None:
    _spawn(_ddp_worker, tmp_path)
