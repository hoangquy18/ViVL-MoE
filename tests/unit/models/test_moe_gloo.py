"""2-process gloo: idle experts under DDP with accumulation, identical bias updates, eval counters."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from vivlmoe.config.schema import DistCfg, EncoderCfg, MoECfg
from vivlmoe.models.encoder import MultimodalEncoder
from vivlmoe.models.moe.balancing import aux_free_bias_step, expert_fractions
from vivlmoe.models.outputs import TEXT, VISUAL
from vivlmoe.training.ddp import (
    check_all_trainable_params_received_grad,
    sync_context,
    unwrap,
    wrap_ddp,
)
from vivlmoe.training.distributed import (
    all_gather_object,
    all_gather_varlen,
    all_reduce_sum_,
    barrier,
    destroy_distributed,
    init_distributed,
    seed_everything,
)

pytestmark = pytest.mark.gloo
WORLD = 2
ENC = EncoderCfg(
    dim=16, num_layers=2, heads=2, ffn_dim=32, dropout=0.0, attn_dropout=0.0, ffn_inner_dropout=0.0
)
MOE = MoECfg(num_shared=1, num_routed=4, top_k=2)
ACCUM = 2


def _init(rank: int, tmpdir: str, name: str):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(WORLD)
    return init_distributed(
        DistCfg(backend="gloo", timeout_minutes=1), init_method=f"file://{tmpdir}/{name}"
    )


def _micro_batches() -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """World x accum micro-batches (identical on every rank; rank r consumes the r-th slice)."""
    g = torch.Generator().manual_seed(7)
    out = []
    for _ in range(WORLD * ACCUM):
        x = torch.randn(3, 5, ENC.dim, generator=g)
        mask = torch.ones(3, 5, dtype=torch.bool)
        mask[0, 3:] = False
        mids = torch.full((3, 5), VISUAL)
        mids[2] = TEXT
        out.append((x, mask, mids))
    return out


def _loss(
    enc: MultimodalEncoder, x: torch.Tensor, mask: torch.Tensor, mids: torch.Tensor
) -> torch.Tensor:
    out = enc(x, mask, mids)
    return out.last_hidden_state[mask].pow(2).mean()


def _worker(rank: int, tmpdir: str) -> None:
    ctx = _init(rank, tmpdir, "moe")
    try:
        seed_everything(0)
        enc = MultimodalEncoder(ENC, MOE)
        with torch.no_grad():  # expert 3 is idle on every rank (selection uses the bias, Eq. 21)
            for _, ffn in enc.moe_layers():
                ffn.router.expert_bias[3] = -1e4
        ddp = wrap_ddp(enc, ctx, DistCfg(backend="gloo"))
        assert unwrap(ddp) is enc
        batches = _micro_batches()
        mine = batches[rank * ACCUM : (rank + 1) * ACCUM]
        ddp.train()
        for micro, (x, mask, mids) in enumerate(mine):
            with sync_context(ddp, sync=micro == ACCUM - 1):
                out = ddp(x, mask, mids)
                (out.last_hidden_state[mask].pow(2).mean() / ACCUM).backward()
        check_all_trainable_params_received_grad(ddp)  # idle expert 3 included
        for _, ffn in enc.moe_layers():
            assert all(p.grad is not None for p in ffn.routed_experts[3].parameters())
            assert int(ffn.router.dispatch_counts[3]) == 0
        grads = torch.cat([p.grad.flatten() for p in enc.parameters()])
        gathered = all_gather_varlen(grads.unsqueeze(0), ctx)
        assert torch.equal(gathered[0], gathered[1])
        # reference: single model over all world x accum micro-batches (mean of the losses)
        seed_everything(0)
        ref = MultimodalEncoder(ENC, MOE)
        ref.load_state_dict(enc.state_dict())
        ref.train()
        loss = torch.stack([_loss(ref, *b) for b in batches]).mean()
        loss.backward()
        ref_grads = torch.cat([p.grad.flatten() for p in ref.parameters()])
        assert torch.allclose(grads, ref_grads, atol=1e-5), float((grads - ref_grads).abs().max())
        # aux-free bias update on all-reduced counts -> identical biases on both ranks
        local = enc.dispatch_stats().clone()
        total = all_reduce_sum_(local.clone(), ctx)
        assert int(total[:, -1].sum()) == 2 * int(local[:, -1].sum())
        expected_f = expert_fractions(total)
        before = torch.stack(enc.router_bias_state())
        logs = aux_free_bias_step(enc, ctx, eta=0.5)
        after = torch.stack(enc.router_bias_state())
        assert torch.allclose(
            after, before + 0.5 * torch.sign(MOE.top_k / MOE.num_routed - expected_f)
        )
        biases = all_gather_object(after.tolist(), ctx)
        assert biases[0] == biases[1]
        assert logs["moe/tokens"] == float(total[0, -1])
        assert int(enc.dispatch_stats().sum()) == 0
        # eval forward leaves the counters untouched
        ddp.eval()
        with torch.no_grad():
            ddp(*mine[0])
        assert int(enc.dispatch_stats().sum()) == 0
        barrier(ctx)
    finally:
        destroy_distributed()


def test_moe_ddp_idle_expert_accumulation_and_bias_update(tmp_path: Path) -> None:
    mp.spawn(_worker, args=(str(tmp_path),), nprocs=WORLD, join=True)
