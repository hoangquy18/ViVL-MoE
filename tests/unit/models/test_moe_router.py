"""Router (Eq. 15 / 21), MoEFFN (Eq. 18) and MoE buffer semantics on tiny modules (M6)."""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from tests.unit.models.conftest import TINY_YAML, make_contrastive_batch
from vivlmoe.config.schema import ConfigError, EncoderCfg, MoECfg
from vivlmoe.models.backbone import ViVLBackbone
from vivlmoe.models.encoder import MultimodalEncoder
from vivlmoe.models.factory import load_model_cfg
from vivlmoe.models.moe.moe_ffn import MoEFFN
from vivlmoe.models.moe.router import Router
from vivlmoe.models.outputs import TEXT, VISUAL, RouterOutput
from vivlmoe.training.checkpoint import ProvenanceTracker, build_checkpoint, load_model_state
from vivlmoe.training.distributed import DistContext
from vivlmoe.upcycling.key_mapping import moe_ffn_keys

CTX = DistContext.single_process("cpu")
DIM, N_R, K = 8, 4, 2


def _router(aux_free_bias: bool = True, seed: int = 0) -> Router:
    torch.manual_seed(seed)
    return Router(DIM, N_R, K, aux_free_bias=aux_free_bias, init_std=0.5)


def _tokens(t: int = 12, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(t, DIM, generator=g)
    mids = torch.randint(0, 2, (t,), generator=g)
    return x, mids


def test_selection_uses_bias_but_probs_do_not() -> None:
    router = _router().eval()
    x, mids = _tokens()
    logits = F.linear(x, router.proj.weight)
    worst = logits.argmin(dim=-1)  # per-token expert with the smallest logit
    # a huge bias on expert 0 forces it into every selection (Eq. 21) ...
    router.expert_bias.zero_()
    router.expert_bias[0] = 1e3
    out = router(x, mids, layer_idx=3)
    assert isinstance(out, RouterOutput) and out.layer_idx == 3
    assert (out.topk_idx == 0).any(dim=-1).all()
    # ... but the gates are the renormalised softmax of z alone (Eq. 15), never of z + b
    expected = torch.softmax(logits.gather(1, out.topk_idx), dim=-1)
    assert torch.allclose(out.topk_probs, expected, atol=1e-6)
    assert torch.allclose(out.logits, logits, atol=1e-6)
    biased = torch.softmax((logits + router.expert_bias).gather(1, out.topk_idx), dim=-1)
    assert not torch.allclose(out.topk_probs, biased)
    # with the bias switched off the plain top-k on z is used
    plain = _router(aux_free_bias=False).eval()
    plain.load_state_dict(router.state_dict())
    ref = plain(x, mids, layer_idx=0)
    assert torch.equal(
        ref.topk_idx.sort(dim=-1).values, logits.topk(K, dim=-1).indices.sort(dim=-1).values
    )
    if bool((worst == 0).any()):
        assert not torch.equal(ref.topk_idx.sort(dim=-1).values, out.topk_idx.sort(dim=-1).values)


def test_moecfg_validates_top_k_and_router_rejects_it() -> None:
    with pytest.raises(ConfigError, match="top_k"):
        MoECfg(num_routed=1, top_k=2)
    assert MoECfg(num_routed=1, top_k=1).top_k == 1  # paper: k = 1 when N_r = 1 (nr1_k1.yaml)
    with pytest.raises(ConfigError, match="top_k"):
        MoECfg(num_routed=4, top_k=0)
    with pytest.raises(ConfigError, match="num_shared"):
        MoECfg(num_shared=-1)
    with pytest.raises(ValueError, match="top_k"):
        Router(DIM, 2, 3)


def test_counts_only_in_training_and_bias_untouched_in_forward() -> None:
    router = _router()
    x, mids = _tokens(t=10)
    router.expert_bias.fill_(0.25)
    router.train()
    router(x, mids, layer_idx=0)
    assert int(router.dispatch_counts.sum()) == 10 * K and int(router.token_count) == 10
    assert router.dispatch_counts.dtype == torch.int64 and router.token_count.dtype == torch.int64
    router(x, mids, layer_idx=0)
    assert int(router.token_count) == 20  # accumulates across micro-batches
    router.eval()
    counts = router.dispatch_counts.clone()
    router(x, mids, layer_idx=0)
    assert torch.equal(router.dispatch_counts, counts) and int(router.token_count) == 20
    assert torch.equal(router.expert_bias, torch.full((N_R,), 0.25))  # never updated in forward
    router.reset_counts()
    assert int(router.dispatch_counts.sum()) == 0 and int(router.token_count) == 0
    # counts are one_hot sums over tokens and slots, never bincount over a flattened view
    router.train()
    out = router(x, mids, layer_idx=0)
    assert torch.equal(router.dispatch_counts, F.one_hot(out.topk_idx, N_R).sum(dim=(0, 1)))


def test_router_output_stays_fp32_under_cpu_bf16_autocast() -> None:
    router = _router()
    x, mids = _tokens()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = router(x, mids, layer_idx=0)
        assert out.logits.dtype == torch.float32 and out.topk_probs.dtype == torch.float32
        ffn = MoEFFN(DIM, 16, 0.0, MoECfg(num_shared=1, num_routed=N_R, top_k=K), layer_idx=0)
        h, ro = ffn.forward_block(x.view(3, 4, DIM), mids.view(3, 4), None)
    assert h.dtype == x.dtype and ro.logits.dtype == torch.float32
    ref = torch.softmax(F.linear(x, router.proj.weight), dim=-1)
    assert torch.allclose(torch.softmax(out.logits, dim=-1), ref, atol=1e-6)


def test_padded_tokens_are_not_routed() -> None:
    torch.manual_seed(0)
    ffn = MoEFFN(DIM, 16, 0.1, MoECfg(num_shared=1, num_routed=N_R, top_k=K), layer_idx=1).eval()
    g = torch.Generator().manual_seed(3)
    x = torch.randn(2, 5, DIM, generator=g)
    mids = torch.full((2, 5), TEXT)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    h, ro = ffn.forward_block(x, mids, mask)
    assert ro.num_tokens == 7 and ro.layer_idx == 1 and ro.modality_ids.shape == (7,)
    assert torch.equal(h[~mask], torch.zeros(3, DIM))
    full, _ = ffn.forward_block(x, mids, None)
    assert torch.allclose(h[mask], full[mask], atol=1e-6)  # per-token op: valid tokens unchanged
    ffn.train()
    ffn.forward_block(x, mids, mask)
    assert int(ffn.router.token_count) == 7  # padded positions never enter the counters


def test_eq18_shared_mean_and_half_rule() -> None:
    torch.manual_seed(0)
    x, mids = _tokens(t=6)
    ffn2 = MoEFFN(DIM, 16, 0.0, MoECfg(num_shared=2, num_routed=N_R, top_k=K), layer_idx=0).eval()
    h, ro = ffn2.forward_tokens(x, mids)
    shared = torch.stack([e(x) for e in ffn2.shared_experts]).mean(dim=0)
    routed = sum(
        ro.topk_probs[:, j : j + 1]
        * torch.stack([ffn2.routed_experts[int(i)](x[t]) for t, i in enumerate(ro.topk_idx[:, j])])
        for j in range(K)
    )
    assert torch.allclose(h, 0.5 * (shared + routed), atol=1e-6)
    ffn0 = MoEFFN(DIM, 16, 0.0, MoECfg(num_shared=0, num_routed=N_R, top_k=K), layer_idx=0).eval()
    ffn0.routed_experts.load_state_dict(ffn2.routed_experts.state_dict())
    ffn0.router.load_state_dict(ffn2.router.state_dict())
    h0, _ = ffn0.forward_tokens(x, mids)
    assert torch.allclose(h0, 0.5 * routed, atol=1e-6)


def test_backbone_router_init_std_and_state_dict_keys() -> None:
    cfg = load_model_cfg(TINY_YAML)
    moe = MoECfg(num_shared=1, num_routed=8, top_k=2, router_init_std=0.5)
    torch.manual_seed(0)
    model = ViVLBackbone(dataclasses.replace(cfg, moe=moe))
    layers = model.encoder.moe_layers()
    assert len(layers) == cfg.encoder.num_layers and model.encoder.is_moe
    for _, ffn in layers:
        std = float(ffn.router.proj.weight.std())
        assert 0.35 < std < 0.65, std  # init_new_weights (0.02) must not overwrite the router
        assert torch.equal(ffn.router.expert_bias, torch.zeros(8))
    sd = model.state_dict()
    for layer in range(cfg.encoder.num_layers):
        prefix = f"encoder.layers.{layer}.ffn."
        assert {k for k in sd if k.startswith(prefix)} == set(moe_ffn_keys(layer, moe))
    assert not any("dispatch_counts" in k or "token_count" in k for k in sd)


def test_moe_buffers_roundtrip_through_checkpoint() -> None:
    cfg = dataclasses.replace(
        load_model_cfg(TINY_YAML), moe=MoECfg(num_shared=1, num_routed=4, top_k=2)
    )
    torch.manual_seed(0)
    model = ViVLBackbone(cfg)
    enc = model.encoder
    for i, (_, ffn) in enumerate(enc.moe_layers()):
        ffn.router.expert_bias.fill_(0.1 * (i + 1))
        ffn.router.dispatch_counts.fill_(7)
        ffn.router.token_count.fill_(9)
    prov = ProvenanceTracker.from_config(cfg, model).as_dict()
    ck = build_checkpoint(
        model=model,
        excluded=[],
        provenance=prov,
        config={"model": {}},
        train_state={"step": 1, "epoch": 0},
        task_state={},
        ctx=CTX,
    )
    keys = set(ck["model"])
    assert "encoder.layers.0.ffn.router.expert_bias" in keys
    assert not any("dispatch_counts" in k or "token_count" in k for k in keys)
    torch.manual_seed(1)
    fresh = ViVLBackbone(cfg)
    load_model_state(fresh, ck, prov)
    for (_, a), (_, b) in zip(enc.moe_layers(), fresh.encoder.moe_layers(), strict=True):
        assert torch.equal(a.router.expert_bias, b.router.expert_bias)
        assert int(b.router.dispatch_counts.sum()) == 0 and int(b.router.token_count) == 0


def test_encoder_moe_api_and_contrastive_router_outputs() -> None:
    cfg = dataclasses.replace(
        load_model_cfg(TINY_YAML), moe=MoECfg(num_shared=1, num_routed=4, top_k=2)
    )
    torch.manual_seed(0)
    model = ViVLBackbone(cfg).train()
    batch = make_contrastive_batch()
    out = model(batch, "contrastive")
    n_layers = cfg.encoder.num_layers
    assert [ro.layer_idx for ro in out.router_outputs] == list(range(n_layers)) * 2
    visual, text = out.router_outputs[:n_layers], out.router_outputs[n_layers:]
    n_valid = int(batch["attention_mask"].sum())
    assert all(ro.num_tokens == 3 * cfg.resampler.num_latents for ro in visual)
    assert all(bool((ro.modality_ids == VISUAL).all()) for ro in visual)
    assert all(ro.num_tokens == n_valid and bool((ro.modality_ids == TEXT).all()) for ro in text)
    stats = model.encoder.dispatch_stats()
    assert stats.shape == (n_layers, 5) and stats.dtype == torch.int64
    assert (stats[:, -1] == 3 * cfg.resampler.num_latents + n_valid).all()
    assert (stats[:, :-1].sum(dim=1) == 2 * stats[:, -1]).all()
    model.encoder.reset_dispatch_stats()
    assert int(model.encoder.dispatch_stats().sum()) == 0
    dense = MultimodalEncoder(EncoderCfg(dim=8, num_layers=1, heads=2, ffn_dim=16))
    assert not dense.is_moe and dense.moe_layers() == [] and dense.dispatch_stats().numel() == 0
    with pytest.raises(RuntimeError):
        dense.apply_router_bias_update(torch.zeros(1, 4), 0.1)
