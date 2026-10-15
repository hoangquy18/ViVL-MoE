"""Sparse upcycling (paper Sec. 5.3, plan decision 21) on the tiny dense backbone."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from tests.conftest import REPO_ROOT
from tests.unit.models.conftest import TINY_YAML, make_contrastive_batch
from vivlmoe.config.schema import MoECfg
from vivlmoe.models.backbone import ViVLBackbone
from vivlmoe.models.factory import load_model_cfg
from vivlmoe.models.outputs import TEXT
from vivlmoe.upcycling.key_mapping import (
    FFN_TENSORS,
    dense_ffn_key,
    routed_expert_key,
    shared_expert_key,
)
from vivlmoe.upcycling.upcycle import (
    UpcycleError,
    noise_seed,
    upcycle_info,
    upcycle_model,
    upcycle_state_dict,
    verify_equivalence,
)
from vivlmoe.utils.misc import count_parameters


@pytest.fixture(scope="module")
def dense() -> ViVLBackbone:
    torch.manual_seed(0)
    return ViVLBackbone(load_model_cfg(TINY_YAML)).eval()


def _toy_dense_sd(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    sd = {
        "encoder.layers.0.ffn.fc1.weight": torch.randn(8, 4, generator=g),
        "encoder.layers.0.ffn.fc1.bias": torch.randn(8, generator=g),
        "encoder.layers.0.ffn.fc2.weight": torch.randn(4, 8, generator=g),
        "encoder.layers.0.ffn.fc2.bias": torch.randn(4, generator=g),
        "encoder.layers.0.ln1.weight": torch.ones(4),
    }
    return sd


def test_upcycle_state_dict_keys_copies_and_noise(dense: ViVLBackbone) -> None:
    moe_cfg = MoECfg(num_shared=2, num_routed=3, top_k=2, router_init_std=0.3)
    dense_sd = dense.state_dict()
    out = upcycle_state_dict(dense_sd, moe_cfg, noise_std_rel=0.05, seed=0)
    moe = ViVLBackbone(dataclasses.replace(dense.cfg, moe=moe_cfg))
    assert set(out) == set(moe.state_dict())
    moe.load_state_dict(out, strict=True)
    n_layers = dense.cfg.encoder.num_layers
    for layer in range(n_layers):
        for t in FFN_TENSORS:
            base = dense_sd[dense_ffn_key(layer, t)]
            for s in range(2):
                assert torch.equal(out[shared_expert_key(layer, s, t)], base)  # exact copies
            for e in range(3):
                got = out[routed_expert_key(layer, e, t)]
                if t.endswith(".bias"):
                    assert torch.equal(got, base)  # noise_on_bias=False
                else:
                    diff = got - base
                    rel = float(diff.std() / base.std())
                    assert 0.035 < rel < 0.065, rel  # N(0, (0.05 std(W))^2)
                    assert abs(float(diff.mean())) < 0.01 * float(base.std())
        proj = out[f"encoder.layers.{layer}.ffn.router.proj.weight"]
        assert proj.shape == (3, dense.cfg.encoder.dim)
        assert torch.equal(out[f"encoder.layers.{layer}.ffn.router.expert_bias"], torch.zeros(3))
    all_proj = torch.cat(
        [out[f"encoder.layers.{i}.ffn.router.proj.weight"].flatten() for i in range(n_layers)]
    )
    assert 0.2 < float(all_proj.std()) < 0.4  # router_init_std = 0.3
    # every non-FFN tensor is copied verbatim
    ffn_keys = {k for k in dense_sd if ".ffn." in k}
    assert all(torch.equal(out[k], dense_sd[k]) for k in dense_sd if k not in ffn_keys)
    assert not any(k in out for k in ffn_keys)
    # routed experts differ from each other; noise_on_bias perturbs the biases too
    assert not torch.equal(
        out[routed_expert_key(0, 0, "fc1.weight")], out[routed_expert_key(0, 1, "fc1.weight")]
    )
    sd2 = dict(dense_sd)  # the tiny biases are zero (std 0 -> no noise): give fc1.bias some spread
    sd2[dense_ffn_key(0, "fc1.bias")] = torch.randn(
        dense.cfg.encoder.ffn_dim, generator=torch.Generator().manual_seed(4)
    )
    without = upcycle_state_dict(sd2, moe_cfg, noise_std_rel=0.05, seed=0)
    with_bias = upcycle_state_dict(sd2, moe_cfg, noise_std_rel=0.05, noise_on_bias=True, seed=0)
    assert torch.equal(
        without[routed_expert_key(0, 0, "fc1.bias")], sd2[dense_ffn_key(0, "fc1.bias")]
    )
    assert not torch.equal(
        with_bias[routed_expert_key(0, 0, "fc1.bias")], sd2[dense_ffn_key(0, "fc1.bias")]
    )
    assert torch.equal(
        with_bias[routed_expert_key(0, 0, "fc1.weight")], out[routed_expert_key(0, 0, "fc1.weight")]
    )


def test_upcycle_rejects_malformed_inputs(dense: ViVLBackbone) -> None:
    moe_cfg = MoECfg(num_shared=1, num_routed=2, top_k=1)
    moe_sd = upcycle_state_dict(dense.state_dict(), moe_cfg, noise_std_rel=0.0)
    with pytest.raises(UpcycleError, match="no dense FFN keys"):
        upcycle_state_dict(moe_sd, moe_cfg)
    with pytest.raises(UpcycleError, match="already contains MoE keys"):
        upcycle_state_dict({**dense.state_dict(), **moe_sd}, moe_cfg)
    with pytest.raises(ValueError, match="noise_std_rel"):
        upcycle_state_dict(dense.state_dict(), moe_cfg, noise_std_rel=-1.0)
    moe = upcycle_model(dense, moe_cfg, noise_std_rel=0.0)
    with pytest.raises(UpcycleError, match="dense backbone"):
        upcycle_model(moe, moe_cfg)


_HASH_CODE = textwrap.dedent(
    """
    import hashlib, sys, torch
    sys.path.insert(0, {root!r})
    from tests.unit.upcycling.test_upcycle import _toy_dense_sd, _routed_digest
    from vivlmoe.config.schema import MoECfg
    print(_routed_digest(_toy_dense_sd(), MoECfg(num_shared=1, num_routed=3, top_k=2), seed={seed}))
    """
)


def _routed_digest(sd: dict[str, torch.Tensor], moe_cfg: MoECfg, seed: int) -> str:
    out = upcycle_state_dict(sd, moe_cfg, noise_std_rel=0.1, seed=seed)
    parts = [out[k] for k in sorted(out) if "routed_experts" in k or "router.proj" in k]
    return hashlib.sha256(torch.cat([p.flatten() for p in parts]).numpy().tobytes()).hexdigest()


def test_seeds_are_deterministic_across_processes_and_pythonhashseed() -> None:
    sd = _toy_dense_sd()
    moe_cfg = MoECfg(num_shared=1, num_routed=3, top_k=2)
    assert _routed_digest(sd, moe_cfg, 0) == _routed_digest(sd, moe_cfg, 0)
    assert _routed_digest(sd, moe_cfg, 0) != _routed_digest(sd, moe_cfg, 1)
    assert noise_seed(0, 1, 2, "fc1.weight") != noise_seed(0, 1, 3, "fc1.weight")
    assert noise_seed(0, 1, "router", "proj.weight") == noise_seed(0, 1, "router", "proj.weight")
    for hashseed in ("1", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed, "HF_HUB_OFFLINE": "1"}
        proc = subprocess.run(
            [sys.executable, "-c", _HASH_CODE.format(root=str(REPO_ROOT), seed=0)],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == _routed_digest(sd, moe_cfg, 0)


@pytest.mark.parametrize("num_shared", [1, 2, 3])
def test_exact_equivalence_with_noise_zero(dense: ViVLBackbone, num_shared: int) -> None:
    moe_cfg = MoECfg(num_shared=num_shared, num_routed=4, top_k=2)
    moe = upcycle_model(dense, moe_cfg, noise_std_rel=0.0)
    assert moe.cfg.moe == moe_cfg and moe.encoder.is_moe and not moe.training
    batch = make_contrastive_batch()
    report = verify_equivalence(dense, moe, batch, atol=1e-5)
    assert report["max_abs_diff"] < 1e-5
    a, b = dense(batch, "contrastive"), moe(batch, "contrastive")
    assert torch.allclose(a.image_embeds, b.image_embeds, atol=1e-5)
    assert torch.allclose(a.text_embeds, b.text_embeds, atol=1e-5)
    assert torch.equal(a.logit_scale, b.logit_scale)
    assert len(b.router_outputs) == 2 * dense.cfg.encoder.num_layers


def test_half_rule_for_zero_shared_experts(dense: ViVLBackbone) -> None:
    moe = upcycle_model(dense, MoECfg(num_shared=0, num_routed=4, top_k=2), noise_std_rel=0.0)
    batch = make_contrastive_batch()
    with pytest.raises(UpcycleError, match="not equivalent"):
        verify_equivalence(dense, moe, batch, atol=1e-5)
    g = torch.Generator().manual_seed(9)
    x = torch.randn(2, 5, dense.cfg.encoder.dim, generator=g)
    mids = torch.full((2, 5), TEXT)
    for layer in range(dense.cfg.encoder.num_layers):
        dense_ffn = dense.encoder.layers[layer].ffn
        h, ro = moe.encoder.layers[layer].ffn.forward_block(x, mids, None)
        assert torch.allclose(h, 0.5 * dense_ffn(x), atol=1e-6)  # Eq. 18 with N_s = 0
        assert torch.allclose(ro.topk_probs.sum(dim=-1), torch.ones(10), atol=1e-6)


def test_noise_breaks_equivalence_and_grad_state_is_kept(dense: ViVLBackbone) -> None:
    dense.vision_encoder.set_trainable(False) if hasattr(
        dense.vision_encoder, "set_trainable"
    ) else None
    for p in dense.vision_encoder.parameters():
        p.requires_grad_(False)
    moe_cfg = MoECfg(num_shared=1, num_routed=4, top_k=2)
    moe = upcycle_model(dense, moe_cfg, noise_std_rel=0.5, seed=3)
    with pytest.raises(UpcycleError, match="not equivalent"):
        verify_equivalence(dense, moe, make_contrastive_batch(), atol=1e-5)
    assert not any(p.requires_grad for p in moe.vision_encoder.parameters())
    assert all(p.requires_grad for p in moe.text_encoder.parameters())
    for _, ffn in moe.encoder.moe_layers():
        assert ffn.router.proj.weight.requires_grad
        assert all(p.requires_grad for p in ffn.routed_experts.parameters())
    assert count_parameters(moe) > count_parameters(dense)
    info = upcycle_info(moe_cfg, 0.01, False, 3, source="dense.pt")
    assert json.loads(json.dumps(info))["seed_scheme"].startswith("zlib.crc32")
    assert info["num_routed"] == 4 and info["router_init_std"] == moe_cfg.router_init_std
    for p in dense.vision_encoder.parameters():  # restore the module-scoped fixture
        p.requires_grad_(True)


def test_verify_equivalence_shape_mismatch_is_an_error(dense: ViVLBackbone, tmp_path: Path) -> None:
    other = upcycle_model(dense, MoECfg(num_shared=1, num_routed=2, top_k=1), noise_std_rel=0.0)
    batch = make_contrastive_batch(b=2)
    assert verify_equivalence(dense, other, batch)["max_abs_diff"] < 1e-5
    assert dense.training is False and other.training is False
