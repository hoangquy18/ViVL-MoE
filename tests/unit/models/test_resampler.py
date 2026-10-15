from __future__ import annotations

import torch

from vivlmoe.config.schema import ModelCfg, ResamplerCfg
from vivlmoe.models.resampler import PerceiverResampler


def test_tiny_shapes(tiny_cfg: ModelCfg) -> None:
    r = PerceiverResampler(tiny_cfg.encoder.dim, tiny_cfg.resampler)
    out = r(torch.randn(2, 5, tiny_cfg.encoder.dim))
    assert out.shape == (2, tiny_cfg.resampler.num_latents, tiny_cfg.encoder.dim)


def test_paper_shapes_730_to_64() -> None:
    cfg = ResamplerCfg(num_latents=64, depth=2, heads=8, dim_head=64, ff_mult=4, dropout=0.0)
    r = PerceiverResampler(768, cfg).eval()
    assert r.latents.shape == (64, 768)
    attn = r.layers[0]["attn"]
    assert attn.to_q.weight.shape == (512, 768)
    assert attn.to_kv.weight.shape == (1024, 768)
    assert attn.to_out.weight.shape == (768, 512)
    assert r.layers[0]["ff"].fc1.weight.shape == (3072, 768)
    with torch.no_grad():
        out = r(torch.randn(1, 730, 768))
    assert out.shape == (1, 64, 768)
    assert torch.isfinite(out).all()


def test_kv_are_patch_tokens_only_unless_opt_in() -> None:
    base = ResamplerCfg(num_latents=4, depth=1, heads=2, dim_head=8, ff_mult=2, dropout=0.0)
    torch.manual_seed(0)
    r0 = PerceiverResampler(16, base).eval()
    r1 = PerceiverResampler(
        16, ResamplerCfg(**{**base.__dict__, "concat_latents_to_kv": True})
    ).eval()
    r1.load_state_dict(r0.state_dict())
    x = torch.randn(2, 6, 16)
    with torch.no_grad():
        assert not torch.allclose(r0(x), r1(x))


def test_rejects_wrong_width(tiny_cfg: ModelCfg) -> None:
    r = PerceiverResampler(tiny_cfg.encoder.dim, tiny_cfg.resampler)
    try:
        r(torch.randn(1, 5, tiny_cfg.encoder.dim + 1))
    except ValueError as exc:
        assert "resampler expects" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
