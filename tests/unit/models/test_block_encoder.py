from __future__ import annotations

import pytest
import torch

from vivlmoe.config.schema import EncoderCfg
from vivlmoe.models.attention import MultiHeadSelfAttention
from vivlmoe.models.block import TransformerBlock
from vivlmoe.models.encoder import MultimodalEncoder
from vivlmoe.models.ffn import DenseFFN


def _cfg(**kw) -> EncoderCfg:
    base = dict(
        dim=16,
        num_layers=2,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        attn_dropout=0.0,
        ffn_inner_dropout=0.0,
    )
    return EncoderCfg(**{**base, **kw})


def test_post_ln_order_matches_reference() -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    block = TransformerBlock(cfg, DenseFFN(cfg.dim, cfg.ffn_dim)).eval()
    x = torch.randn(2, 5, cfg.dim)
    mids = torch.zeros(2, 5, dtype=torch.long)
    out, ro = block(x, None, mids)
    assert ro is None
    y = block.ln1(x + block.attn(x, None))  # MHSA -> Add & LN (Fig. 3)
    ref = block.ln2(y + block.ffn(y))  # FFN -> Add & LN
    assert torch.allclose(out, ref, atol=1e-6)
    pre_ln = x + block.attn(block.ln1(x), None)
    pre_ln = pre_ln + block.ffn(block.ln2(pre_ln))
    assert not torch.allclose(out, pre_ln, atol=1e-3)


def test_attention_mask_semantics_true_keeps() -> None:
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(16, 4).eval()
    x = torch.randn(1, 4, 16)
    keep_all = attn(x, torch.ones(1, 4, dtype=torch.bool))
    assert torch.allclose(keep_all, attn(x, None), atol=1e-6)
    with pytest.raises(ValueError, match="key_padding_mask must be bool"):
        attn(x, torch.ones(1, 4, dtype=torch.long))


def test_key_padding_invariance() -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    enc = MultimodalEncoder(cfg).eval()
    x = torch.randn(2, 6, cfg.dim)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[0, 4:] = False
    mask[1, 2:] = False
    mids = torch.ones(2, 6, dtype=torch.long)
    x2 = x.clone()
    x2[~mask] = torch.randn((~mask).sum(), cfg.dim) * 10
    h1 = enc(x, mask, mids).last_hidden_state
    h2 = enc(x2, mask, mids).last_hidden_state
    assert torch.allclose(h1[mask], h2[mask], atol=1e-5)
    assert not torch.allclose(h1[~mask], h2[~mask])
    # a shorter unpadded sequence equals the valid prefix of the padded one
    h_short = enc(x[1:2, :2], None, mids[1:2, :2]).last_hidden_state
    assert torch.allclose(h_short[0], h1[1, :2], atol=1e-5)


def test_dense_encoder_moe_api_is_empty() -> None:
    enc = MultimodalEncoder(_cfg())
    assert not enc.is_moe
    assert enc.moe_layers() == []
    assert enc.dispatch_stats().shape == (0, 0)
    assert enc.dispatch_stats().dtype == torch.int64
    assert enc.router_bias_state() == []
    enc.reset_dispatch_stats()
    with pytest.raises(RuntimeError, match="dense encoder"):
        enc.apply_router_bias_update(torch.zeros(2, 8), 1e-3)
    assert [n for n, _ in enc.named_parameters()][:2] == [
        "layers.0.attn.qkv.weight",
        "layers.0.attn.qkv.bias",
    ]
    assert any(n == "layers.0.ffn.fc1.weight" for n, _ in enc.named_parameters())


def test_encoder_validates_shapes() -> None:
    enc = MultimodalEncoder(_cfg())
    with pytest.raises(ValueError, match="modality_ids"):
        enc(torch.randn(1, 3, 16), None, torch.zeros(1, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="encoder expects"):
        enc(torch.randn(1, 3, 8), None, torch.zeros(1, 3, dtype=torch.long))


def test_dense_ffn_forward_block_returns_no_router_output() -> None:
    ffn = DenseFFN(8, 16, inner_dropout=0.0)
    x = torch.randn(2, 3, 8)
    h, ro = ffn.forward_block(x, torch.zeros(2, 3, dtype=torch.long), None)
    assert ro is None
    assert torch.allclose(h, ffn(x))
