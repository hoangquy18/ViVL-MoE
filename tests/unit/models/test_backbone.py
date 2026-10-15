from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from vivlmoe.config.schema import ModelCfg
from vivlmoe.models.backbone import ViVLBackbone
from vivlmoe.models.outputs import TEXT, VISUAL, ContrastiveOutput

from .conftest import make_contrastive_batch


@pytest.fixture()
def backbone(tiny_cfg: ModelCfg) -> ViVLBackbone:
    torch.manual_seed(0)
    return ViVLBackbone.from_config(tiny_cfg).eval()


def test_attribute_names_and_dims(backbone: ViVLBackbone, tiny_cfg: ModelCfg) -> None:
    for name in (
        "vision_encoder",
        "vision_proj",
        "resampler",
        "text_encoder",
        "text_proj",
        "modality_embed",
        "encoder",
    ):
        backbone.get_submodule(name)
    assert backbone.vision_proj.weight.shape == (32, 32)
    assert backbone.modality_embed.weight.shape == (2, 32)
    assert backbone.vision_encoder.num_tokens == 5
    assert math.isclose(backbone.logit_scale.item(), math.log(1 / 0.07), rel_tol=1e-6)
    keys = backbone.state_dict().keys()
    assert "encoder.layers.0.ffn.fc1.weight" in keys and "logit_scale" in keys


def test_contrastive_output_normalised_fp32_and_clamped(backbone: ViVLBackbone) -> None:
    batch = make_contrastive_batch()
    out = backbone(batch, "contrastive")
    assert isinstance(out, ContrastiveOutput)
    assert out.image_embeds.dtype == torch.float32 and out.text_embeds.dtype == torch.float32
    assert torch.allclose(out.image_embeds.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(out.text_embeds.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert math.isclose(out.logit_scale.item(), 1 / 0.07, rel_tol=1e-5)
    assert out.router_outputs == []
    with torch.no_grad():
        backbone.logit_scale.fill_(math.log(1000.0))
    assert math.isclose(backbone(batch, "contrastive").logit_scale.item(), 100.0, rel_tol=1e-5)


def test_separate_passes_no_cross_modal_leakage(backbone: ViVLBackbone) -> None:
    batch = make_contrastive_batch()
    with torch.no_grad():
        ref = backbone(batch, "contrastive")
        other_images = dict(batch, pixel_values=torch.randn_like(batch["pixel_values"]))
        out_i = backbone(other_images, "contrastive")
        other_text = dict(batch, input_ids=torch.randint(3, 1000, batch["input_ids"].shape))
        out_t = backbone(other_text, "contrastive")
    assert torch.allclose(out_i.text_embeds, ref.text_embeds, atol=1e-6)
    assert not torch.allclose(out_i.image_embeds, ref.image_embeds)
    assert torch.allclose(out_t.image_embeds, ref.image_embeds, atol=1e-6)
    assert not torch.allclose(out_t.text_embeds, ref.text_embeds)


def test_encode_modes_match_contrastive(backbone: ViVLBackbone) -> None:
    batch = make_contrastive_batch()
    with torch.no_grad():
        ref = backbone(batch, "contrastive")
        img = backbone({"pixel_values": batch["pixel_values"]}, "encode_image")["embeds"]
        txt = backbone(
            {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
            "encode_text",
        )["embeds"]
    assert torch.allclose(img, ref.image_embeds, atol=1e-6)
    assert torch.allclose(txt, ref.text_embeds, atol=1e-6)
    with pytest.raises(ValueError, match="unknown mode"):
        backbone(batch, "joint")


def test_text_padding_invariance(backbone: ViVLBackbone) -> None:
    batch = make_contrastive_batch()
    longer = dict(batch)
    pad = torch.full((3, 4), 1, dtype=torch.long)
    longer["input_ids"] = torch.cat([batch["input_ids"], pad], dim=1)
    longer["attention_mask"] = torch.cat(
        [batch["attention_mask"], torch.zeros(3, 4, dtype=torch.long)], dim=1
    )
    with torch.no_grad():
        a = backbone(batch, "encode_text")["embeds"]
        b = backbone(longer, "encode_text")["embeds"]
    assert torch.allclose(a, b, atol=1e-5)


def test_masked_mean_ignores_pads() -> None:
    h = torch.arange(12, dtype=torch.float32).view(1, 4, 3)
    mask = torch.tensor([[True, True, False, False]])
    assert torch.allclose(ViVLBackbone.masked_mean(h, mask), h[0, :2].mean(0, keepdim=True))
    assert torch.allclose(ViVLBackbone.masked_mean(h, None), h.mean(1))
    assert torch.equal(
        ViVLBackbone.masked_mean(h, torch.zeros(1, 4, dtype=torch.bool)), torch.zeros(1, 3)
    )
    assert ViVLBackbone.masked_mean(h.to(torch.bfloat16), mask).dtype == torch.float32


def test_joint_layout_and_modality_embeddings(backbone: ViVLBackbone) -> None:
    batch = make_contrastive_batch()
    with torch.no_grad():
        img = backbone.encode_image_tokens(batch["pixel_values"])
        txt = backbone.encode_text_tokens(batch["input_ids"], batch["attention_mask"])
        text_mask = batch["attention_mask"].bool()
        out, key_mask, mids = backbone.forward_joint(img, None, txt, text_mask)
    n_i, n_t = img.shape[1], txt.shape[1]
    assert out.last_hidden_state.shape == (3, n_i + n_t, backbone.dim)
    assert key_mask.shape == (3, n_i + n_t) and key_mask[:, :n_i].all()
    assert torch.equal(key_mask[:, n_i:], text_mask)
    assert (mids[:, :n_i] == VISUAL).all() and (mids[:, n_i:] == TEXT).all()
    # image-only joint pass
    out_i, mask_i, mids_i = backbone.forward_joint(img, None, None, None)
    assert (
        out_i.last_hidden_state.shape == (3, n_i, backbone.dim)
        and mask_i.all()
        and (mids_i == VISUAL).all()
    )
    with pytest.raises(ValueError, match="text_mask"):
        backbone.forward_joint(img, None, txt, None)


def test_modality_embed_in_single_pass_flag(tiny_cfg: ModelCfg) -> None:
    torch.manual_seed(0)
    with_e = ViVLBackbone(tiny_cfg).eval()
    torch.manual_seed(0)
    without = ViVLBackbone(
        dataclasses.replace(tiny_cfg, modality_embed_in_single_pass=False)
    ).eval()
    without.load_state_dict(with_e.state_dict())
    batch = make_contrastive_batch()
    with torch.no_grad():
        a = with_e(batch, "contrastive").image_embeds
        b = without(batch, "contrastive").image_embeds
    assert not torch.allclose(a, b)


def test_cpu_bf16_autocast_keeps_fp32_embeddings(backbone: ViVLBackbone) -> None:
    batch = make_contrastive_batch()
    with torch.autocast("cpu", dtype=torch.bfloat16), torch.no_grad():
        out = backbone(batch, "contrastive")
    assert out.image_embeds.dtype == torch.float32 and out.text_embeds.dtype == torch.float32
    assert out.logit_scale.dtype == torch.float32
    assert torch.allclose(out.image_embeds.norm(dim=-1), torch.ones(3), atol=1e-3)


def test_init_new_weights_leaves_towers_untouched(tiny_cfg: ModelCfg) -> None:
    torch.manual_seed(0)
    bb = ViVLBackbone(tiny_cfg)
    tower_before = {k: v.clone() for k, v in bb.vision_encoder.state_dict().items()}
    proj_before = bb.vision_proj.weight.clone()
    bb.init_new_weights()
    assert all(torch.equal(v, bb.vision_encoder.state_dict()[k]) for k, v in tower_before.items())
    assert not torch.equal(proj_before, bb.vision_proj.weight)
    assert bb.vision_proj.weight.std().item() < 0.05 and bb.vision_proj.bias.abs().sum() == 0


def test_init_module_skips_router_subtrees() -> None:
    """Review fix: a MoE router (M6) keeps its own ``router_init_std`` init; ``init_new_weights`` skips it."""
    from vivlmoe.models.projections import init_module_

    class Ffn(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = torch.nn.Module()
            self.router.proj = torch.nn.Linear(4, 3)
            self.fc = torch.nn.Linear(4, 4)

    ffn = Ffn()
    with torch.no_grad():
        ffn.router.proj.weight.fill_(7.0)
        ffn.fc.weight.fill_(7.0)
        init_module_(torch.nn.Sequential(ffn), 0.02, skip=("router",))
    assert torch.all(ffn.router.proj.weight == 7.0)
    assert not torch.any(ffn.fc.weight == 7.0)
