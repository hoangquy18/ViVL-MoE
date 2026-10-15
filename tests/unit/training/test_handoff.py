"""Stage-3 -> downstream handoff: a backbone checkpoint initialises the seq2seq / classifier heads."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vivlmoe.config.schema import ModelCfg
from vivlmoe.models.factory import build_task_model, load_model_cfg
from vivlmoe.models.towers import find_tower_paths
from vivlmoe.training.checkpoint import (
    ProvenanceTracker,
    build_checkpoint,
    load_partial,
)
from vivlmoe.training.checkpoint import (
    find_tower_paths as name_based_tower_paths,
)
from vivlmoe.training.distributed import DistContext
from vivlmoe.training.state import TrainState

TINY_YAML = Path(__file__).resolve().parents[3] / "configs" / "model" / "tiny.yaml"
CTX = DistContext.single_process("cpu")


@pytest.fixture(scope="module")
def tiny_cfg() -> ModelCfg:
    return load_model_cfg(TINY_YAML)


def _backbone_ckpt(backbone: torch.nn.Module) -> dict:
    """Sparse-style checkpoint: frozen pristine ViT excluded (provenance faked as hf:tiny)."""
    for p in backbone.vision_encoder.parameters():
        p.requires_grad_(False)
    prov = {"vision_encoder": "hf:tiny-vit", "text_encoder": "trained"}
    return build_checkpoint(
        model=backbone,
        excluded=["vision_encoder"],
        provenance=prov,
        config={},
        train_state=TrainState().state_dict(),
        task_state={},
        ctx=CTX,
    )


def _same(a: torch.nn.Module, b: torch.nn.Module, prefix: str) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    keys = [k for k in sa if k.startswith(prefix)]
    return bool(keys) and all(torch.equal(sa[k], sb[k]) for k in keys)


@pytest.mark.parametrize("kind", ["seq2seq", "classifier"])
def test_backbone_checkpoint_into_head(tiny_cfg: ModelCfg, kind: str) -> None:
    torch.manual_seed(0)
    backbone = build_task_model("backbone", tiny_cfg)
    ck = _backbone_ckpt(backbone)
    assert not any(k.startswith("vision_encoder") for k in ck["model"])

    torch.manual_seed(1)
    head = build_task_model(kind, tiny_cfg)
    paths = find_tower_paths(head)
    assert (
        paths
        == name_based_tower_paths(head)
        == {
            "vision_encoder": "backbone.vision_encoder",
            "text_encoder": "backbone.text_encoder",
        }
    )
    current = {"backbone.vision_encoder": "hf:tiny-vit", "backbone.text_encoder": "trained"}
    tracker = ProvenanceTracker(current)
    report = load_partial(head, ck, ["all"], tracker.as_dict())
    assert report.key_prefix == "backbone." and report.rematerialised == ["backbone.vision_encoder"]
    tracker.inherit(report.provenance)
    assert tracker.as_dict() == current
    for module in ("encoder", "text_encoder", "text_proj", "resampler", "modality_embed"):
        assert _same(backbone, head.backbone, module + "."), module
    assert torch.equal(head.backbone.logit_scale, backbone.logit_scale)

    torch.manual_seed(1)
    head2 = build_task_model(kind, tiny_cfg)
    fresh = {k: v.clone() for k, v in head2.state_dict().items()}
    report = load_partial(head2, ck, ["encoder", "text_encoder"], current)
    assert report.rematerialised == [] and set(report.provenance) == {"backbone.text_encoder"}
    assert _same(backbone, head2.backbone, "encoder.") and _same(
        backbone, head2.backbone, "text_encoder."
    )
    assert torch.equal(head2.backbone.text_proj.weight, fresh["backbone.text_proj.weight"])
    # the head's own modules are untouched by either load
    own = "decoder" if kind == "seq2seq" else "cls_head"
    assert all(torch.equal(v, fresh[k]) for k, v in head2.state_dict().items() if k.startswith(own))
