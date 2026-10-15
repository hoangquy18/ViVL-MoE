from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from vivlmoe.config.schema import ModelCfg
from vivlmoe.models.factory import build_task_model
from vivlmoe.models.freezing import (
    apply_freeze,
    assert_no_unused_trainable_params,
    freeze_paths_for_preset,
)
from vivlmoe.models.heads.classifier import MultiImageClassifier
from vivlmoe.models.outputs import ClassifierOutput

B, M, A, L = 2, 3, 6, 9


def _batch() -> dict:
    g = torch.Generator().manual_seed(4)
    attn = torch.ones(B, A, L, dtype=torch.long)
    attn[:, :, 7:] = 0
    return {
        "pixel_values": torch.randn(B, M, 3, 32, 32, generator=g),
        "image_mask": torch.tensor([[True, True, False], [True, False, False]]),
        "input_ids": torch.randint(3, 1000, (B, A, L), generator=g),
        "attention_mask": attn,
        "labels": torch.randint(0, 4, (B, A), generator=g),
        "uids": list(range(B)),
    }


@pytest.fixture()
def model(tiny_cfg: ModelCfg) -> MultiImageClassifier:
    torch.manual_seed(0)
    return build_task_model("classifier", tiny_cfg).eval()


def test_shapes_and_loss_is_sum_of_aspect_means(model: MultiImageClassifier) -> None:
    batch = _batch()
    with torch.no_grad():
        out = model(batch, "joint_cls")
    assert isinstance(out, ClassifierOutput)
    assert out.logits.shape == (B, A, 4) and out.logits.dtype == torch.float32
    ref = sum(F.cross_entropy(out.logits[:, a], batch["labels"][:, a]) for a in range(A))
    assert torch.allclose(out.loss, ref)
    with torch.no_grad():
        no_labels = model({k: v for k, v in batch.items() if k != "labels"}, "joint_cls")
    assert no_labels.loss is None


def test_padded_image_slot_invariance(model: MultiImageClassifier) -> None:
    batch = _batch()
    changed = dict(batch, pixel_values=batch["pixel_values"].clone())
    changed["pixel_values"][0, 2] = torch.randn(3, 32, 32) * 5  # masked slot
    changed["pixel_values"][1, 1:] = torch.randn(2, 3, 32, 32) * 5
    with torch.no_grad():
        a = model(batch, "joint_cls").logits
        b = model(changed, "joint_cls").logits
    assert torch.allclose(a, b, atol=1e-5)
    changed["pixel_values"][0, 0] = torch.randn(3, 32, 32)  # a valid slot
    with torch.no_grad():
        c = model(changed, "joint_cls").logits
    assert not torch.allclose(a[0], c[0]) and torch.allclose(a[1], c[1], atol=1e-5)


@pytest.mark.parametrize("pool", ["mean_joint", "mean_text", "concat_mean_v_t"])
def test_zero_image_sample(tiny_cfg: ModelCfg, pool: str) -> None:
    cfg = dataclasses.replace(
        tiny_cfg, classifier=dataclasses.replace(tiny_cfg.classifier, pool=pool)
    )
    torch.manual_seed(0)
    model = build_task_model("classifier", cfg).eval()
    batch = _batch()
    batch["image_mask"] = torch.tensor([[True, False, False], [False, False, False]])
    with torch.no_grad():
        out = model(batch, "joint_cls")
        # the zero-image sample only depends on its text: swapping its images changes nothing
        other = dict(batch, pixel_values=torch.randn_like(batch["pixel_values"]))
        out2 = model(other, "joint_cls")
    assert torch.isfinite(out.logits).all() and torch.isfinite(out.loss)
    assert torch.allclose(out.logits[1], out2.logits[1], atol=1e-5)
    assert not torch.allclose(out.logits[0], out2.logits[0])
    expected_in = 2 * model.backbone.dim if pool == "concat_mean_v_t" else model.backbone.dim
    assert model.cls_head[1].weight.shape == (4, expected_in)


def test_all_images_missing_raises(model: MultiImageClassifier) -> None:
    batch = _batch()
    batch["image_mask"] = torch.zeros(B, M, dtype=torch.bool)
    with pytest.raises(ValueError, match="no valid image"):
        model(batch, "joint_cls")


def test_downstream_cls_preset_no_unused_params(tiny_cfg: ModelCfg) -> None:
    model = build_task_model("classifier", tiny_cfg)
    apply_freeze(model, freeze_paths_for_preset(model, "downstream_cls"))
    assert_no_unused_trainable_params(model, _batch(), "joint_cls")
