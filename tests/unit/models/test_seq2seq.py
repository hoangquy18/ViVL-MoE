from __future__ import annotations

import pytest
import torch

from vivlmoe.config.schema import ModelCfg
from vivlmoe.models.factory import build_task_model
from vivlmoe.models.freezing import (
    apply_freeze,
    assert_no_unused_trainable_params,
    freeze_paths_for_preset,
)
from vivlmoe.models.heads.seq2seq import Seq2SeqVLModel
from vivlmoe.models.outputs import Seq2SeqOutput


def _caption_batch(b: int = 2) -> dict:
    g = torch.Generator().manual_seed(3)
    labels = torch.randint(2, 1000, (b, 6), generator=g)
    labels[0, 4:] = -100
    return {
        "pixel_values": torch.randn(b, 3, 32, 32, generator=g),
        "labels": labels,
        "uids": list(range(b)),
        "refs": [["a"], ["b"]],
        "prompts": ["", ""],
    }


def _vqa_batch(b: int = 2) -> dict:
    batch = _caption_batch(b)
    batch["input_ids"] = torch.randint(3, 1000, (b, 5))
    batch["attention_mask"] = torch.ones(b, 5, dtype=torch.long)
    batch["attention_mask"][1, 3:] = 0
    return batch


@pytest.fixture()
def model(tiny_cfg: ModelCfg) -> Seq2SeqVLModel:
    torch.manual_seed(0)
    return build_task_model("seq2seq", tiny_cfg)


def test_encoder_stack_removed(model: Seq2SeqVLModel) -> None:
    assert len(model.decoder.encoder.block) == 0
    assert not model.decoder.encoder.final_layer_norm.weight.requires_grad
    assert model.decoder.shared.weight.requires_grad  # tied embedding stays trainable
    prefixes = {n.split(".")[0] for n, _ in model.named_parameters()}
    assert prefixes == {"backbone", "bridge", "decoder"}
    assert model.bridge.proj.weight.shape == (32, 32)


def test_forward_with_ignore_index_labels_and_generate(model: Seq2SeqVLModel) -> None:
    batch = _caption_batch()
    out = model(batch, "joint_caption")
    assert isinstance(out, Seq2SeqOutput)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.logits.shape == (2, 6, 1000)
    out.loss.backward()
    model.eval()
    gen = model.generate(batch, num_beams=2, max_new_tokens=5)
    assert gen.dtype == torch.long and gen.shape[0] == 2 and gen.shape[1] <= 6
    assert (gen[:, 0] == 0).all()  # decoder_start_token_id
    gen_vqa = model.generate(_vqa_batch(), num_beams=2, max_new_tokens=3)
    assert gen_vqa.shape[0] == 2


def test_vqa_uses_question_and_caption_does_not(model: Seq2SeqVLModel) -> None:
    model.eval()
    batch = _vqa_batch()
    with torch.no_grad():
        a = model(batch, "joint_vqa").loss
        b = model(dict(batch, input_ids=torch.randint(3, 1000, (2, 5))), "joint_vqa").loss
        c = model(batch, "joint_caption").loss
        d = model(dict(batch, input_ids=torch.randint(3, 1000, (2, 5))), "joint_caption").loss
    assert not torch.allclose(a, b)
    assert torch.allclose(c, d)
    with pytest.raises(ValueError, match="unknown mode"):
        model(batch, "joint_cls")


def test_labels_absent_returns_no_loss(model: Seq2SeqVLModel) -> None:
    batch = _caption_batch()
    del batch["labels"]
    out = model(batch, "joint_caption")
    assert out.loss is None and out.logits is None and out.router_outputs == []


@pytest.mark.parametrize(
    ("preset", "mode", "make_batch"),
    [
        ("downstream_caption", "joint_caption", _caption_batch),
        ("downstream_seq2seq", "joint_vqa", _vqa_batch),
    ],
)
def test_no_unused_trainable_params(tiny_cfg: ModelCfg, preset: str, mode: str, make_batch) -> None:
    model = build_task_model("seq2seq", tiny_cfg)
    apply_freeze(model, freeze_paths_for_preset(model, preset))
    assert not model.backbone.logit_scale.requires_grad
    assert not model.backbone.vision_encoder.trainable
    assert_no_unused_trainable_params(model, make_batch(), mode)


def test_caption_with_seq2seq_preset_reports_text_tower_unused(tiny_cfg: ModelCfg) -> None:
    model = build_task_model("seq2seq", tiny_cfg)
    apply_freeze(model, freeze_paths_for_preset(model, "downstream_seq2seq"))
    with pytest.raises(RuntimeError, match="backbone.text_encoder"):
        assert_no_unused_trainable_params(model, _caption_batch(), "joint_caption")
