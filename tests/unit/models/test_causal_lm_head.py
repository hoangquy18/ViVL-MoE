"""v2 LLM generation head on the tiny Qwen3-Next (hybrid linear / full attention, like Qwen3.5)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from vivlmoe.config.schema import ModelCfg
from vivlmoe.models.factory import build_task_model, load_model_cfg
from vivlmoe.models.freezing import (
    apply_freeze,
    assert_no_unused_trainable_params,
    freeze_paths_for_preset,
)
from vivlmoe.models.heads.causal_lm import CausalLMVLModel, position_ids_from_mask

V2_TINY = Path(__file__).resolve().parents[3] / "configs" / "model" / "v2_tiny.yaml"
EOS, PAD = 1, 0


def _batch(vqa: bool, labels: bool = True) -> dict:
    g = torch.Generator().manual_seed(5)
    batch = {
        "pixel_values": torch.rand(3, 3, 32, 32, generator=g),
        "llm_input_ids": torch.tensor([[5, 6, 7], [8, 9, PAD], [10, PAD, PAD]]),
        "llm_attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]]),
        "uids": [0, 1, 2],
    }
    if labels:
        batch["labels"] = torch.tensor(
            [[11, EOS, -100, -100], [12, 13, 14, EOS], [EOS, -100, -100, -100]]
        )
    if vqa:  # questions of 4 / 2 / 3 valid tokens (tiny "roberta" layout: <s>=0, <pad>=1, </s>=2)
        batch["input_ids"] = torch.tensor([[0, 5, 6, 2], [0, 2, 1, 1], [0, 7, 2, 1]])
        batch["attention_mask"] = (batch["input_ids"] != 1).long()
    return batch


def _row(batch: dict, i: int) -> dict:
    return {k: v[i : i + 1] if torch.is_tensor(v) else v[i : i + 1] for k, v in batch.items()}


@pytest.fixture(scope="module")
def model() -> CausalLMVLModel:
    torch.manual_seed(0)
    m = build_task_model("seq2seq", load_model_cfg(V2_TINY))
    assert isinstance(m, CausalLMVLModel)
    return m.eval()


@pytest.mark.parametrize("vqa", [False, True])
def test_sequences_are_compact_and_left_padded(model: CausalLMVLModel, vqa: bool) -> None:
    batch = _batch(vqa)
    mode = "joint_vqa" if vqa else "joint_caption"
    with torch.no_grad():
        enc, key_mask = model._joint(batch, mode)
        embeds, mask, targets = model._sequences(enc, key_mask, batch, batch["labels"])
    latents = model.cfg.resampler.num_latents
    joint = [latents + (n if vqa else 0) for n in (4, 2, 3)]
    lengths = [j + p + a for j, p, a in zip(joint, (3, 2, 1), (2, 4, 1), strict=True)]
    assert embeds.shape == (3, max(lengths), 32)
    for i, length in enumerate(lengths):
        pad = max(lengths) - length
        assert mask[i].tolist() == [0] * pad + [1] * length  # leading padding only
        assert (embeds[i, :pad] == 0).all()
        answer = batch["labels"][i][batch["labels"][i] != -100]
        assert targets[i, -len(answer) :].tolist() == answer.tolist()
        assert (targets[i, : -len(answer)] == -100).all()
        start = pad + joint[i]  # the prompt follows the joint states
        prompt = batch["llm_input_ids"][i][batch["llm_attention_mask"][i].bool()]
        expected = model.llm.get_input_embeddings()(prompt)
        assert torch.equal(embeds[i, start : start + len(prompt)], expected)
    pos = position_ids_from_mask(mask)
    assert pos[0, -1] == lengths[0] - 1 and pos[2, -lengths[2]] == 0


@pytest.mark.parametrize("vqa", [False, True])
def test_padded_batch_loss_equals_unpadded_samples(model: CausalLMVLModel, vqa: bool) -> None:
    """Left padding is exact through the linear-attention layers: the batch loss (token mean) equals
    the per-sample HF causal-LM losses on unpadded sequences."""
    batch = _batch(vqa)
    mode = "joint_vqa" if vqa else "joint_caption"
    with torch.no_grad():
        loss = model(batch, mode).loss
        total, count = 0.0, 0
        for i in range(3):
            row = _row(batch, i)
            enc, key_mask = model._joint(row, mode)
            embeds, mask, targets = model._sequences(enc, key_mask, row, row["labels"])
            assert bool(mask.all())  # a single sample is never padded
            n = int((targets != -100).sum())
            total += float(model.llm(inputs_embeds=embeds, labels=targets).loss) * n
            count += n
    assert float(loss) == pytest.approx(total / count, abs=1e-5)


@pytest.mark.parametrize("num_beams", [1, 2])
def test_batched_generation_equals_per_sample(model: CausalLMVLModel, num_beams: int) -> None:
    batch = _batch(vqa=True, labels=False)
    kw = dict(num_beams=num_beams, max_new_tokens=5, eos_token_id=EOS, pad_token_id=PAD)
    batched = model.generate(batch, **kw)
    for i in range(3):
        single = model.generate(_row(batch, i), **kw)[0].tolist()
        row = batched[i].tolist()
        assert row[: len(single)] == single and all(t == PAD for t in row[len(single) :])


def test_no_labels_runs_the_joint_pass_only(model: CausalLMVLModel) -> None:
    out = model(_batch(vqa=True, labels=False), "joint_vqa")
    assert out.loss is None and out.logits is None and isinstance(out.router_outputs, list)
    with pytest.raises(ValueError, match="unknown mode"):
        model(_batch(vqa=True), "contrastive")


@pytest.mark.parametrize(
    ("preset", "mode", "vqa"),
    [("downstream_caption", "joint_caption", False), ("downstream_seq2seq", "joint_vqa", True)],
)
def test_no_unused_trainable_params(preset: str, mode: str, vqa: bool) -> None:
    model = build_task_model("seq2seq", load_model_cfg(V2_TINY))
    # what DownstreamTask.always_frozen() adds for a sigmoid backbone
    apply_freeze(model, [*freeze_paths_for_preset(model, preset), "backbone.logit_bias"])
    assert any(p.requires_grad for p in model.llm.parameters())
    assert_no_unused_trainable_params(model, _batch(vqa), mode)


def test_pretrained_llm_is_loaded_by_its_class_in_fp32(tmp_path: Path) -> None:
    tiny = dict(vocab_size=1000, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8)  # fmt: skip
    Qwen3ForCausalLM(Qwen3Config(**tiny)).to(torch.bfloat16).save_pretrained(tmp_path)
    cfg: ModelCfg = load_model_cfg(V2_TINY)
    gen = dataclasses.replace(
        cfg.generation, name=str(tmp_path), pretrained=True, llm_class="Qwen3ForCausalLM"
    )
    model = build_task_model("seq2seq", dataclasses.replace(cfg, generation=gen))
    assert isinstance(model.llm, Qwen3ForCausalLM) and model.llm_provenance == f"hf:{tmp_path}"
    assert {p.dtype for p in model.llm.parameters()} == {torch.float32}
    linear = dataclasses.replace(gen, projector="linear")
    model = build_task_model("seq2seq", dataclasses.replace(cfg, generation=linear))
    assert isinstance(model.projector.net, torch.nn.Linear)
