"""v2 generation head: the joint ``f_enc`` states prefix a decoder-only LLM (``generation.kind: causal_lm``).

Every sample becomes one compact sequence ``[joint states ; prompt ; answer + EOS]``. The joint states
are the projected ``f_enc`` outputs at valid positions (the visual latents, then the valid question
tokens for VQA). The prompt is ``task.llm_prompt_template`` in the LLM vocabulary, and the answer is
present only when training.

Sequences are **left-padded** to the batch maximum with ``position_ids = cumsum(mask) - 1``. The
linear-attention layers of Qwen3.5 (Gated DeltaNet) zero padded positions, which is exact only for
leading padding (no hole inside a sequence). HF ``generate`` derives the same positions from the
attention mask, so training and decoding see identical inputs.

Labels are built here (``-100`` everywhere but the answer). ``lm_head`` runs only at the positions that
predict an answer token (Qwen3.5 has a 248K vocabulary), so no logits are returned. Without ``labels``
(eval batches) only the joint pass runs, which is what routing probes need.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

from vivlmoe.config.schema import ModelCfg
from vivlmoe.models.backbone import ViVLBackbone
from vivlmoe.models.heads.llm_loader import load_causal_lm
from vivlmoe.models.heads.seq2seq import MODES
from vivlmoe.models.outputs import EncoderOutput, Seq2SeqOutput
from vivlmoe.models.projections import init_module_

log = logging.getLogger(__name__)


class Projector(nn.Module):
    """``f_enc`` width -> LLM embedding width: ``linear``, or ``mlp2`` (Linear-GELU-Linear, LLaVA-1.5)."""

    def __init__(self, dim: int, hidden: int, kind: str) -> None:
        super().__init__()
        if kind == "linear":
            self.net: nn.Module = nn.Linear(dim, hidden)
        elif kind == "mlp2":
            self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        else:
            raise ValueError(f"unknown projector {kind!r}; expected linear | mlp2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def position_ids_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """``cumsum(mask) - 1`` with pads set to 1, exactly as HF ``generate`` derives them."""
    return (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)


class CausalLMVLModel(nn.Module):
    def __init__(self, backbone: ViVLBackbone, cfg: ModelCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        gen = cfg.generation
        if gen.pretrained:
            self.llm = load_causal_lm(gen.name, llm_class=gen.llm_class)
            self.llm_provenance = f"hf:{gen.name}"
        else:
            if gen.tiny_config is None:
                raise ValueError("model.generation.tiny_config is required when pretrained=false")
            self.llm = load_causal_lm(None, tiny_config=gen.tiny_config, tiny_arch=gen.tiny_arch)
            self.llm_provenance = "random"
        if gen.gradient_checkpointing:
            self.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        hidden = int(self.llm.get_input_embeddings().embedding_dim)
        self.projector = Projector(backbone.dim, hidden, gen.projector)
        init_module_(self.projector, cfg.init_std)
        log.info(
            "CausalLMVLModel: %s (%s), projector %s %d -> %d",
            type(self.llm).__name__,
            self.llm_provenance,
            gen.projector,
            backbone.dim,
            hidden,
        )

    # ---------------------------------------------------------------- sequences
    def _joint(self, batch: dict, mode: str) -> tuple[EncoderOutput, torch.Tensor]:
        """Joint pass -> ``(EncoderOutput, key_mask [B,S] bool)`` (latents first, then the question)."""
        if mode not in MODES:
            raise ValueError(f"CausalLMVLModel: unknown mode {mode!r}; expected one of {MODES}")
        image_tokens = self.backbone.encode_image_tokens(batch["pixel_values"])
        if mode == "joint_vqa":
            text_tokens = self.backbone.encode_text_tokens(
                batch["input_ids"], batch["attention_mask"]
            )
            text_mask = batch["attention_mask"].bool()
        else:
            text_tokens, text_mask = None, None
        enc, key_mask, _ = self.backbone.forward_joint(image_tokens, None, text_tokens, text_mask)
        return enc, key_mask

    def _sequences(
        self,
        enc: EncoderOutput,
        key_mask: torch.Tensor,
        batch: dict,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Left-padded ``[joint ; prompt ; answer]`` -> ``(inputs_embeds [B,T,H], attention_mask
        [B,T] long, targets [B,T])`` with the answer ids in ``targets`` and ``-100`` elsewhere."""
        embed = self.llm.get_input_embeddings()
        dtype = embed.weight.dtype
        joint = self.projector(enc.last_hidden_state[key_mask]).to(dtype)  # valid positions only
        prompt_mask = batch["llm_attention_mask"].bool()
        prompt_ids = batch["llm_input_ids"][prompt_mask]
        if labels is None:
            answer_mask = torch.zeros_like(prompt_mask[:, :0])
        else:
            answer_mask = labels != -100
        answer_ids = labels[answer_mask] if labels is not None else prompt_ids[:0]
        counts = [m.sum(1).tolist() for m in (key_mask, prompt_mask, answer_mask)]
        parts = zip(
            joint.split(counts[0]),
            embed(prompt_ids).split(counts[1]),
            embed(answer_ids).split(counts[2]),
            answer_ids.split(counts[2]),
            strict=True,
        )
        length = max(sum(c) for c in zip(*counts, strict=True))
        rows, masks, targets = [], [], []
        for j, p, a, a_ids in parts:
            pad = length - len(j) - len(p) - len(a)
            rows.append(F.pad(torch.cat([j, p, a]), (0, 0, pad, 0)))
            masks.append(F.pad(a_ids.new_ones(length - pad), (pad, 0)))
            targets.append(F.pad(a_ids, (length - len(a_ids), 0), value=-100))
        return torch.stack(rows), torch.stack(masks), torch.stack(targets)

    @staticmethod
    def _infer_mode(batch: dict) -> str:
        return "joint_vqa" if "input_ids" in batch else "joint_caption"

    # ---------------------------------------------------------------- task API
    def forward(self, batch: dict, mode: str) -> Seq2SeqOutput:
        enc, key_mask = self._joint(batch, mode)
        labels = batch.get("labels")
        if labels is None:  # eval batches: routing probes only need the joint pass
            return Seq2SeqOutput(loss=None, logits=None, router_outputs=enc.router_outputs)
        embeds, mask, targets = self._sequences(enc, key_mask, batch, labels)
        hidden = self.llm.base_model(
            inputs_embeds=embeds,
            attention_mask=mask,
            position_ids=position_ids_from_mask(mask),
            use_cache=False,
        ).last_hidden_state
        predicts = targets[:, 1:] != -100  # position t predicts token t + 1
        logits = self.llm.get_output_embeddings()(hidden[:, :-1][predicts])
        loss = F.cross_entropy(logits.float(), targets[:, 1:][predicts])
        return Seq2SeqOutput(loss=loss, logits=None, router_outputs=enc.router_outputs)

    @torch.no_grad()
    def generate(self, batch: dict, mode: str | None = None, **generate_kwargs) -> torch.LongTensor:
        """HF ``generate`` from the left-padded ``[joint ; prompt]`` embeddings -> ``[B, L]`` new ids.

        Pass the LLM tokenizer's ``eos_token_id`` / ``pad_token_id`` (the EOS the answers were
        trained with); the evaluator does this for causal-LM heads.
        """
        enc, key_mask = self._joint(batch, mode or self._infer_mode(batch))
        embeds, mask, _ = self._sequences(enc, key_mask, batch, None)
        return self.llm.generate(inputs_embeds=embeds, attention_mask=mask, **generate_kwargs)


__all__ = ["CausalLMVLModel", "Projector", "position_ids_from_mask"]
