"""ViT5 generation head (paper Sec. 6.1, Fig. 4a/b): joint ``f_enc`` output -> bridge -> T5 decoder.

The T5 encoder stack is removed (plan decision 19): ``decoder.encoder.block = ModuleList()`` and the
remaining encoder-only parameters are frozen, so the decoder consumes ``encoder_outputs`` produced by
the backbone. Captioning uses image tokens only (``joint_caption``); VQA uses image + question
(``joint_vqa``). Targets carry ``-100`` at padded positions (CE ignore index).
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from vivlmoe.config.schema import ModelCfg
from vivlmoe.models.backbone import ViVLBackbone
from vivlmoe.models.outputs import EncoderOutput, Seq2SeqOutput
from vivlmoe.models.projections import init_module_
from vivlmoe.utils.hf import load_fp32, resolve_model_path

log = logging.getLogger(__name__)

MODES = ("joint_caption", "joint_vqa")


class Bridge(nn.Module):
    """``Linear(d -> d_model)`` with an optional LayerNorm (``generation.bridge_norm``)."""

    def __init__(self, dim: int, d_model: int, norm: bool) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, d_model)
        self.norm = nn.LayerNorm(d_model) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


def _remove_encoder_stack(t5: T5ForConditionalGeneration) -> None:
    """Drop the unused T5 encoder layers and freeze what is left of the encoder (not the shared embedding)."""
    t5.encoder.block = nn.ModuleList()
    keep = {id(p) for p in t5.decoder.parameters()} | {id(p) for p in t5.lm_head.parameters()}
    keep |= {id(p) for p in t5.shared.parameters()}
    for name, p in t5.encoder.named_parameters():
        if id(p) not in keep:
            p.requires_grad_(False)
            log.debug("froze encoder-only T5 parameter %s", name)


class Seq2SeqVLModel(nn.Module):
    def __init__(self, backbone: ViVLBackbone, cfg: ModelCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        gen = cfg.generation
        if gen.pretrained:
            self.decoder = load_fp32(T5ForConditionalGeneration, resolve_model_path(gen.name))
            self.decoder_provenance = f"hf:{gen.name}"
        else:
            if gen.tiny_config is None:
                raise ValueError("model.generation.tiny_config is required when pretrained=false")
            t5_cfg = T5Config(**gen.tiny_config)
            if t5_cfg.decoder_start_token_id is None:
                raise ValueError("model.generation.tiny_config must set decoder_start_token_id")
            self.decoder = T5ForConditionalGeneration(t5_cfg)
            self.decoder_provenance = "random"
        if self.decoder.config.decoder_start_token_id is None:
            raise ValueError("T5 config has no decoder_start_token_id; generation would fail")
        _remove_encoder_stack(self.decoder)
        if gen.gradient_checkpointing:
            self.decoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.bridge = Bridge(backbone.dim, int(self.decoder.config.d_model), gen.bridge_norm)
        init_module_(self.bridge, cfg.init_std)

    # ---------------------------------------------------------------- encoding
    def _encode(self, batch: dict, mode: str) -> tuple[EncoderOutput, torch.Tensor, torch.Tensor]:
        """Joint pass -> ``(EncoderOutput, bridged states [B,S,d_model], attention_mask [B,S] long)``."""
        if mode not in MODES:
            raise ValueError(f"Seq2SeqVLModel: unknown mode {mode!r}; expected one of {MODES}")
        image_tokens = self.backbone.encode_image_tokens(batch["pixel_values"])
        if mode == "joint_vqa":
            text_tokens = self.backbone.encode_text_tokens(
                batch["input_ids"], batch["attention_mask"]
            )
            text_mask = batch["attention_mask"].bool()
        else:
            text_tokens, text_mask = None, None
        enc, key_mask, _ = self.backbone.forward_joint(image_tokens, None, text_tokens, text_mask)
        return enc, self.bridge(enc.last_hidden_state), key_mask.long()

    @staticmethod
    def _infer_mode(batch: dict) -> str:
        return "joint_vqa" if "input_ids" in batch else "joint_caption"

    # ---------------------------------------------------------------- task API
    def forward(self, batch: dict, mode: str) -> Seq2SeqOutput:
        enc, states, attn = self._encode(batch, mode)
        labels = batch.get("labels")
        if labels is None:  # eval batches: routing probes only need the encoder pass
            return Seq2SeqOutput(loss=None, logits=None, router_outputs=enc.router_outputs)
        out = self.decoder(
            encoder_outputs=BaseModelOutput(last_hidden_state=states),
            attention_mask=attn,
            labels=labels,
        )
        return Seq2SeqOutput(loss=out.loss, logits=out.logits, router_outputs=enc.router_outputs)

    @torch.no_grad()
    def generate(self, batch: dict, mode: str | None = None, **generate_kwargs) -> torch.LongTensor:
        """Beam/greedy decoding with HF ``generate`` on the bridged encoder states -> ``[B, L]`` ids."""
        _, states, attn = self._encode(batch, mode or self._infer_mode(batch))
        return self.decoder.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=states),
            attention_mask=attn,
            **generate_kwargs,
        )
