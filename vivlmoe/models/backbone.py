"""ViVL backbone (paper Sec. 4.2, Fig. 3): towers -> projections -> resampler -> shared encoder ``f_enc``.

Modes (plan decision 2 - one ``forward(batch, mode)`` per micro-batch):

* ``contrastive`` -> :class:`ContrastiveOutput`: two *separate* single-modality passes through
  ``f_enc`` (no cross-modal leakage), masked mean pooling, fp32 L2-normalised embeddings and the
  clamped temperature ``exp(min(logit_scale, ln logit_scale_max))`` (Eq. 13).
* ``encode_image`` / ``encode_text`` -> ``{"embeds": [B,d]}`` for retrieval evaluation.

Downstream heads use :meth:`forward_joint` (``X = [V~ + e_v ; T~ + e_t]``, Sec. 4.2 joint mode).
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from torch import nn

from vivlmoe.config.schema import ModelCfg
from vivlmoe.models.encoder import MultimodalEncoder
from vivlmoe.models.outputs import TEXT, VISUAL, ContrastiveOutput, EncoderOutput
from vivlmoe.models.projections import (
    ModalityEmbedding,
    TextProjection,
    VisionProjection,
    init_module_,
)
from vivlmoe.models.resampler import PerceiverResampler
from vivlmoe.models.towers import build_vision_tower
from vivlmoe.models.towers.text import TextTower

log = logging.getLogger(__name__)

NEW_MODULES = ("vision_proj", "text_proj", "resampler", "modality_embed", "encoder")


class ViVLBackbone(nn.Module):
    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        self.cfg = cfg
        dim = cfg.encoder.dim
        self.vision_encoder = build_vision_tower(cfg.vision)
        self.vision_proj = VisionProjection(self.vision_encoder.hidden_size, dim)
        self.resampler = PerceiverResampler(dim, cfg.resampler)
        self.text_encoder = TextTower(cfg.text)
        self.text_proj = TextProjection(self.text_encoder.hidden_size, dim)
        self.modality_embed = ModalityEmbedding(dim)
        self.encoder = MultimodalEncoder(cfg.encoder, cfg.moe)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / cfg.logit_scale_init)))
        self.init_new_weights()
        log.info(
            "ViVLBackbone: vision tokens %d x %d -> W_v %s -> %d latents; text d_t=%d; f_enc %d x dim %d (moe=%s)",
            self.vision_encoder.num_tokens,
            self.vision_encoder.hidden_size,
            tuple(self.vision_proj.weight.shape),
            cfg.resampler.num_latents,
            self.text_encoder.hidden_size,
            cfg.encoder.num_layers,
            dim,
            cfg.moe is not None,
        )

    @classmethod
    def from_config(cls, cfg: ModelCfg) -> ViVLBackbone:
        return cls(cfg)

    @torch.no_grad()
    def init_new_weights(self) -> None:
        """Normal(0, init_std) for every random-init module (Sec. 4.2); towers are left untouched.

        MoE routers (``encoder.layers[i].ffn.router``, milestone M6) own their initialisation
        (``MoECfg.router_init_std``) and are skipped here.
        """
        std = self.cfg.init_std
        for name in NEW_MODULES:
            init_module_(getattr(self, name), std, skip=("router",))
        self.resampler.latents.normal_(0.0, std)

    @property
    def dim(self) -> int:
        return self.cfg.encoder.dim

    @property
    def num_visual_tokens(self) -> int:
        return self.cfg.resampler.num_latents

    # ---------------------------------------------------------------- encoders
    def encode_image_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``V~ = R_v(W_v f_v(I))`` -> ``[B, num_latents, d]`` (no modality embedding)."""
        return self.resampler(self.vision_proj(self.vision_encoder(pixel_values)))

    def encode_text_tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """``T~ = W_t f_t(c)`` -> ``[B, m, d]`` (no modality embedding)."""
        return self.text_proj(self.text_encoder(input_ids, attention_mask))

    def _modality_ids(self, b: int, s: int, modality: int, device: torch.device) -> torch.Tensor:
        return torch.full((b, s), modality, dtype=torch.long, device=device)

    def forward_single(
        self, tokens: torch.Tensor, mask: torch.Tensor | None, modality: int
    ) -> EncoderOutput:
        """One single-modality pass through ``f_enc`` (contrastive mode)."""
        if modality not in (VISUAL, TEXT):
            raise ValueError(f"modality must be {VISUAL} (visual) or {TEXT} (text), got {modality}")
        if self.cfg.modality_embed_in_single_pass:
            tokens = tokens + self.modality_embed.weight[modality].to(tokens.dtype)
        mids = self._modality_ids(tokens.shape[0], tokens.shape[1], modality, tokens.device)
        return self.encoder(tokens, mask, mids)

    def forward_joint(
        self,
        image_tokens: torch.Tensor,
        image_mask: torch.Tensor | None,
        text_tokens: torch.Tensor | None,
        text_mask: torch.Tensor | None,
    ) -> tuple[EncoderOutput, torch.Tensor, torch.Tensor]:
        """Joint pass over ``[V~ + e_v ; T~ + e_t]``.

        ``image_mask [B,Ni]`` bool (None = all valid); ``text_tokens`` may be None (captioning).
        Returns ``(EncoderOutput, key_mask [B,S] bool, modality_ids [B,S])``.
        """
        b, n_i, _ = image_tokens.shape
        device = image_tokens.device
        e_v = self.modality_embed.weight[VISUAL].to(image_tokens.dtype)
        parts = [image_tokens + e_v]
        masks = [
            image_mask.bool()
            if image_mask is not None
            else torch.ones(b, n_i, dtype=torch.bool, device=device)
        ]
        mids = [self._modality_ids(b, n_i, VISUAL, device)]
        if text_tokens is not None:
            if text_mask is None:
                raise ValueError("text_mask is required when text_tokens are given")
            n_t = text_tokens.shape[1]
            parts.append(text_tokens + self.modality_embed.weight[TEXT].to(text_tokens.dtype))
            masks.append(text_mask.bool())
            mids.append(self._modality_ids(b, n_t, TEXT, device))
        x = torch.cat(parts, dim=1)
        key_mask = torch.cat(masks, dim=1)
        modality_ids = torch.cat(mids, dim=1)
        return self.encoder(x, key_mask, modality_ids), key_mask, modality_ids

    @staticmethod
    def masked_mean(h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """fp32 mean over valid positions (``mask [B,S]`` True=keep); all-masked rows give zeros."""
        h = h.float()
        if mask is None:
            return h.mean(dim=1)
        m = mask.to(h.dtype).unsqueeze(-1)
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def scaled_logit_scale(self) -> torch.Tensor:
        """``exp(min(logit_scale, ln logit_scale_max))`` (plan decision 9)."""
        return self.logit_scale.clamp(max=math.log(self.cfg.logit_scale_max)).exp()

    # ---------------------------------------------------------------- task modes
    def _image_embeds(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, EncoderOutput]:
        out = self.forward_single(self.encode_image_tokens(pixel_values), None, VISUAL)
        return F.normalize(self.masked_mean(out.last_hidden_state, None), dim=-1), out

    def _text_embeds(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, EncoderOutput]:
        mask = attention_mask.bool()
        out = self.forward_single(self.encode_text_tokens(input_ids, attention_mask), mask, TEXT)
        return F.normalize(self.masked_mean(out.last_hidden_state, mask), dim=-1), out

    def forward(
        self, batch: dict, mode: str = "contrastive"
    ) -> ContrastiveOutput | dict[str, torch.Tensor]:
        if mode == "contrastive":
            img, out_v = self._image_embeds(batch["pixel_values"])
            txt, out_t = self._text_embeds(batch["input_ids"], batch["attention_mask"])
            return ContrastiveOutput(
                image_embeds=img,
                text_embeds=txt,
                logit_scale=self.scaled_logit_scale(),
                router_outputs=out_v.router_outputs + out_t.router_outputs,
            )
        if mode == "encode_image":
            return {"embeds": self._image_embeds(batch["pixel_values"])[0]}
        if mode == "encode_text":
            return {"embeds": self._text_embeds(batch["input_ids"], batch["attention_mask"])[0]}
        raise ValueError(
            f"ViVLBackbone: unknown mode {mode!r} (contrastive|encode_image|encode_text)"
        )
