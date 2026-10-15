"""Immutable output containers shared by models, losses, tasks and analysis.

No module caches routing state: every MoE layer returns a :class:`RouterOutput` per call and the
losses / probes consume the list. Modality ids: 0 = visual, 1 = text (everywhere).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

VISUAL, TEXT = 0, 1


@dataclass
class RouterOutput:
    """Routing decisions of one MoE layer for the T valid (non-padded) tokens of one encoder call."""

    layer_idx: int
    logits: torch.Tensor  # [T, N_r] fp32, raw router logits z(x) (differentiable)
    topk_idx: torch.Tensor  # [T, k] long, selected experts S(x) = Top-k(z + b)
    topk_probs: (
        torch.Tensor
    )  # [T, k] fp32, p~ renormalised over S(x) from z (Eq. 15), differentiable
    modality_ids: torch.Tensor  # [T] long, 0 = visual, 1 = text

    @property
    def num_experts(self) -> int:
        return int(self.logits.shape[-1])

    @property
    def top_k(self) -> int:
        return int(self.topk_idx.shape[-1])

    @property
    def num_tokens(self) -> int:
        return int(self.logits.shape[0])

    def detach(self) -> RouterOutput:
        return RouterOutput(
            self.layer_idx,
            self.logits.detach(),
            self.topk_idx.detach(),
            self.topk_probs.detach(),
            self.modality_ids.detach(),
        )


@dataclass
class EncoderOutput:
    last_hidden_state: torch.Tensor  # [B, S, d]
    router_outputs: list[RouterOutput] = field(
        default_factory=list
    )  # one per MoE layer ([] for dense)


@dataclass
class ContrastiveOutput:
    image_embeds: torch.Tensor  # [B, d] L2-normalised fp32
    text_embeds: torch.Tensor  # [B, d] L2-normalised fp32
    logit_scale: torch.Tensor  # scalar, already exp() and clamped
    router_outputs: list[RouterOutput] = field(
        default_factory=list
    )  # visual-pass + text-pass outputs
    logit_bias: torch.Tensor | None = None  # sigmoid head only (model.contrastive_head)


@dataclass
class TeacherStudentOutput:
    student: torch.Tensor  # [B, P] fp32 z_vi = FC1(XLM-R <s>)
    teacher: torch.Tensor | None = None  # [B, P] fp32 z_en (filled by the task, no grad)


@dataclass
class Seq2SeqOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor | None  # [B, L, V]
    router_outputs: list[RouterOutput] = field(default_factory=list)


@dataclass
class ClassifierOutput:
    logits: torch.Tensor  # [B, A, C]
    loss: torch.Tensor | None
    router_outputs: list[RouterOutput] = field(default_factory=list)


ModelOutput = ContrastiveOutput | TeacherStudentOutput | Seq2SeqOutput | ClassifierOutput
