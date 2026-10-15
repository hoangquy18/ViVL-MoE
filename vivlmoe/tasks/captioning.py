"""Image captioning (KTVIC, UIT-ViIC; paper Sec. 6.1, Fig. 4a): image tokens only -> ViT5 decoder
(v2: the decoder-only LLM head, ``model.generation.kind: causal_lm``).

Training examples are every (image, caption) annotation (5 per KTVIC image) with the caption as the
decoder target (CE with ``-100`` pads); evaluation decodes one caption per unique image with beam
search and scores it against all references (multi-ref BLEU-4, ROUGE-L, CIDEr). Since no text
tokens enter the joint pass, the text tower / ``W_t`` are frozen (preset ``downstream_caption``) and
the modality-importance loss is identically 0 (decision 19).
"""

from __future__ import annotations

import logging
from typing import Any

from torch import nn
from torch.utils.data import DataLoader

from vivlmoe.data.builders import build_caption_eval, build_image_text_train
from vivlmoe.data.collators import Seq2SeqCollator
from vivlmoe.tasks.downstream_common import DownstreamTask
from vivlmoe.tasks.evaluators.base import Evaluator
from vivlmoe.tasks.evaluators.generation import (
    GENERATION_METRICS,
    GenerationEvaluator,
    validate_metric_names,
)
from vivlmoe.tasks.registry import TASKS
from vivlmoe.training.state import StepContext, StepOutput

log = logging.getLogger(__name__)


@TASKS.register("captioning")
class CaptioningTask(DownstreamTask):
    name = "captioning"
    model_kind = "seq2seq"
    forward_mode = "joint_caption"
    freeze_preset = "downstream_caption"

    def _collator(self, train: bool) -> Seq2SeqCollator:
        return self.generation_collator(train, with_text=False)

    def build_train_loader(self) -> DataLoader:
        _, train_tf, _ = self.transforms
        ds, _groups = build_image_text_train(
            self.train_source(),
            train_tf,
            self.draft_size,
            run_dir=self.run_dir,
            write_dropped=self.ctx.is_main,
        )
        # every annotation is an example (groups=None -> singleton groups), unlike the contrastive
        # one-caption-per-image sampler
        return self.make_train_loader(ds, self._collator(train=True), groups=None)

    def build_eval_loaders(self) -> dict[str, DataLoader]:
        _, _, eval_tf = self.transforms
        loaders: dict[str, DataLoader] = {}
        for split, src in self.cfg.data.evals.items():
            ds = build_caption_eval(
                src, eval_tf, self.draft_size, run_dir=self.run_dir, write_dropped=self.ctx.is_main
            )
            loaders[split] = self.make_eval_loader(ds, self._collator(train=False))
            log.info("caption eval split %s: %d images", split, len(ds))
        return loaders

    def build_evaluators(self) -> dict[str, Evaluator]:
        validate_metric_names(self.cfg.task.metrics, GENERATION_METRICS, "captioning")
        return {
            split: GenerationEvaluator(
                self.ctx,
                self.cfg.task,
                self.tokenizers.require_decoder(),
                mode=self.forward_mode,
                amp_dtype=self.amp_dtype,
                pass_special_ids=self.causal_lm,
            )
            for split in self.cfg.data.evals
        }

    def train_step(self, model: nn.Module, batch: dict[str, Any], sc: StepContext) -> StepOutput:
        out = model(batch, self.forward_mode)  # the single forward per micro-batch
        if out.loss is None:
            raise RuntimeError("captioning train batch has no labels")
        return self.finish_train_step(
            model, out.loss, out.router_outputs, batch, self.count_label_tokens(batch)
        )

    def selection_metric(self) -> tuple[str | None, str]:
        if not self.cfg.data.evals:
            return None, "max"
        metric = self.cfg.task.metrics[0] if self.cfg.task.metrics else "bleu4"
        return f"{next(iter(self.cfg.data.evals))}/{metric}", "max"


__all__ = ["CaptioningTask"]
