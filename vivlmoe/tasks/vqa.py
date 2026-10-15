"""Visual question answering (OpenViVQA, ViTextVQA, ViOCR-VQA; paper Sec. 6.1, Fig. 4b).

Joint pass over ``[image tokens ; question tokens]`` -> ViT5 decoder generating the answer (v2: the
decoder-only LLM head, ``model.generation.kind: causal_lm``, with ``task.llm_prompt_template``).
``task.train_on_all_answers`` expands every gold answer into a training example; evaluation keeps
one item per question with all gold answers as references (ROUGE-L / CIDEr for OpenViVQA,
token-F1 / EM for the short-answer sets).
"""

from __future__ import annotations

import logging
from typing import Any

from torch import nn
from torch.utils.data import DataLoader

from vivlmoe.data.builders import build_vqa
from vivlmoe.data.collators import Seq2SeqCollator
from vivlmoe.tasks.downstream_common import DownstreamTask
from vivlmoe.tasks.evaluators.base import Evaluator
from vivlmoe.tasks.evaluators.generation import GenerationEvaluator, validate_metric_names
from vivlmoe.tasks.registry import TASKS
from vivlmoe.training.state import StepContext, StepOutput

log = logging.getLogger(__name__)

VQA_METRICS = ("rougeL", "cider", "f1", "em")
DEFAULT_TEMPLATE = "{question}"


@TASKS.register("vqa")
class VQATask(DownstreamTask):
    name = "vqa"
    model_kind = "seq2seq"
    forward_mode = "joint_vqa"
    freeze_preset = "downstream_seq2seq"

    def resolve_template(self, adapter_template: str) -> str:
        """``task.prompt_template`` wins when set explicitly; otherwise the adapter's (``vqa_generic`` option)."""
        task_template = self.cfg.task.prompt_template
        if task_template != DEFAULT_TEMPLATE:
            if adapter_template != DEFAULT_TEMPLATE and adapter_template != task_template:
                log.warning(
                    "task.prompt_template %r overrides the adapter's %r",
                    task_template,
                    adapter_template,
                )
            return task_template
        return adapter_template

    def _collator(self, train: bool, template: str) -> Seq2SeqCollator:
        return self.generation_collator(train, with_text=True, template=template)

    def build_train_loader(self) -> DataLoader:
        _, train_tf, _ = self.transforms
        ds, template = build_vqa(
            self.train_source(),
            train_tf,
            self.draft_size,
            run_dir=self.run_dir,
            write_dropped=self.ctx.is_main,
            expand_answers=self.cfg.task.train_on_all_answers,
        )
        log.info(
            "vqa train: %d examples (train_on_all_answers=%s)",
            len(ds),
            self.cfg.task.train_on_all_answers,
        )
        return self.make_train_loader(
            ds, self._collator(True, self.resolve_template(template)), groups=None
        )

    def build_eval_loaders(self) -> dict[str, DataLoader]:
        _, _, eval_tf = self.transforms
        loaders: dict[str, DataLoader] = {}
        for split, src in self.cfg.data.evals.items():
            ds, template = build_vqa(
                src,
                eval_tf,
                self.draft_size,
                run_dir=self.run_dir,
                write_dropped=self.ctx.is_main,
                expand_answers=False,
            )
            loaders[split] = self.make_eval_loader(
                ds, self._collator(False, self.resolve_template(template))
            )
            log.info("vqa eval split %s: %d questions", split, len(ds))
        return loaders

    def build_evaluators(self) -> dict[str, Evaluator]:
        validate_metric_names(self.cfg.task.metrics, VQA_METRICS, "vqa")
        return {
            split: GenerationEvaluator(
                self.ctx,
                self.cfg.task,
                self.tokenizers.require_decoder(),
                mode=self.forward_mode,
                amp_dtype=self.amp_dtype,
                allowed_metrics=VQA_METRICS,
                pass_special_ids=self.causal_lm,
            )
            for split in self.cfg.data.evals
        }

    def train_step(self, model: nn.Module, batch: dict[str, Any], sc: StepContext) -> StepOutput:
        out = model(batch, self.forward_mode)
        if out.loss is None:
            raise RuntimeError("vqa train batch has no labels")
        return self.finish_train_step(
            model, out.loss, out.router_outputs, batch, self.count_label_tokens(batch)
        )


__all__ = ["VQA_METRICS", "VQATask"]
