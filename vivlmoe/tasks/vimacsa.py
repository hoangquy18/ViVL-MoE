"""ViMACSA multimodal aspect-category sentiment analysis (paper Sec. 6.1, Fig. 4c; decision 18).

Legacy protocol: each review yields one joint sequence per aspect (``"<aspect> </s></s> <review>"``
+ optional ResNet-152 image aspect tags), up to ``task.max_images`` images encoded once per sample,
4 labels per aspect (None / Negative / Neutral / Positive). Loss = sum over aspects of the
batch-mean CE (computed by the classifier) + router aux; metric = macro-F1 per aspect then mean.
"""

from __future__ import annotations

import logging
from typing import Any

from torch import nn
from torch.utils.data import DataLoader

from vivlmoe.config.schema import ConfigError, DataSourceCfg
from vivlmoe.data.builders import build_vimacsa
from vivlmoe.data.collators import MultiImageClsCollator
from vivlmoe.tasks.downstream_common import DownstreamTask
from vivlmoe.tasks.evaluators.base import Evaluator
from vivlmoe.tasks.evaluators.classification import ClassificationEvaluator
from vivlmoe.tasks.registry import TASKS
from vivlmoe.training.state import StepContext, StepOutput

log = logging.getLogger(__name__)

VIMACSA_METRICS = ("macro_f1",)


@TASKS.register("vimacsa")
class ViMACSATask(DownstreamTask):
    name = "vimacsa"
    model_kind = "classifier"
    forward_mode = "joint_cls"
    freeze_preset = "downstream_cls"
    needs_decoder_tokenizer = False

    def _validate(self) -> None:
        t = self.cfg.task
        bad = [m for m in t.metrics if m not in VIMACSA_METRICS]
        if bad or not t.metrics:
            raise ConfigError(
                f"vimacsa: task.metrics must be {list(VIMACSA_METRICS)}, got {t.metrics}"
            )
        if self.cfg.model.classifier.num_aspects != len(t.aspects):
            raise ConfigError("model.classifier.num_aspects must equal len(task.aspects)")

    def _collator(self) -> MultiImageClsCollator:
        t = self.cfg.task
        return MultiImageClsCollator(
            self.tokenizers.vi,
            t.aspects,
            use_image_tags=t.use_image_tags,
            cls_max_len=self.cfg.data.tokenizer.cls_max_len,
            legacy_parity=t.legacy_parity,
        )

    def _dataset(self, src: DataSourceCfg, train: bool) -> Any:
        spec, train_tf, eval_tf = self.transforms
        return build_vimacsa(
            src,
            train_tf if train else eval_tf,
            image_hw=(spec.crop_size, spec.crop_size),
            max_images=self.cfg.task.max_images,
            draft_size=self.draft_size,
            run_dir=self.run_dir,
            write_dropped=self.ctx.is_main,
        )

    def build_train_loader(self) -> DataLoader:
        self._validate()
        ds = self._dataset(self.train_source(), train=True)
        return self.make_train_loader(ds, self._collator(), groups=None)

    def build_eval_loaders(self) -> dict[str, DataLoader]:
        self._validate()
        loaders: dict[str, DataLoader] = {}
        for split, src in self.cfg.data.evals.items():
            ds = self._dataset(src, train=False)
            loaders[split] = self.make_eval_loader(ds, self._collator())
            log.info("vimacsa eval split %s: %d samples", split, len(ds))
        return loaders

    def build_evaluators(self) -> dict[str, Evaluator]:
        self._validate()
        t = self.cfg.task
        return {
            split: ClassificationEvaluator(
                self.ctx,
                t.aspects,
                exclude_none=t.exclude_none_in_f1,
                mode=self.forward_mode,
                num_labels=self.cfg.model.classifier.num_labels,
                amp_dtype=self.amp_dtype,
            )
            for split in self.cfg.data.evals
        }

    def train_step(self, model: nn.Module, batch: dict[str, Any], sc: StepContext) -> StepOutput:
        out = model(batch, self.forward_mode)
        if out.loss is None:
            raise RuntimeError("vimacsa train batch has no labels")
        acc = float((out.logits.argmax(-1) == batch["labels"]).float().mean())
        return self.finish_train_step(
            model,
            out.loss,
            out.router_outputs,
            batch,
            num_tokens=int(batch["attention_mask"].sum()),
            metrics={"acc": acc},
        )

    def selection_metric(self) -> tuple[str | None, str]:
        if not self.cfg.data.evals:
            return None, "max"
        return f"{next(iter(self.cfg.data.evals))}/macro_f1", "max"


__all__ = ["VIMACSA_METRICS", "ViMACSATask"]
