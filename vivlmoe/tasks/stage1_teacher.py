"""Stage 1 - teacher learning (paper Sec. 4.1, Eq. 12): align XLM-R with the frozen CLIP text encoder.

``L_1 = (1/N) sum_i ||z_vi - z_en||^2`` with ``z_en = CLIP_text(en)`` (frozen teacher, owned by
the task - plan decision 11: no DDP / checkpoint bookkeeping for it) and
``z_vi = FC1(XLM-R(vi)[<s>])`` (:class:`~vivlmoe.models.stage1.Stage1Student`). Training data is
the PhoMT parallel corpus; evaluation is the teacher MSE on the PhoMT dev split and, with
``task.eval_clip_retrieval``, R@K in the CLIP joint space on image-text dev splits (Appendix A)
using cached CLIP image embeddings.

v2 reuses this task to distil mmBERT from the SigLIP 2 text tower (``model.stage1.teacher_kind:
siglip``): the teacher reads captions in either language (``caption_text`` sources, src == tgt) or
PhoMT English, the student the Vietnamese side, and dev retrieval runs in the SigLIP joint space
with the teacher's own image tower and preprocessing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from vivlmoe.config.schema import ConfigError, DataSourceCfg, RunConfig
from vivlmoe.data.builders import (
    build_parallel_text,
    build_records,
    build_transform_for,
    draft_size_for,
)
from vivlmoe.data.collators import RetrievalTextCollator, TeacherCollator
from vivlmoe.data.datasets import ParallelTextDataset, RetrievalEvalDataset
from vivlmoe.data.loaders import build_eval_loader, build_train_loader
from vivlmoe.data.records import ImageTextRecord, ParallelTextRecord
from vivlmoe.data.tokenization import TokenizerBundle, build_tokenizers
from vivlmoe.data.transforms import build_transform, square_spec
from vivlmoe.losses.teacher import TeacherMSELoss
from vivlmoe.models.towers.teachers import teacher_output_dim, teacher_vision_cfg
from vivlmoe.tasks.base import LoaderBundle, Task
from vivlmoe.tasks.evaluators.base import Evaluator
from vivlmoe.tasks.evaluators.retrieval import (
    ClipSpaceRetrievalEvaluator,
    RetrievalLoaders,
    build_retrieval_loaders,
)
from vivlmoe.tasks.evaluators.teacher_mse import TeacherMSEEvaluator
from vivlmoe.tasks.registry import TASKS
from vivlmoe.training.distributed import DistContext
from vivlmoe.training.state import StepContext, StepOutput

log = logging.getLogger(__name__)


class StudentTextCollator:
    """Retrieval text side for the Stage-1 student: ``input_ids`` -> ``vi_input_ids`` (+ mask)."""

    def __init__(self, tokenizer: Any, max_len: int) -> None:
        self._inner = RetrievalTextCollator(tokenizer, max_len)

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        b = self._inner(items)
        b["vi_input_ids"] = b.pop("input_ids")
        b["vi_attention_mask"] = b.pop("attention_mask")
        return b


def _student_embeds(model: nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    return F.normalize(model(batch, "stage1").student, dim=-1)


@TASKS.register("stage1_teacher")
class TeacherStudentTask(Task):
    name = "stage1_teacher"

    def __init__(self, cfg: RunConfig, ctx: DistContext) -> None:
        super().__init__(cfg, ctx)
        self.loss = TeacherMSELoss(cfg.model.stage1.loss_preset)
        self._teacher: nn.Module | None = None
        self._image_teacher: nn.Module | None = None
        self._tokenizers: TokenizerBundle | None = None
        self._split_kinds: dict[str, str] = {}  # eval split -> 'parallel' | 'retrieval'
        self._dev_split_sha1: dict[str, str | None] = {}
        self._transform_spec: dict[str, Any] | None = None
        self._evaluators: dict[str, Evaluator] | None = None

    # ------------------------------------------------------------------ model / teachers
    def build_model(self) -> nn.Module:
        from vivlmoe.models.factory import build_task_model

        return build_task_model("stage1", self.cfg.model)

    def teacher(self, device: torch.device | None = None) -> nn.Module:
        """The frozen CLIP text teacher, built lazily and kept on ``device`` (eval, no grad)."""
        if self._teacher is None:
            from vivlmoe.models.factory import build_teacher

            self._teacher = build_teacher(self.cfg.model)
        target = device or self.ctx.device
        if next(self._teacher.parameters()).device != target:
            self._teacher.to(target)
        self._teacher.eval()
        return self._teacher

    def image_teacher(self) -> nn.Module:
        """The frozen CLIP image teacher (Appendix-A retrieval), lazily built on ``ctx.device``."""
        if self._image_teacher is None:
            from vivlmoe.models.factory import build_image_teacher

            self._image_teacher = build_image_teacher(self.cfg.model)
            expected = self.cfg.model.stage1.teacher_target
            if expected != "projected":
                raise ConfigError(
                    "task.eval_clip_retrieval compares student and CLIP image embeddings in the "
                    f"projected joint space; model.stage1.teacher_target={expected!r} is not comparable"
                )
        if next(self._image_teacher.parameters()).device != self.ctx.device:
            self._image_teacher.to(self.ctx.device)
        self._image_teacher.eval()
        return self._image_teacher

    # ------------------------------------------------------------------ data
    def tokenizers(self) -> TokenizerBundle:
        if self._tokenizers is None:
            self._tokenizers = build_tokenizers(self.cfg.model, self.cfg.data, need_clip=True)
        return self._tokenizers

    def _run_dir(self) -> Path | None:
        return Path(self.cfg.run.dir) / self.cfg.run.name if self.ctx.is_main else None

    def _teacher_collator(self) -> TeacherCollator:
        tok = self.tokenizers()
        lowercase = self.cfg.model.stage1.teacher_lowercase
        return TeacherCollator(
            tok.require_clip(), tok.vi, tok.en_max_len, tok.vi_max_len, lowercase
        )

    def build_train_loader(self) -> DataLoader:
        """Parallel-text train loader (``build_parallel_text`` + ``TeacherCollator``)."""
        cfg, d = self.cfg, self.cfg.data
        if d.train is None:
            raise ConfigError("stage1_teacher needs data.train (a parallel-text source)")
        train_ds = build_parallel_text(d.train, self._run_dir())
        return build_train_loader(
            train_ds, self._teacher_collator(), d, self.ctx.world_size, self.ctx.rank, cfg.run.seed
        )

    def build_eval_loaders(self) -> dict[str, DataLoader | RetrievalLoaders]:
        """Eval loaders per ``data.evals`` entry: teacher-MSE (parallel text) or CLIP-space retrieval.

        Public so ``scripts/evaluate.py`` can recompute dev metrics without indexing ``data.train``.
        """
        cfg, d = self.cfg, self.cfg.data
        tok = self.tokenizers()
        collate = self._teacher_collator()
        evals: dict[str, DataLoader | RetrievalLoaders] = {}
        self._split_kinds.clear()
        for split, src in d.evals.items():
            records, adapter = build_records(src, None, self._run_dir(), self.ctx.is_main)
            info = adapter.holdout_info.get(src.split)
            self._dev_split_sha1[split] = info.sha1 if info is not None else None
            if records.record_type is ParallelTextRecord:
                evals[split] = build_eval_loader(
                    ParallelTextDataset(records), collate, d, self.ctx.world_size, self.ctx.rank
                )
                self._split_kinds[split] = "parallel"
            elif records.record_type is ImageTextRecord:
                if not cfg.task.eval_clip_retrieval:
                    raise ConfigError(
                        f"data.evals.{split} is an image-text split; stage1_teacher evaluates it "
                        "only with task.eval_clip_retrieval: true"
                    )
                evals[split] = self._retrieval_loaders(split, src, records, adapter, tok)
                self._split_kinds[split] = "retrieval"
            else:
                raise ConfigError(
                    f"data.evals.{split}: stage1_teacher cannot evaluate {records.record_type.__name__} records"
                )
        return evals

    def build_loaders(self) -> LoaderBundle:
        return LoaderBundle(train=self.build_train_loader(), evals=self.build_eval_loaders())

    def _retrieval_loaders(
        self, split: str, src: DataSourceCfg, records: Any, adapter: Any, tok: TokenizerBundle
    ) -> RetrievalLoaders:
        if self.cfg.model.stage1.teacher_kind == "siglip":  # the teacher's own preprocessing
            spec = square_spec(teacher_vision_cfg(self.cfg.model))
            eval_tf = build_transform(spec, train=False)
        else:
            spec, _train_tf, eval_tf = build_transform_for(self.cfg)
        self._transform_spec = spec.to_dict()
        dataset = RetrievalEvalDataset(
            records, adapter.image_store(src.split), eval_tf, draft_size_for(self.cfg, spec)
        )
        log.info(
            "clip-space retrieval split %s: %d images / %d captions",
            split,
            len(dataset.image_ids),
            len(dataset.captions),
        )
        return build_retrieval_loaders(
            dataset, StudentTextCollator(tok.vi, tok.vi_max_len), self.cfg.data, self.ctx
        )

    # ------------------------------------------------------------------ training
    def train_step(self, model: nn.Module, batch: dict[str, Any], sc: StepContext) -> StepOutput:
        teacher = self.teacher(sc.device)
        # The Eq. 12 regression target is computed in fp32 regardless of the Trainer's autocast
        # (bf16 by default): the teacher is frozen and cheap next to XLM-R + backward, and the
        # TeacherMSEEvaluator compares against the same fp32 target.
        with torch.autocast(device_type=sc.device.type, enabled=False):
            z_en = teacher(batch["en_input_ids"], batch["en_attention_mask"]).float()
        out = model(batch, "stage1")
        loss = self.loss(out.student, z_en)
        with torch.no_grad():
            metrics = {
                "student_norm": out.student.detach().float().norm(dim=-1).mean().item(),
                "teacher_norm": z_en.norm(dim=-1).mean().item(),
            }
        return StepOutput(
            loss=loss,
            batch_size=len(batch["uids"]),
            num_tokens=int(batch["vi_attention_mask"].sum().item()),
            metrics=metrics,
        )

    # ------------------------------------------------------------------ evaluation
    def evaluators(self) -> dict[str, Evaluator]:
        if not self._split_kinds and self.cfg.data.evals:
            raise RuntimeError("evaluators() must be called after build_eval_loaders()")
        if self._evaluators is not None:
            return self._evaluators
        preset = self.cfg.model.stage1.loss_preset
        out: dict[str, Evaluator] = {}
        clip_ret: ClipSpaceRetrievalEvaluator | None = None
        for split, kind in self._split_kinds.items():
            if kind == "parallel":
                out[split] = TeacherMSEEvaluator(self.ctx, self.teacher, preset)
            else:
                if clip_ret is None:  # one instance: its image cache is keyed by split
                    clip_ret = ClipSpaceRetrievalEvaluator(
                        self.ctx,
                        self.image_teacher,
                        _student_embeds,
                        embed_dim=self._teacher_dim(),
                    )
                out[split] = clip_ret
        self._evaluators = out
        return out

    def _teacher_dim(self) -> int:
        return teacher_output_dim(self.cfg.model)

    def selection_metric(self) -> tuple[str | None, str]:
        for split, kind in self._split_kinds.items():
            if kind == "parallel":
                return f"{split}/mse", "min"
        return None, "min"

    # ------------------------------------------------------------------ state
    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "dev_split_sha1": dict(self._dev_split_sha1),
            "transform_spec": self._transform_spec,
        }


__all__ = ["StudentTextCollator", "TeacherStudentTask"]
