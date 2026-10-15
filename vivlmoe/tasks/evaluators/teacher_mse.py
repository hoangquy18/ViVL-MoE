"""Stage-1 dev metric: the teacher MSE of Eq. 12 over a parallel-text split (``{'mse'}``).

Per-sample distances ``d_i`` (``eq12``: ``||z_vi - z_en||^2`` summed over dims; ``legacy``: mean
squared error of the L2-normalised vectors, i.e. exactly what the training loss averages) are
gathered by uid with :func:`gather_ordered_tensors` on every rank - a collective that also asserts
the shards cover ``0..N-1`` - and rank 0 returns their mean (``sum / count``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from vivlmoe.data.loaders import move_to_device
from vivlmoe.losses.teacher import TeacherLossPreset
from vivlmoe.tasks.evaluators.base import Evaluator, gather_ordered_tensors
from vivlmoe.tasks.evaluators.dump import metrics_path, write_metrics_json
from vivlmoe.training.distributed import DistContext
from vivlmoe.training.state import EvalContext

log = logging.getLogger(__name__)

TeacherFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def per_sample_teacher_distance(
    student: torch.Tensor, teacher: torch.Tensor, preset: TeacherLossPreset
) -> torch.Tensor:
    """``[B]`` per-sample values whose batch mean equals ``TeacherMSELoss(preset)``."""
    if student.shape != teacher.shape or student.ndim != 2:
        raise ValueError(
            f"student/teacher must be [B,P] with equal shapes, got "
            f"{tuple(student.shape)} / {tuple(teacher.shape)}"
        )
    s, t = student.float(), teacher.float()
    if preset == "eq12":
        return (s - t).pow(2).sum(dim=-1)
    if preset == "legacy":
        return (F.normalize(s, dim=-1) - F.normalize(t, dim=-1)).pow(2).mean(dim=-1)
    raise ValueError(f"teacher loss preset must be eq12|legacy, got {preset!r}")


class TeacherMSEEvaluator(Evaluator):
    """``teacher(en_input_ids, en_attention_mask) -> z_en``; ``model(batch, 'stage1').student -> z_vi``."""

    name = "teacher_mse"

    def __init__(
        self, ctx: DistContext, teacher: Callable[[], nn.Module], preset: TeacherLossPreset
    ) -> None:
        self.ctx = ctx
        self._teacher = teacher
        self.preset = preset

    def run(self, model: nn.Module, loader: DataLoader, ec: EvalContext) -> dict[str, float] | None:
        teacher = self._teacher()
        uids: list[int] = []
        dists: list[torch.Tensor] = []
        for batch in loader:
            batch = move_to_device(batch, ec.ctx.device)
            z_en = teacher(batch["en_input_ids"], batch["en_attention_mask"])
            out: Any = model(batch, "stage1")
            dists.append(per_sample_teacher_distance(out.student, z_en, self.preset).detach().cpu())
            uids.extend(int(u) for u in batch["uids"])
        ids = torch.tensor(uids, dtype=torch.int64)
        local = torch.cat(dists) if dists else torch.zeros(0)
        all_d = gather_ordered_tensors(ids, local, ec.ctx)
        if not ec.ctx.is_main:
            return None
        if all_d.numel() == 0:
            raise RuntimeError(f"eval split {ec.split_name!r} produced no samples")
        metrics = {"mse": float(all_d.sum() / all_d.numel()), "num_samples": float(all_d.numel())}
        write_metrics_json(
            metrics_path(ec.run_dir, ec.split_name, ec.step),
            {
                "split": ec.split_name,
                "step": ec.step,
                "evaluator": self.name,
                "loss_preset": self.preset,
                "metrics": metrics,
            },
        )
        return metrics


__all__ = ["TeacherMSEEvaluator", "per_sample_teacher_distance"]
