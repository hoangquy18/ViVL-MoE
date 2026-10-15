"""Stage-1 teacher loss (paper Eq. 12): ``L_1 = (1/N) sum_i ||z_vi - z_en||^2``."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

TeacherLossPreset = Literal["eq12", "legacy"]


class TeacherMSELoss:
    """``eq12``: squared L2 distance summed over dims, mean over the batch.
    ``legacy``: L2-normalise both sides, then ``F.mse_loss`` (mean over dims, the legacy code's variant).
    """

    def __init__(self, preset: TeacherLossPreset = "eq12") -> None:
        if preset not in ("eq12", "legacy"):
            raise ValueError(f"teacher loss preset must be eq12|legacy, got {preset!r}")
        self.preset = preset

    def __call__(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        if student.shape != teacher.shape or student.ndim != 2:
            raise ValueError(
                f"student/teacher must be [B,P] with equal shapes, got "
                f"{tuple(student.shape)} / {tuple(teacher.shape)}"
            )
        s, t = student.float(), teacher.float().detach()
        if self.preset == "eq12":
            return (s - t).pow(2).sum(dim=-1).mean()
        return F.mse_loss(F.normalize(s, dim=-1), F.normalize(t, dim=-1))
