"""Stage-1 student (paper Sec. 4.1, Eq. 12): XLM-R ``<s>`` hidden state -> FC1 -> ``z_vi`` in the teacher space.

v2 (mmBERT distilled from SigLIP 2) pools with the masked mean of the text states instead
(``model.stage1.student_pooling: mean``).
"""

from __future__ import annotations

from torch import nn

from vivlmoe.config.schema import ModelCfg
from vivlmoe.models.backbone import ViVLBackbone
from vivlmoe.models.outputs import TeacherStudentOutput
from vivlmoe.models.projections import init_module_
from vivlmoe.models.towers.text import TextTower


class Stage1Student(nn.Module):
    """Attributes ``text_encoder`` (handed to Stage 2 via ``init_from``) and ``fc1``."""

    def __init__(self, cfg: ModelCfg, teacher_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.text_encoder = TextTower(cfg.text)
        d_t = self.text_encoder.hidden_size
        hidden = cfg.stage1.fc1_hidden
        self.fc1: nn.Module
        if hidden is None:
            self.fc1 = nn.Linear(d_t, teacher_dim)
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(d_t, hidden), nn.GELU(), nn.Linear(hidden, teacher_dim)
            )
        self.teacher_dim = teacher_dim
        self.pooling = cfg.stage1.student_pooling
        init_module_(self.fc1, cfg.init_std)

    def forward(self, batch: dict, mode: str = "stage1") -> TeacherStudentOutput:
        if mode != "stage1":
            raise ValueError(f"Stage1Student: unknown mode {mode!r}")
        mask = batch["vi_attention_mask"]
        h = self.text_encoder(batch["vi_input_ids"], mask)
        pooled = ViVLBackbone.masked_mean(h, mask.bool()) if self.pooling == "mean" else h[:, 0]
        return TeacherStudentOutput(student=self.fc1(pooled).float())
