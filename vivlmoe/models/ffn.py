"""Dense feed-forward network of ``f_enc`` and the expert unit of the MoE layers (plan decision 4).

``fc1 -> GELU -> Dropout(inner) -> fc2``. The identical class is used for the dense FFN and for every
expert so that sparse upcycling (paper Sec. 5.3) is an exact copy; the inner dropout is what separates
identical shared experts during training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vivlmoe.models.outputs import RouterOutput


class DenseFFN(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, inner_dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.dropout = nn.Dropout(inner_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))

    def forward_block(
        self, x: torch.Tensor, modality_ids: torch.Tensor, token_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, RouterOutput | None]:
        """Block-level interface shared with ``MoEFFN``: ``(hidden [B,S,d], router_output|None)``."""
        return self.forward(x), None
