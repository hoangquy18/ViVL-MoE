"""Tiny stand-in for the C-RADIO remote-code model, for hermetic CPU tests.

It honours the contract :class:`~vivlmoe.models.towers.vision_radio.RadioVisionTower` relies on, as
documented for ``AutoModel.from_pretrained("nvidia/C-RADIOv4-SO400M", trust_remote_code=True)``:
``model(x)`` with ``x`` in [0, 1] (normalised internally) returns ``(summary [B, D], spatial [B, N, D])``
in NLC layout, and the model exposes ``patch_size``, ``preferred_resolution`` and
``get_nearest_supported_resolution``.
"""

from __future__ import annotations

import torch
from torch import nn


class RadioStub(nn.Module):
    def __init__(self, embed_dim: int = 32, patch_size: int = 16, preferred_resolution: int = 32):
        super().__init__()
        self.patch_size = patch_size
        self.preferred_resolution = (preferred_resolution, preferred_resolution)
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.summary_head = nn.Linear(embed_dim, embed_dim)

    def get_nearest_supported_resolution(self, height: int, width: int) -> tuple[int, int]:
        p = self.patch_size
        return max(p, round(height / p) * p), max(p, round(width / p) * p)

    def forward(
        self, x: torch.Tensor, feature_fmt: str = "NLC"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.patch_embed((x - 0.5) / 0.5).flatten(2).transpose(1, 2)
        feats = feats + self.mlp(feats)
        return self.summary_head(feats.mean(dim=1)), feats
