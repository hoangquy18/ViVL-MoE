"""Tower modules and the tower-path lookup shared with checkpoint provenance."""

from __future__ import annotations

from torch import nn

from vivlmoe.config.schema import VisionCfg
from vivlmoe.models.towers.clip_teacher import CLIPImageTeacher, CLIPTextTeacher
from vivlmoe.models.towers.siglip_teacher import SiglipImageTeacher, SiglipTextTeacher
from vivlmoe.models.towers.teachers import teacher_output_dim, teacher_vision_cfg
from vivlmoe.models.towers.text import TextTower
from vivlmoe.models.towers.vision import BaseVisionTower, VisionTower
from vivlmoe.models.towers.vision_radio import RadioVisionTower
from vivlmoe.models.towers.vision_siglip import SiglipVisionTower

TOWER_ATTRS = ("vision_encoder", "text_encoder")
_VISION_TOWERS: dict[str, type[BaseVisionTower]] = {
    "clip": VisionTower,
    "siglip": SiglipVisionTower,
    "radio": RadioVisionTower,
}


def build_vision_tower(cfg: VisionCfg) -> BaseVisionTower:
    """The vision tower class for ``cfg.kind`` (v1: CLIP; v2: SigLIP 2 or C-RADIO)."""
    return _VISION_TOWERS[cfg.kind](cfg)


def find_tower_paths(model: nn.Module) -> dict[str, str]:
    """``{attr: dotted path}`` of the first ``vision_encoder`` / ``text_encoder`` found in ``model``.

    A bare backbone yields ``{"vision_encoder": "vision_encoder", ...}``; downstream heads nest it
    (``"backbone.vision_encoder"``). Towers are identified by their attribute name **and** type so
    state-dict prefixes, provenance keys and freeze paths all agree.
    """
    found: dict[str, str] = {}
    for path, module in model.named_modules():
        if any(path.startswith(p + ".") for p in found.values()):
            continue  # inside a tower (remote code may reuse the names)
        attr = path.rsplit(".", 1)[-1]
        if attr in TOWER_ATTRS and attr not in found:
            if not isinstance(module, BaseVisionTower | TextTower):
                raise TypeError(
                    f"{path!r} is named like a tower but is {type(module).__name__}, not a "
                    "vision tower / TextTower"
                )
            found[attr] = path
    return found


__all__ = [
    "TOWER_ATTRS",
    "BaseVisionTower",
    "CLIPImageTeacher",
    "CLIPTextTeacher",
    "RadioVisionTower",
    "SiglipImageTeacher",
    "SiglipTextTeacher",
    "SiglipVisionTower",
    "TextTower",
    "VisionTower",
    "build_vision_tower",
    "find_tower_paths",
    "teacher_output_dim",
    "teacher_vision_cfg",
]
