"""Tower modules and the tower-path lookup shared with checkpoint provenance."""

from __future__ import annotations

from torch import nn

from vivlmoe.models.towers.clip_teacher import CLIPImageTeacher, CLIPTextTeacher, teacher_output_dim
from vivlmoe.models.towers.text import TextTower
from vivlmoe.models.towers.vision import VisionTower

TOWER_ATTRS = ("vision_encoder", "text_encoder")


def find_tower_paths(model: nn.Module) -> dict[str, str]:
    """``{attr: dotted path}`` of the first ``vision_encoder`` / ``text_encoder`` found in ``model``.

    A bare backbone yields ``{"vision_encoder": "vision_encoder", ...}``; downstream heads nest it
    (``"backbone.vision_encoder"``). Towers are identified by their attribute name **and** type so
    state-dict prefixes, provenance keys and freeze paths all agree.
    """
    found: dict[str, str] = {}
    for path, module in model.named_modules():
        attr = path.rsplit(".", 1)[-1]
        if attr in TOWER_ATTRS and attr not in found:
            if not isinstance(module, VisionTower | TextTower):
                raise TypeError(
                    f"{path!r} is named like a tower but is {type(module).__name__}, not VisionTower/TextTower"
                )
            found[attr] = path
    return found


__all__ = [
    "TOWER_ATTRS",
    "CLIPImageTeacher",
    "CLIPTextTeacher",
    "TextTower",
    "VisionTower",
    "find_tower_paths",
    "teacher_output_dim",
]
