"""Image preprocessing derived from the vision tower's ``CLIPImageProcessor``.

The legacy code normalised with ImageNet statistics while the tower expected CLIP statistics.
Here the spec is read from the *constructed* processor (``.size`` / ``.crop_size`` dicts, mean, std,
resample) so it cannot drift, and it is stored in every checkpoint (``transform_spec``).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from vivlmoe.config.schema import TransformCfg, VisionCfg
from vivlmoe.utils.hf import resolve_model_path

log = logging.getLogger(__name__)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_PIL_RESAMPLE = {0: "nearest", 1: "lanczos", 2: "bilinear", 3: "bicubic", 4: "box", 5: "hamming"}
_TV_RESAMPLE = {
    "nearest": InterpolationMode.NEAREST,
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "lanczos": InterpolationMode.LANCZOS,
    "box": InterpolationMode.BOX,
    "hamming": InterpolationMode.HAMMING,
}
Augment = Literal["none", "random_resized_crop"]


@dataclass(frozen=True)
class ImageTransformSpec:
    size: int  # shortest edge after resize
    crop_size: int  # centre-crop side (network input resolution)
    mean: tuple[float, float, float] = CLIP_MEAN
    std: tuple[float, float, float] = CLIP_STD
    resample: str = "bicubic"
    augment: Augment = "none"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mean"], d["std"] = list(self.mean), list(self.std)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ImageTransformSpec:
        return cls(
            size=int(d["size"]),
            crop_size=int(d["crop_size"]),
            mean=tuple(float(x) for x in d["mean"]),  # type: ignore[arg-type]
            std=tuple(float(x) for x in d["std"]),  # type: ignore[arg-type]
            resample=str(d.get("resample", "bicubic")),
            augment=d.get("augment", "none"),
        )


def _edge(d: Any, *keys: str) -> int:
    """``size``/``crop_size`` may be an int (old configs) or a dict (shortest_edge / height)."""
    if isinstance(d, int):
        return d
    if isinstance(d, dict):
        for k in keys:
            if k in d:
                return int(d[k])
    raise ValueError(f"cannot read an edge length from {d!r}")


def spec_from_clip_processor(
    name_or_dir: str, fallback_image_size: int | None = None
) -> ImageTransformSpec:
    """Read the constructed ``CLIPImageProcessor``; fall back to CLIP constants with a WARNING.

    The fallback (no ``preprocessor_config.json``) uses the vision config's ``image_size`` or
    ``fallback_image_size``; if neither is available it raises.
    """
    path = resolve_model_path(name_or_dir)
    from transformers import CLIPImageProcessor

    try:
        proc = CLIPImageProcessor.from_pretrained(path)
    except (OSError, ValueError) as exc:
        image_size = fallback_image_size
        if image_size is None:
            try:
                from transformers import AutoConfig

                cfg = AutoConfig.from_pretrained(path)
                image_size = int(getattr(cfg, "vision_config", cfg).image_size)
            except (OSError, ValueError, AttributeError) as exc2:
                raise ValueError(
                    f"no preprocessor_config.json for {name_or_dir!r} ({exc}) and no image_size "
                    f"could be derived ({exc2}); pass fallback_image_size"
                ) from exc2
        log.warning(
            "no preprocessor_config.json for %r (%s); using CLIP mean/std with size=%d",
            name_or_dir,
            exc,
            image_size,
        )
        return ImageTransformSpec(size=image_size, crop_size=image_size)
    size = _edge(proc.size, "shortest_edge", "height")
    crop = _edge(proc.crop_size, "height", "shortest_edge") if proc.do_center_crop else size
    resample = _PIL_RESAMPLE.get(int(proc.resample), "bicubic")
    mean = tuple(float(x) for x in proc.image_mean) if proc.do_normalize else (0.0, 0.0, 0.0)
    std = tuple(float(x) for x in proc.image_std) if proc.do_normalize else (1.0, 1.0, 1.0)
    return ImageTransformSpec(size=size, crop_size=crop, mean=mean, std=std, resample=resample)  # type: ignore[arg-type]


def build_transform_spec(vision: VisionCfg, transform: TransformCfg) -> ImageTransformSpec:
    """Spec for a run: ``clip_processor`` (from the tower) or ``manual`` (explicit size)."""
    if transform.source == "manual":
        if transform.size is None:
            raise ValueError("data.transform.size is required when data.transform.source=manual")
        crop = transform.crop_size if transform.crop_size is not None else transform.size
        spec = ImageTransformSpec(size=transform.size, crop_size=crop)
    elif not vision.pretrained:
        tiny = vision.tiny_config or {}
        image_size = int(tiny.get("image_size", 224))
        log.info(
            "vision.pretrained=false: transform derived from tiny_config image_size=%d", image_size
        )
        spec = ImageTransformSpec(size=image_size, crop_size=image_size)
    else:
        spec = spec_from_clip_processor(vision.name)
        if transform.size is not None or transform.crop_size is not None:
            raise ValueError("data.transform.size/crop_size are only allowed with source=manual")
    return ImageTransformSpec(**{**asdict(spec), "augment": transform.augment})


def build_transform(spec: ImageTransformSpec, train: bool = False) -> T.Compose:
    """Resize(shortest edge, antialias) -> CenterCrop -> ToTensor -> Normalize (== CLIPImageProcessor).

    With ``augment='random_resized_crop'`` and ``train=True`` the first two steps become a
    ``RandomResizedCrop`` to ``crop_size``.
    """
    interp = _TV_RESAMPLE[spec.resample]
    ops: list[Any]
    if train and spec.augment == "random_resized_crop":
        ops = [
            T.RandomResizedCrop(
                spec.crop_size, scale=(0.5, 1.0), interpolation=interp, antialias=True
            )
        ]
    else:
        ops = [
            T.Resize(spec.size, interpolation=interp, antialias=True),
            T.CenterCrop(spec.crop_size),
        ]
    ops += [T.ToTensor(), T.Normalize(spec.mean, spec.std)]
    return T.Compose(ops)


def assert_matches_hf_processor(
    name_or_dir: str, images: list[Image.Image], atol: float = 1e-3
) -> None:
    """Test helper: our pipeline must reproduce ``CLIPImageProcessor`` pixel-for-pixel (within atol)."""
    from transformers import CLIPImageProcessor

    proc = CLIPImageProcessor.from_pretrained(resolve_model_path(name_or_dir))
    spec = spec_from_clip_processor(name_or_dir)
    tf = build_transform(spec, train=False)
    ours = torch.stack([tf(img.convert("RGB")) for img in images])
    theirs = proc(images=images, return_tensors="pt")["pixel_values"]
    if ours.shape != theirs.shape:
        raise AssertionError(
            f"shape mismatch: ours {tuple(ours.shape)} vs HF {tuple(theirs.shape)}"
        )
    diff = (ours - theirs).abs().max().item()
    if diff > atol:
        raise AssertionError(f"max |ours - HF| = {diff:.5f} > {atol}")


__all__ = [
    "CLIP_MEAN",
    "CLIP_STD",
    "ImageTransformSpec",
    "assert_matches_hf_processor",
    "build_transform",
    "build_transform_spec",
    "spec_from_clip_processor",
]
