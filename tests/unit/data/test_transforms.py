from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch
from PIL import Image

from vivlmoe.config.schema import TransformCfg, VisionCfg
from vivlmoe.data.transforms import (
    CLIP_MEAN,
    ImageTransformSpec,
    assert_matches_hf_processor,
    build_transform,
    build_transform_spec,
    spec_from_clip_processor,
)


def test_tiny_spec_shapes_and_roundtrip() -> None:
    spec = ImageTransformSpec(size=32, crop_size=32)
    tf = build_transform(spec)
    img = Image.new("RGB", (50, 70), (255, 0, 0))
    out = tf(img)
    assert out.shape == (3, 32, 32) and out.dtype == torch.float32
    assert torch.isclose(out[0, 0, 0], torch.tensor((1.0 - CLIP_MEAN[0]) / spec.std[0]))
    assert ImageTransformSpec.from_dict(spec.to_dict()) == spec
    aug = ImageTransformSpec(size=32, crop_size=32, augment="random_resized_crop")
    assert build_transform(aug, train=True)(img).shape == (3, 32, 32)
    assert build_transform(aug, train=False)(img).shape == (3, 32, 32)


def test_spec_from_tiny_config_and_manual() -> None:
    vis = VisionCfg(pretrained=False, tiny_config={"image_size": 32})
    spec = build_transform_spec(vis, TransformCfg())
    assert (spec.size, spec.crop_size, spec.augment) == (32, 32, "none")
    spec = build_transform_spec(
        vis, TransformCfg(source="manual", size=48, crop_size=40, augment="random_resized_crop")
    )
    assert (spec.size, spec.crop_size, spec.augment) == (48, 40, "random_resized_crop")
    with pytest.raises(ValueError):
        build_transform_spec(vis, TransformCfg(source="manual"))


def test_spec_from_processor_and_fallback(tmp_path: Path, caplog) -> None:
    d = tmp_path / "model"
    d.mkdir()
    (d / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "image_processor_type": "CLIPImageProcessor",
                "size": {"shortest_edge": 96},
                "crop_size": {"height": 80, "width": 80},
                "do_center_crop": True,
                "do_normalize": True,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.2, 0.2, 0.2],
                "resample": 3,
            }
        ),
        encoding="utf-8",
    )
    spec = spec_from_clip_processor(str(d))
    assert spec == ImageTransformSpec(96, 80, (0.5, 0.5, 0.5), (0.2, 0.2, 0.2), "bicubic")
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "config.json").write_text(
        json.dumps(
            {
                "model_type": "clip_vision_model",
                "image_size": 64,
                "hidden_size": 32,
                "num_attention_heads": 4,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "patch_size": 16,
            }
        ),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        spec = spec_from_clip_processor(str(bare))
    assert spec.size == 64 and spec.crop_size == 64 and spec.mean == CLIP_MEAN
    assert any("no preprocessor_config.json" in r.message for r in caplog.records)
    assert spec_from_clip_processor(str(tmp_path / "missing"), fallback_image_size=24).size == 24
    with pytest.raises(ValueError):
        spec_from_clip_processor(str(tmp_path / "missing"))


@pytest.mark.hf_cache
def test_parity_with_clip_image_processor() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    images = [
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
        for h, w in [(300, 400), (224, 224), (500, 260)]
    ]
    assert_matches_hf_processor("openai/clip-vit-base-patch32", images, atol=1e-3)
    spec = spec_from_clip_processor("openai/clip-vit-base-patch32")
    assert (spec.size, spec.crop_size) == (224, 224)


# ---------------------------------------------------------------------------------- v2 towers
def test_tower_spec_for_siglip_is_square_bilinear_half_normalised() -> None:
    sig = VisionCfg(kind="siglip", pretrained=False, tiny_config={"image_size": 32}, image_size=48)
    spec = build_transform_spec(sig, TransformCfg(source="tower"))
    assert (spec.size, spec.crop_size, spec.resize_mode, spec.resample) == (
        48,
        48,
        "square",
        "bilinear",
    )
    assert spec.mean == spec.std == (0.5, 0.5, 0.5)


def test_tower_spec_for_radio_keeps_pixels_in_unit_range() -> None:
    rad = VisionCfg(kind="radio", pretrained=False, tiny_config={"preferred_resolution": 32})
    spec = build_transform_spec(rad, TransformCfg(source="tower"))
    assert (spec.size, spec.mean, spec.std) == (32, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    x = build_transform(spec)(Image.new("RGB", (50, 30), (255, 128, 0)))
    assert x.shape == (3, 32, 32) and 0.0 <= float(x.min()) and float(x.max()) <= 1.0
    # pretrained C-RADIO: preferred resolution without running its remote code
    pre = VisionCfg(kind="radio", name="nvidia/C-RADIOv4-SO400M")
    assert build_transform_spec(pre, TransformCfg(source="tower")).size == 512


def test_clip_processor_source_is_rejected_for_v2_towers() -> None:
    sig = VisionCfg(kind="siglip", pretrained=False, tiny_config={"image_size": 32})
    with pytest.raises(ValueError, match="source: tower"):
        build_transform_spec(sig, TransformCfg())


def test_spec_dict_from_an_older_checkpoint_loads() -> None:
    d = ImageTransformSpec(size=32, crop_size=32).to_dict()
    d.pop("resize_mode")
    assert ImageTransformSpec.from_dict(d).resize_mode == "shortest_edge_crop"
