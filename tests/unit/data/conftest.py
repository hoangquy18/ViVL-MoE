"""Fixtures for the data tests: tiny images, transforms, fake tokenizers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.fakes import build_fake_bundle
from vivlmoe.data.tokenization import TokenizerBundle
from vivlmoe.data.transforms import ImageTransformSpec, build_transform

TINY_SPEC = ImageTransformSpec(size=32, crop_size=32)


def write_images(root: Path, names: Sequence[str], size: int = 40, seed: int = 0) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for name in names:
        arr = rng.integers(0, 256, size=(size, size + 8, 3), dtype=np.uint8)
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr, "RGB").save(path)


@pytest.fixture()
def tiny_transform():
    return build_transform(TINY_SPEC, train=False)


@pytest.fixture(scope="session")
def bundle() -> TokenizerBundle:
    return build_fake_bundle()
