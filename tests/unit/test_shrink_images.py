"""scripts/shrink_images.py: shorter side capped in place, formats / EXIF kept, safe to rerun."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "shrink_images.py"


def _load():
    spec = importlib.util.spec_from_file_location("shrink_images", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _noise(w: int, h: int, mode: str = "RGB") -> Image.Image:
    rng = np.random.default_rng(0)
    channels = {"RGB": 3, "RGBA": 4}[mode]
    return Image.fromarray(rng.integers(0, 255, (h, w, channels), dtype=np.uint8), mode)


def test_target_size_caps_the_shorter_side() -> None:
    mod = _load()
    assert mod.target_size((640, 480), 512) is None  # COCO-sized: untouched
    assert mod.target_size((2000, 1000), 512) == (1024, 512)
    assert mod.target_size((900, 3000), 512) == (512, 1707)


def test_shrinks_in_place_keeps_formats_and_exif(tmp_path: Path) -> None:
    mod = _load()
    images = tmp_path / "images" / "sub"
    images.mkdir(parents=True)
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation tag: the training loader ignores it, it must survive anyway
    _noise(1600, 1200).save(images / "big.jpg", quality=95, exif=exif)
    _noise(640, 480).save(images / "coco.jpg", quality=95)
    _noise(1200, 1000, "RGBA").save(images / "alpha.png")
    _noise(1200, 1000).convert("P").save(images / "palette.png")
    _noise(1200, 1000).save(images / "anim.gif")
    (images / "broken.jpg").write_bytes(b"not an image")
    (images / "big.jpg.shrink.tmp").write_bytes(b"left by an interrupted run")
    sizes = {p.name: p.stat().st_size for p in images.iterdir()}

    assert mod.main(["--root", str(tmp_path), "--workers", "1"]) == 0
    assert not (images / "big.jpg.shrink.tmp").exists()
    with Image.open(images / "big.jpg") as im:
        assert im.format == "JPEG" and im.size == (683, 512)
        assert im.getexif().get(0x0112) == 6
    with Image.open(images / "alpha.png") as im:
        assert im.format == "PNG" and im.mode == "RGBA" and min(im.size) == 512
    with Image.open(images / "palette.png") as im:
        assert im.format == "PNG" and min(im.size) == 512
    for name in ("coco.jpg", "anim.gif", "broken.jpg"):  # small / unsupported / undecodable
        assert (images / name).stat().st_size == sizes[name]
    assert (images / "big.jpg").stat().st_size < sizes["big.jpg"]

    after = {p.name: p.stat().st_mtime_ns for p in images.iterdir()}
    assert mod.main(["--root", str(tmp_path), "--workers", "1"]) == 0  # rerun: nothing left to do
    assert {p.name: p.stat().st_mtime_ns for p in images.iterdir()} == after


def test_dry_run_changes_nothing(tmp_path: Path) -> None:
    mod = _load()
    (tmp_path / "images").mkdir()
    _noise(1600, 1200).save(tmp_path / "images" / "big.jpg")
    before = (tmp_path / "images" / "big.jpg").read_bytes()
    assert mod.main(["--root", str(tmp_path), "--dry-run", "--workers", "1"]) == 0
    assert (tmp_path / "images" / "big.jpg").read_bytes() == before


def test_parallel_run_as_a_script(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    for i in range(3):
        _noise(1200, 900).save(tmp_path / "images" / f"{i}.jpg")
    cmd = [sys.executable, str(SCRIPT), "--root", str(tmp_path), "--workers", "2"]
    subprocess.run(cmd, check=True, capture_output=True)
    for i in range(3):
        with Image.open(tmp_path / "images" / f"{i}.jpg") as im:
            assert im.size == (683, 512)
