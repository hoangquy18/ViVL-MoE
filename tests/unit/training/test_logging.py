"""setup_logging: train.log keeps the package's DEBUG records but not those of the image decoder."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import torch
from PIL import Image

from vivlmoe.training.distributed import DistContext
from vivlmoe.training.logging import setup_logging


def test_train_log_leaves_out_the_debug_records_of_image_decoding(tmp_path: Path) -> None:
    root, pil = logging.getLogger(), logging.getLogger("PIL")
    saved = list(root.handlers), root.level, pil.level
    try:
        ctx = DistContext(
            rank=0, local_rank=0, world_size=1, device=torch.device("cpu"), backend="none"
        )
        setup_logging(ctx, tmp_path)
        png = io.BytesIO()
        Image.new("RGB", (4, 4)).save(png, format="PNG")
        png.seek(0)
        Image.open(png).load()  # PIL logs every chunk it reads at DEBUG
        logging.getLogger("vivlmoe.training.distributed").debug("rendezvous details")
        for h in root.handlers:
            h.flush()
        text = (tmp_path / "train.log").read_text(encoding="utf-8")
        assert "rendezvous details" in text and "PIL" not in text
    finally:  # setup_logging replaces the root handlers; restore pytest's
        for h in list(root.handlers):
            root.removeHandler(h)
            h.close()
        for h in saved[0]:
            root.addHandler(h)
        root.setLevel(saved[1])
        pil.setLevel(saved[2])
