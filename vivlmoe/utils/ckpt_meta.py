"""Checkpoint sidecar metadata.

Every checkpoint ``<name>.pt`` written by :mod:`vivlmoe.training.checkpoint` gets a JSON sidecar
``<name>.meta.json`` holding the non-tensor fields (resolved config, tower provenance, excluded
modules, fingerprint, step, metrics). Readers that only need metadata (config inheritance via
``model.from_checkpoint``) use the sidecar and never load tensors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

META_FIELDS = (
    "format_version",
    "config",
    "tower_provenance",
    "excluded_modules",
    "model_fingerprint",
    "step",
    "epoch",
    "metrics",
    "created",
    "world_size",
    "dev_split_sha1",
    "transform_spec",
    "upcycle_info",
)


def meta_path(ckpt_path: str | Path) -> Path:
    p = Path(ckpt_path)
    return (
        p.with_name(p.name + ".meta.json")
        if p.suffix == ".pt"
        else p.with_suffix(p.suffix + ".meta.json")
    )


def write_meta(ckpt_path: str | Path, ckpt: dict[str, Any]) -> Path:
    path = meta_path(ckpt_path)
    meta = {k: ckpt[k] for k in META_FIELDS if k in ckpt}
    with path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1, default=str)
    return path


def read_meta(ckpt_path: str | Path) -> dict[str, Any]:
    """Read the sidecar; fall back to a memory-mapped ``torch.load`` of the checkpoint itself."""
    path = meta_path(ckpt_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_file}")
    import torch

    ckpt = torch.load(ckpt_file, map_location="cpu", mmap=True, weights_only=False)
    return {k: ckpt[k] for k in META_FIELDS if k in ckpt}
