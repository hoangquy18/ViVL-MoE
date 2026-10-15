"""Hugging Face model-path resolution.

Configs always name models by their Hugging Face id (e.g. ``apple/DFN5B-CLIP-ViT-H-14-378``).
Repos that only ship ``pytorch_model.bin`` cannot be loaded by transformers>=4.5x on torch<2.6, so
``scripts/fetch_models.py`` converts them into local safetensors directories and records the mapping in
``configs/_base/model_paths.yaml`` (or the file named by ``$VIVLMOE_MODEL_PATHS``). :func:`resolve_model_path`
applies that mapping transparently; unmapped ids are returned unchanged.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml


def repo_root() -> Path:
    """Return the repository root (the directory containing ``vivlmoe/``)."""
    return Path(__file__).resolve().parents[2]


def model_paths_file() -> Path:
    env = os.environ.get("VIVLMOE_MODEL_PATHS")
    return Path(env) if env else repo_root() / "configs" / "_base" / "model_paths.yaml"


@lru_cache(maxsize=1)
def local_model_map() -> dict[str, str]:
    """Mapping ``hf_id -> local directory`` written by ``scripts/fetch_models.py`` (empty if absent)."""
    path = model_paths_file()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    mapping = data.get("model_paths", {})
    if not isinstance(mapping, dict):
        raise ValueError(f"{path}: 'model_paths' must be a mapping of hf_id -> directory")
    return {str(k): str(v) for k, v in mapping.items()}


def resolve_model_path(name: str) -> str:
    """Return a local directory for ``name`` when one is registered, else ``name`` itself."""
    if os.path.isdir(name):
        return name
    mapped = local_model_map().get(name)
    if mapped is None:
        return name
    path = Path(mapped)
    if not path.is_absolute():
        path = repo_root() / path
    if not path.is_dir():
        raise FileNotFoundError(
            f"model_paths.yaml maps {name!r} to {path}, which does not exist; "
            "re-run scripts/fetch_models.py or fix configs/_base/model_paths.yaml"
        )
    return str(path)
