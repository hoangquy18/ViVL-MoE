#!/usr/bin/env python
"""Verify the Python environment can run ViVL-MoE (imports, versions, model loading gate, HF cache)."""

from __future__ import annotations

import importlib
import os
import sys

REQUIRED = [
    "torch",
    "torchvision",
    "transformers",
    "safetensors",
    "yaml",
    "numpy",
    "PIL",
    "tqdm",
    "tensorboard",
    "rouge_score",
    "pycocoevalcap",
    "sentencepiece",
    "google.protobuf",
    "pandas",
    "sklearn",
    "pytest",
]
OPTIONAL = ["wandb", "datasets", "underthesea", "evaluate", "sacrebleu", "ruff"]


def _version(mod: object) -> str:
    return str(getattr(mod, "__version__", getattr(mod, "VERSION", "?")))


def main() -> int:
    ok = True
    print("== required modules ==")
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            print(f"  ok   {name:18s} {_version(mod)}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  MISS {name:18s} ({type(exc).__name__}: {exc})")
    print("== optional modules ==")
    for name in OPTIONAL:
        try:
            mod = importlib.import_module(name)
            print(f"  ok   {name:18s} {_version(mod)}")
        except Exception:  # noqa: BLE001
            print(f"  --   {name:18s} (optional, not installed)")

    import torch

    print("== torch ==")
    print(
        f"  version {torch.__version__}  cuda={torch.cuda.is_available()}  mps={torch.backends.mps.is_available()}"
    )
    try:
        from transformers.utils.import_utils import check_torch_load_is_safe

        check_torch_load_is_safe()
        print("  pytorch_model.bin loading: allowed (torch>=2.6)")
    except Exception:  # noqa: BLE001
        print(
            "  pytorch_model.bin loading: BLOCKED by transformers on this torch -> "
            "run scripts/fetch_models.py to convert .bin-only repos to safetensors"
        )

    print("== local model paths ==")
    from vivlmoe.utils.hf import local_model_map, model_paths_file

    mapping = local_model_map()
    if not mapping:
        print(f"  none registered ({model_paths_file()} absent)")
    for hf_id, path in mapping.items():
        exists = os.path.isdir(path)
        print(f"  {'ok  ' if exists else 'MISS'} {hf_id} -> {path}")
        ok = ok and exists

    print("== HF cache (small models) ==")
    try:
        from huggingface_hub import try_to_load_from_cache

        for repo, files in {
            "openai/clip-vit-base-patch32": [
                "config.json",
                "model.safetensors",
                "preprocessor_config.json",
            ],
            "FacebookAI/xlm-roberta-base": ["config.json", "model.safetensors", "tokenizer.json"],
        }.items():
            if repo in mapping:
                print(f"  ok   {repo} served from local dir {mapping[repo]}")
                continue
            missing = [f for f in files if not isinstance(try_to_load_from_cache(repo, f), str)]
            print(f"  {'ok  ' if not missing else 'MISS'} {repo} missing={missing}")
    except Exception as exc:  # noqa: BLE001
        print(f"  could not inspect cache: {exc}")

    print("ENV OK" if ok else "ENV INCOMPLETE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raise SystemExit(main())
