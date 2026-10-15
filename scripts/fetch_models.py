#!/usr/bin/env python
"""Download pretrained models and convert ``.bin``-only repos to safetensors.

transformers>=4.5x refuses ``pytorch_model.bin`` when torch<2.6 (CVE gate). ``apple/DFN5B-CLIP-ViT-H-14-378``
and ``VietAI/vit5-large`` ship ``.bin`` only, so this script downloads them once into ``models/<name>/``,
writes ``model.safetensors`` and registers the directories in ``configs/_base/model_paths.yaml``.
Repos that already provide ``model.safetensors`` are simply completed in the Hugging Face cache.

Usage::

    python scripts/fetch_models.py --only small     # clip-vit-base-patch32 + xlm-roberta-base (smoke tests)
    python scripts/fetch_models.py --only paper     # DFN ViT-H/14-378 + xlm-roberta-large + ViT5-large
    python scripts/fetch_models.py                  # both (v1)
    python scripts/fetch_models.py --only v2        # v2 backbone (requirements-v2.txt environment)

``--only v2`` completes the Hugging Face cache for C-RADIOv4-SO400M (at ``model.vision.revision`` of
``configs/model/v2.yaml``), mmBERT-base, the SigLIP 2 giant / So400m towers and Qwen3.5-0.8B. These ship
safetensors (sharded ones through ``model.safetensors.index.json``), so nothing is converted. C-RADIO's
remote code is imported once here, by name and revision as the tower loads it, so the torchrun ranks find
it cached instead of writing it concurrently. The printed commit is the one to pin as ``vision.revision``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "0"

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# (hf_id, kind, needs_conversion)
MODELS: dict[str, list[tuple[str, str, bool]]] = {
    "small": [
        ("openai/clip-vit-base-patch32", "clip", False),
        ("FacebookAI/xlm-roberta-base", "text", False),
        ("VietAI/vit5-base", "t5", True),
    ],
    "paper": [
        ("apple/DFN5B-CLIP-ViT-H-14-378", "clip", True),
        ("FacebookAI/xlm-roberta-large", "text", False),
        ("VietAI/vit5-large", "t5", True),
    ],
    "v2": [
        ("nvidia/C-RADIOv4-SO400M", "radio", False),
        ("jhu-clsp/mmBERT-base", "text", False),
        ("google/siglip2-giant-opt-patch16-384", "siglip", False),  # distillation teacher
        ("google/siglip2-so400m-patch14-384", "siglip", False),  # alternative vision tower
        ("Qwen/Qwen3.5-0.8B", "llm", False),
    ],
}
V2_PATTERNS = ["*.json", "*.py", "*.txt", "*.model", "*.jinja", "tokenizer*", "*.safetensors"]
COMMON_PATTERNS = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer*",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "spiece.model",
    "sentencepiece.bpe.model",
]
EXPECTED = {
    "apple/DFN5B-CLIP-ViT-H-14-378": {
        "vision.hidden_size": 1280,
        "vision.image_size": 378,
        "vision.patch_size": 14,
        "projection_dim": 1024,
    },
    "openai/clip-vit-base-patch32": {
        "vision.hidden_size": 768,
        "vision.image_size": 224,
        "vision.patch_size": 32,
        "projection_dim": 512,
    },
    "FacebookAI/xlm-roberta-large": {"hidden_size": 1024},
    "FacebookAI/xlm-roberta-base": {"hidden_size": 768},
    "VietAI/vit5-large": {"d_model": 1024, "decoder_start_token_id": 0},
    "VietAI/vit5-base": {"d_model": 768, "decoder_start_token_id": 0},
    "jhu-clsp/mmBERT-base": {"hidden_size": 768},
    "google/siglip2-giant-opt-patch16-384": {"vision.image_size": 384, "vision.patch_size": 16},
    "google/siglip2-so400m-patch14-384": {
        "vision.hidden_size": 1152,
        "vision.image_size": 384,
        "vision.patch_size": 14,
    },
}


def _cfg_value(cfg: object, dotted: str) -> object:
    obj = cfg
    for part in dotted.split("."):
        if part == "vision":
            obj = obj.vision_config
        else:
            obj = getattr(obj, part)
    return obj


def assert_expected(hf_id: str, path: str) -> None:
    from transformers import AutoConfig

    if hf_id not in EXPECTED:
        return
    cfg = AutoConfig.from_pretrained(path)
    for key, want in EXPECTED.get(hf_id, {}).items():
        got = _cfg_value(cfg, key)
        if got != want:
            raise SystemExit(f"{hf_id}: expected {key}={want}, got {got} (wrong repo or revision?)")
    print(f"  verified config of {hf_id}: {EXPECTED.get(hf_id, {})}")


def convert_bin_to_safetensors(bin_path: Path, out_path: Path) -> None:
    import torch
    from safetensors.torch import save_file

    print(f"  converting {bin_path.name} -> {out_path.name} (this can take a few minutes)")
    sd = torch.load(bin_path, map_location="cpu", weights_only=True)
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    seen_storages: set[int] = set()
    out: dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value
        storage_ptr = tensor.untyped_storage().data_ptr()
        if storage_ptr in seen_storages or not tensor.is_contiguous():
            tensor = tensor.clone().contiguous()
        seen_storages.add(tensor.untyped_storage().data_ptr())
        out[key] = tensor
    save_file(out, str(out_path), metadata={"format": "pt"})
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1e9:.2f} GB, {len(out)} tensors)")


def fetch_cached(hf_id: str, kind: str) -> bool:
    """Complete the HF cache for a repo that ships safetensors. Returns False if it does not."""
    from huggingface_hub import snapshot_download

    patterns = COMMON_PATTERNS + ["model.safetensors"]
    snap = snapshot_download(repo_id=hf_id, allow_patterns=patterns)
    if not (Path(snap) / "model.safetensors").exists():
        print(f"  {hf_id}: repo ships no model.safetensors -> converting pytorch_model.bin locally")
        return False
    required = ["config.json"] + (["preprocessor_config.json"] if kind == "clip" else [])
    missing = [f for f in required if not (Path(snap) / f).exists()]
    if missing:
        raise SystemExit(f"{hf_id}: snapshot {snap} is missing {missing}")
    print(f"  cached {hf_id} at {snap}")
    assert_expected(hf_id, snap)
    return True


def fetch_converted(hf_id: str, kind: str, models_dir: Path, keep_bin: bool) -> Path:
    from huggingface_hub import snapshot_download

    local_dir = models_dir / hf_id.split("/")[-1]
    local_dir.mkdir(parents=True, exist_ok=True)
    if (local_dir / "model.safetensors").exists():
        print(f"  {hf_id}: {local_dir / 'model.safetensors'} already present, skipping download")
    else:
        snapshot_download(
            repo_id=hf_id,
            allow_patterns=COMMON_PATTERNS + ["pytorch_model.bin"],
            local_dir=str(local_dir),
        )
        bin_path = local_dir / "pytorch_model.bin"
        if not bin_path.exists():
            raise SystemExit(f"{hf_id}: pytorch_model.bin not found after download in {local_dir}")
        convert_bin_to_safetensors(bin_path, local_dir / "model.safetensors")
        if not keep_bin:
            bin_path.unlink()
    assert_expected(hf_id, str(local_dir))
    return local_dir


def v2_revisions() -> dict[str, str | None]:
    """Pinned revisions of configs/model/v2.yaml (the C-RADIO remote code)."""
    from vivlmoe.config.loader import from_dict, load_yaml_with_includes
    from vivlmoe.config.schema import ModelCfg

    raw = load_yaml_with_includes(REPO_ROOT / "configs" / "model" / "v2.yaml")["model"]
    vision = from_dict(ModelCfg, raw, "model").vision
    return {vision.name: vision.revision}


def fetch_v2(hf_id: str, kind: str, revision: str | None) -> str:
    """Complete the HF cache for a v2 model; returns the snapshot commit."""
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(repo_id=hf_id, revision=revision, allow_patterns=V2_PATTERNS))
    index = snap / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        missing = sorted(f for f in shards if not (snap / f).exists())
        if missing:
            raise SystemExit(f"{hf_id}: snapshot {snap} is missing shards {missing}")
    elif not (snap / "model.safetensors").exists():
        raise SystemExit(f"{hf_id}: no model.safetensors / model.safetensors.index.json in {snap}")
    if kind == "radio":
        from transformers import AutoConfig

        AutoConfig.from_pretrained(hf_id, revision=revision, trust_remote_code=True)
        print(f"  imported the remote code of {hf_id} (trust_remote_code)")
    print(f"  cached {hf_id} at {snap}")
    assert_expected(hf_id, str(snap))
    return snap.name


def register_paths(mapping: dict[str, str]) -> None:
    import yaml

    path = REPO_ROOT / "configs" / "_base" / "model_paths.yaml"
    existing: dict[str, str] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            existing = (yaml.safe_load(fh) or {}).get("model_paths", {}) or {}
    existing.update(mapping)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(
            "# Generated by scripts/fetch_models.py: hf_id -> local safetensors directory (relative to repo root).\n"
        )
        yaml.safe_dump({"model_paths": existing}, fh, sort_keys=True)
    print(f"  registered {len(mapping)} local model dir(s) in {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only", choices=["small", "paper", "v2", "all"], default="all", help="all = small + paper"
    )
    parser.add_argument("--models-dir", default=str(REPO_ROOT / "models"))
    parser.add_argument(
        "--keep-bin", action="store_true", help="keep pytorch_model.bin after conversion"
    )
    args = parser.parse_args()

    groups = ["small", "paper"] if args.only == "all" else [args.only]
    models_dir = Path(args.models_dir)
    mapping: dict[str, str] = {}
    commits: dict[str, str] = {}
    for group in groups:
        print(f"== {group} ==")
        if group == "v2":
            revisions = v2_revisions()
            for hf_id, kind, _ in MODELS[group]:
                commits[hf_id] = fetch_v2(hf_id, kind, revisions.get(hf_id))
            for hf_id, rev in revisions.items():
                if rev is None:
                    print(
                        f"  pin it: model.vision.revision: {commits[hf_id]} (configs/model/v2.yaml)"
                    )
            continue
        for hf_id, kind, needs_conversion in MODELS[group]:
            if needs_conversion or not fetch_cached(hf_id, kind):
                local_dir = fetch_converted(hf_id, kind, models_dir, args.keep_bin)
                try:
                    mapping[hf_id] = str(local_dir.relative_to(REPO_ROOT))
                except ValueError:
                    mapping[hf_id] = str(local_dir)
    if mapping:
        register_paths(mapping)
    fetched = [m[0] for g in groups for m in MODELS[g]]
    print(json.dumps({"fetched": fetched, "converted": mapping, "commits": commits}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
