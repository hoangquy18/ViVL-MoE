#!/usr/bin/env python
"""Zero-shot Vietnamese image-text retrieval with the candidate v2 vision encoders (decision experiment).

    python scripts/probe_vision.py [--config configs/pretrain/stage2_contrastive.yaml] \
        [--splits ktvic_dev,uitviic_dev] [--encoders siglip2-g,siglip2-so400m,radio] \
        [--device cuda] [--batch-size 32] [--out runs/probe_vision.json] [key=value overrides ...]

Each encoder embeds the dev images and their Vietnamese reference captions in one space and is scored
with the training code's retrieval metrics (``retrieval_metrics``: R@1/5/10 both ways and rsum):

* ``siglip2-g`` / ``siglip2-so400m``: SigLIP 2 image and text towers. Captions are lowercased and padded
  to 64 tokens without an attention mask, as SigLIP 2 was trained; images get a square bilinear resize
  and mean/std 0.5.
* ``radio``: C-RADIOv4-SO400M's ``siglip2-g`` adaptor summary (square bicubic resize to 512, input in
  [0, 1]) against the SigLIP2-g text tower, i.e. the space the v2 mmBERT student is distilled into.
  ``--radio-text adaptor`` uses the text encoder bundled with the adaptor instead.

The ``siglip2-g`` text side is also the v2 distillation teacher, so its scores bound what mmBERT can
inherit. The dev splits come from ``data.evals`` of ``--config`` (roots via overrides, as for training).
Runs on one GPU; nothing is trained. Loading C-RADIOv4 executes its remote code (pin ``--radio-revision``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vivlmoe.config import load_config  # noqa: E402
from vivlmoe.data.builders import build_records  # noqa: E402
from vivlmoe.data.datasets import RetrievalEvalDataset  # noqa: E402
from vivlmoe.tasks.evaluators.retrieval import retrieval_metrics  # noqa: E402

log = logging.getLogger("scripts.probe_vision")

SIGLIP2_G = "google/siglip2-giant-opt-patch16-384"
SIGLIP2_SO400M = "google/siglip2-so400m-patch14-384"
RADIO_V4 = "nvidia/C-RADIOv4-SO400M"
SIGLIP_TEXT_LEN = 64


@dataclass
class Encoder:
    """Image and text embedders that share one space."""

    name: str
    transform: Callable[[Any], torch.Tensor]  # PIL image -> [3, H, W]
    encode_images: Callable[[torch.Tensor], torch.Tensor]  # [B, 3, H, W] -> [B, D]
    encode_texts: Callable[[list[str]], torch.Tensor]  # -> [B, D]


def square_transform(
    size: int, interpolation: T.InterpolationMode, normalize: bool
) -> Callable[[Any], torch.Tensor]:
    steps: list[Any] = [T.Resize((size, size), interpolation=interpolation), T.ToTensor()]
    if normalize:
        steps.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    return T.Compose(steps)


class SiglipTextEncoder:
    """SigLIP / SigLIP 2 text tower: pooled (last token + head) embedding, as at training time."""

    def __init__(self, name: str, device: torch.device, lowercase: bool) -> None:
        from transformers import AutoTokenizer, SiglipTextModel

        self.model = SiglipTextModel.from_pretrained(name).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.device, self.lowercase = device, lowercase

    def __call__(self, texts: list[str]) -> torch.Tensor:
        texts = [t.lower() for t in texts] if self.lowercase else list(texts)
        enc = self.tokenizer(
            texts,
            padding="max_length",
            max_length=SIGLIP_TEXT_LEN,
            truncation=True,
            return_tensors="pt",
        )
        return self.model(input_ids=enc["input_ids"].to(self.device)).pooler_output


def build_siglip(name: str, size: int, args: argparse.Namespace, cache: dict[str, Any]) -> Encoder:
    from transformers import SiglipVisionModel

    vision = SiglipVisionModel.from_pretrained(name).to(args.device).eval()
    if name not in cache:
        cache[name] = SiglipTextEncoder(name, args.device, args.lowercase)
    return Encoder(
        name=name,
        transform=square_transform(size, T.InterpolationMode.BILINEAR, normalize=True),
        encode_images=lambda px: vision(pixel_values=px).pooler_output,
        encode_texts=cache[name],
    )


class EncoderUnavailable(RuntimeError):
    """The encoder cannot be scored zero-shot with what the checkpoint ships; it is skipped."""


def build_radio(args: argparse.Namespace, cache: dict[str, Any]) -> Encoder:
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(RADIO_V4, trust_remote_code=True, revision=args.radio_revision)
    if "siglip2-g" not in (getattr(cfg, "adaptor_configs", None) or {}):
        # the Hugging Face build ships the backbone only (adaptor_configs is empty); the teacher
        # adaptor heads come with the torch.hub release. Training uses the backbone features.
        raise EncoderUnavailable(
            f"{RADIO_V4} on the Hugging Face Hub has no 'siglip2-g' adaptor head (adaptor_configs is "
            "empty), so its features cannot be compared with SigLIP 2 text zero-shot"
        )
    cfg.adaptor_names = ["siglip2-g"]
    model = AutoModel.from_pretrained(
        RADIO_V4, config=cfg, trust_remote_code=True, revision=args.radio_revision
    )
    model = model.to(args.device).eval()
    size = int(model.get_nearest_supported_resolution(args.radio_size, args.radio_size)[0])
    adaptor = model.adaptors["siglip2-g"]

    def encode_images(px: torch.Tensor) -> torch.Tensor:
        summary, _ = model(px)["siglip2-g"]
        return summary

    if args.radio_text == "adaptor":

        def encode_texts(texts: list[str]) -> torch.Tensor:
            tokens = adaptor.tokenizer([t.lower() if args.lowercase else t for t in texts])
            return adaptor.encode_text(tokens.to(args.device), normalize=True)

    else:
        if SIGLIP2_G not in cache:
            cache[SIGLIP2_G] = SiglipTextEncoder(SIGLIP2_G, args.device, args.lowercase)
        encode_texts = cache[SIGLIP2_G]
    return Encoder(
        name=f"{RADIO_V4}+siglip2-g ({args.radio_text} text)",
        transform=square_transform(size, T.InterpolationMode.BICUBIC, normalize=False),
        encode_images=encode_images,
        encode_texts=encode_texts,
    )


ENCODER_BUILDERS: dict[str, Callable[[argparse.Namespace, dict[str, Any]], Encoder]] = {
    "siglip2-g": lambda args, cache: build_siglip(SIGLIP2_G, 384, args, cache),
    "siglip2-so400m": lambda args, cache: build_siglip(SIGLIP2_SO400M, 384, args, cache),
    "radio": build_radio,
}


def _autocast(device: torch.device) -> Any:
    return torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda")


def embed_split(
    enc: Encoder, ds: RetrievalEvalDataset, args: argparse.Namespace
) -> dict[str, float]:
    def stack(items: list[dict[str, Any]]) -> torch.Tensor:
        return torch.stack([it["pixel_values"] for it in items])

    loader = DataLoader(
        ds.images, batch_size=args.batch_size, collate_fn=stack, num_workers=args.workers
    )
    img, txt = [], []
    with torch.inference_mode(), _autocast(args.device):
        for px in loader:
            img.append(enc.encode_images(px.to(args.device)).float().cpu())
        for i in range(0, len(ds.captions), args.batch_size):
            txt.append(enc.encode_texts(ds.captions[i : i + args.batch_size]).float().cpu())
    return retrieval_metrics(torch.cat(img), torch.cat(txt), ds)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/pretrain/stage2_contrastive.yaml")
    ap.add_argument("--splits", default="ktvic_dev,uitviic_dev")
    ap.add_argument("--encoders", default="siglip2-g,siglip2-so400m,radio")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-lowercase", dest="lowercase", action="store_false")
    ap.add_argument("--radio-size", type=int, default=512)
    ap.add_argument("--radio-revision", default=None, help="pin C-RADIOv4's remote code")
    ap.add_argument("--radio-text", choices=["siglip2-g", "adaptor"], default="siglip2-g")
    ap.add_argument("--out", default="runs/probe_vision.json")
    ap.add_argument("overrides", nargs="*", help="key.path=value config overrides (dataset roots)")
    args = ap.parse_args(argv)
    args.device = torch.device(args.device)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config, args.overrides)
    splits = [s for s in args.splits.split(",") if s]
    names = [e for e in args.encoders.split(",") if e]
    unknown = [s for s in splits if s not in cfg.data.evals] + [
        e for e in names if e not in ENCODER_BUILDERS
    ]
    if unknown:
        ap.error(f"unknown split / encoder: {unknown}")
    records = {s: build_records(cfg.data.evals[s], write_dropped=False) for s in splits}

    cache: dict[str, Any] = {}
    results: dict[str, dict[str, dict[str, float]]] = {s: {} for s in splits}
    skipped: dict[str, str] = {}
    for name in names:
        try:
            enc = ENCODER_BUILDERS[name](args, cache)
        except EncoderUnavailable as exc:
            skipped[name] = str(exc)
            print(f"skipped {name}: {exc}")
            continue
        for split in splits:
            recs, adapter = records[split]
            src = cfg.data.evals[split]
            ds = RetrievalEvalDataset(recs, adapter.image_store(src.split), enc.transform)
            results[split][name] = embed_split(enc, ds, args)
            m = results[split][name]
            print(
                f"{split:12s} {name:16s} i2t R@1/5/10 {m['i2t_r1']:.1f}/{m['i2t_r5']:.1f}/"
                f"{m['i2t_r10']:.1f}  t2i R@1/5/10 {m['t2i_r1']:.1f}/{m['t2i_r5']:.1f}/"
                f"{m['t2i_r10']:.1f}  rsum {m['rsum']:.1f}  ({len(ds.image_ids)} images, "
                f"{len(ds.captions)} captions)"
            )
        del enc
        if args.device.type == "cuda":
            torch.cuda.empty_cache()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
