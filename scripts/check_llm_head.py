#!/usr/bin/env python
"""Sanity-check the v2 LLM head on the training box before relying on it.

    python scripts/check_llm_head.py [--model Qwen/Qwen3.5-0.8B] [--llm-class Qwen3_5ForCausalLM] \
        [--device cuda] [--max-ppl 200]
    python scripts/check_llm_head.py --tiny qwen3_next --device cpu      # CPU smoke, random weights

The LLM is loaded exactly as training loads it (``vivlmoe.models.heads.llm_loader.load_causal_lm``),
then checked; the exit code is non-zero if any check fails:

1. no missing weights (``load_causal_lm`` raises, catching transformers#49324);
2. perplexity on a few Vietnamese sentences is below ``--max-ppl`` (a randomly initialised language
   model or ``lm_head`` scores around the vocabulary size); skipped for ``--tiny``;
3. one backward pass with prefix embeddings gives every parameter a gradient (training runs DDP with
   ``find_unused_parameters=False``, so leftovers such as unused heads must be frozen or removed);
4. greedy generation from a left-padded ``inputs_embeds`` batch equals per-sample generation (the head
   left-pads and Qwen3.5's linear-attention layers cannot tolerate holes in a sequence);
5. whether the fast linear-attention kernels are installed (flash-linear-attention, causal-conv1d).
"""

from __future__ import annotations

import argparse
import importlib
import logging
import math
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vivlmoe.models.heads.llm_loader import MissingWeightsError, load_causal_lm  # noqa: E402

VI_SENTENCES = [
    "Một người đàn ông đang đạp xe trên con đường ven biển vào buổi sáng.",
    "Hai đứa trẻ đang chơi đá bóng trên bãi cỏ trước nhà.",
    "Chợ nổi Cái Răng là một điểm du lịch nổi tiếng ở Cần Thơ.",
    "Bát phở bò nóng hổi được bày trên chiếc bàn gỗ.",
]
TINY_CONFIGS: dict[str, dict[str, Any]] = {
    "qwen3": dict(
        vocab_size=1000, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
        pad_token_id=0, eos_token_id=1, bos_token_id=1,
    ),
    "qwen3_next": dict(
        vocab_size=1000, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, linear_num_value_heads=4,
        linear_num_key_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, mlp_only_layers=[0, 1, 2, 3], num_experts=0,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        max_position_embeddings=256, pad_token_id=0, eos_token_id=1, bos_token_id=1,
    ),
}  # fmt: skip


def check_perplexity(
    model: Any, tokenizer: Any, device: torch.device, max_ppl: float
) -> tuple[bool, str]:
    enc = tokenizer(VI_SENTENCES, return_tensors="pt", padding=True)
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    labels = ids.masked_fill(mask == 0, -100)
    with (
        torch.no_grad(),
        torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"),
    ):
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
    ppl = math.exp(float(loss))
    return (
        ppl < max_ppl,
        f"perplexity {ppl:.1f} on {len(VI_SENTENCES)} Vietnamese sentences (max {max_ppl})",
    )


def check_gradients(model: Any, device: torch.device) -> tuple[bool, str]:
    """Prefix embeddings + tokens -> loss on the last tokens -> every parameter has a gradient."""
    model.train()
    model.zero_grad(set_to_none=True)
    emb = model.get_input_embeddings()
    vocab = int(emb.weight.shape[0])
    ids = torch.randint(2, min(vocab, 30000), (2, 12), device=device)
    prefix = torch.randn(2, 6, emb.weight.shape[1], device=device)
    inputs = torch.cat([prefix, emb(ids)], dim=1)
    labels = torch.full(inputs.shape[:2], -100, dtype=torch.long, device=device)
    labels[:, -6:] = ids[:, -6:]
    with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
        loss = model(inputs_embeds=inputs, labels=labels).loss
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    model.zero_grad(set_to_none=True)
    model.eval()
    if missing:
        return False, f"{len(missing)} trainable parameters got no gradient, e.g. {missing[:4]}"
    return True, "every trainable parameter received a gradient"


def check_batched_generation(model: Any, device: torch.device, pad_id: int) -> tuple[bool, str]:
    """Greedy generation from a left-padded ``inputs_embeds`` batch == per-sample generation."""
    emb = model.get_input_embeddings()
    vocab = int(emb.weight.shape[0])
    gen = torch.Generator().manual_seed(0)
    seqs = [torch.randint(2, min(vocab, 30000), (n,), generator=gen) for n in (5, 11, 3)]
    kw = dict(max_new_tokens=8, do_sample=False, pad_token_id=pad_id, eos_token_id=None)
    width = max(len(s) for s in seqs)
    embeds = torch.zeros(len(seqs), width, emb.weight.shape[1], device=device)
    mask = torch.zeros(len(seqs), width, dtype=torch.long, device=device)
    with torch.no_grad():
        single = []
        for i, s in enumerate(seqs):
            e = emb(s.to(device))
            embeds[i, width - len(s) :], mask[i, width - len(s) :] = e, 1
            one = torch.ones(1, len(s), dtype=torch.long, device=device)
            single.append(
                model.generate(inputs_embeds=e[None], attention_mask=one, **kw)[0].tolist()
            )
        batched = model.generate(inputs_embeds=embeds, attention_mask=mask, **kw).tolist()
    ok = batched == single
    return ok, "left-padded batched generation " + (
        "matches" if ok else "DIFFERS from"
    ) + " per-sample"


def fast_path_report(model: Any) -> str:
    model_type = getattr(model.config, "model_type", "")
    for mod_name in (model_type, model_type.removesuffix("_text")):
        try:
            mod = importlib.import_module(f"transformers.models.{mod_name}.modeling_{mod_name}")
        except ImportError:
            continue
        flag = getattr(mod, "is_fast_path_available", None)
        if flag is not None:
            return f"fast linear-attention kernels available: {bool(flag)}"
    return "no linear-attention layers (fast-path check not applicable)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--llm-class", default="Qwen3_5ForCausalLM", help="'' = AutoModelForCausalLM")
    ap.add_argument("--revision", default=None)
    ap.add_argument(
        "--tiny", choices=sorted(TINY_CONFIGS), default=None, help="random tiny model (CPU smoke)"
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-ppl", type=float, default=200.0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    device = torch.device(args.device)
    torch.manual_seed(0)

    results: list[tuple[str, bool, str]] = []
    try:
        if args.tiny:
            model = load_causal_lm(None, tiny_config=TINY_CONFIGS[args.tiny], tiny_arch=args.tiny)
        else:
            model = load_causal_lm(
                args.model, llm_class=args.llm_class or None, revision=args.revision
            )
    except (MissingWeightsError, ImportError) as exc:
        print(f"FAIL load: {exc}")
        return 1
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    results.append(
        (
            "load",
            True,
            f"{type(model).__name__}, {n_params / 1e6:.1f}M parameters, no missing weights",
        )
    )

    pad_id = model.config.pad_token_id
    if args.tiny:
        results.append(("perplexity", True, "skipped (random tiny model)"))
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.pad_token_id if pad_id is None else pad_id
        results.append(("perplexity", *check_perplexity(model, tokenizer, device, args.max_ppl)))
    results.append(("gradients", *check_gradients(model, device)))
    results.append(
        ("batched generation", *check_batched_generation(model, device, int(pad_id or 0)))
    )
    results.append(("fast path", True, fast_path_report(model)))

    for name, ok, msg in results:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {msg}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
