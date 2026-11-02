#!/usr/bin/env python
"""Sanity-check the v2 LLM head on the training box before relying on it.

    python scripts/check_llm_head.py [--model Qwen/Qwen3.5-0.8B] [--llm-class Qwen3_5ForCausalLM] \
        [--device cuda] [--amp-dtype fp16] [--max-ppl 200]
    python scripts/check_llm_head.py --tiny qwen3_next --device cpu      # CPU smoke, random weights

The LLM is loaded exactly as training loads it (``vivlmoe.models.heads.llm_loader.load_causal_lm``),
then checked under the training autocast (``--amp-dtype`` as ``train.amp.dtype``; CUDA only, like the
Trainer); the exit code is non-zero if any check fails:

1. no missing weights (``load_causal_lm`` raises, catching transformers#49324);
2. perplexity on a few Vietnamese sentences is below ``--max-ppl`` (a randomly initialised language
   model or ``lm_head`` scores around the vocabulary size) and, on CUDA, within ``--max-ppl-drift`` of
   fp32 (Qwen3.5 was trained in bf16: under fp16 an activation above 65504 overflows); skipped for
   ``--tiny``;
3. one backward pass with a left-padded batch of prefix embeddings, laid out as the head lays it out,
   gives every parameter a finite gradient (training runs DDP with ``find_unused_parameters=False``, so
   leftovers such as unused heads must be frozen or removed; the loss is unscaled, the floor that fp16's
   GradScaler backs off to);
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

from vivlmoe.models.heads.causal_lm import position_ids_from_mask  # noqa: E402
from vivlmoe.models.heads.llm_loader import MissingWeightsError, load_causal_lm  # noqa: E402

AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
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


def perplexity_verdict(
    ppl: float, fp32_ppl: float | None, amp: str, max_ppl: float, max_drift: float
) -> tuple[bool, str]:
    """``fp32_ppl`` None: ``ppl`` itself ran in fp32 (no autocast off CUDA). nan/inf fail."""
    where = f"on {len(VI_SENTENCES)} Vietnamese sentences (max {max_ppl})"
    if fp32_ppl is None:
        return ppl < max_ppl, f"perplexity {ppl:.1f} in fp32 {where}"
    drift = abs(ppl / fp32_ppl - 1)
    return (
        ppl < max_ppl and drift <= max_drift,
        f"perplexity {ppl:.1f} under {amp} autocast, {fp32_ppl:.1f} in fp32 "
        f"(drift {drift:.1%}, max {max_drift:.0%}) {where}",
    )


def check_perplexity(
    model: Any, tokenizer: Any, device: torch.device, amp: str, max_ppl: float, max_drift: float
) -> tuple[bool, str]:
    enc = tokenizer(VI_SENTENCES, return_tensors="pt", padding=True, return_attention_mask=True)
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    labels = ids.masked_fill(mask == 0, -100)

    def perplexity(autocast: bool) -> float:
        with torch.no_grad(), torch.autocast(device.type, AMP_DTYPES[amp], enabled=autocast):
            loss = float(model(input_ids=ids, attention_mask=mask, labels=labels).loss)
        return math.inf if loss > 700 else math.exp(loss)  # exp() raises above ~709; nan stays nan

    fp32_ppl = perplexity(False)
    if device.type != "cuda":
        return perplexity_verdict(fp32_ppl, None, amp, max_ppl, max_drift)
    return perplexity_verdict(perplexity(True), fp32_ppl, amp, max_ppl, max_drift)


def check_gradients(model: Any, device: torch.device, amp: str) -> tuple[bool, str]:
    """Left-padded prefix embeddings + tokens -> loss on the last tokens under the training autocast
    -> every parameter has a finite gradient."""
    model.train()
    model.zero_grad(set_to_none=True)
    emb = model.get_input_embeddings()
    vocab = int(emb.weight.shape[0])
    ids = torch.randint(2, min(vocab, 30000), (2, 12), device=device)
    prefix = torch.randn(2, 6, emb.weight.shape[1], device=device)
    mask = torch.ones(2, 18, dtype=torch.long, device=device)
    mask[1, :4] = 0  # the second sample is 4 positions shorter; pads are zero rows, as in the head
    inputs = torch.cat([prefix, emb(ids)], dim=1) * mask[..., None]
    labels = torch.full(inputs.shape[:2], -100, dtype=torch.long, device=device)
    labels[:, -6:] = ids[:, -6:]
    with torch.autocast(device.type, AMP_DTYPES[amp], enabled=device.type == "cuda"):
        loss = model(
            inputs_embeds=inputs,
            attention_mask=mask,
            position_ids=position_ids_from_mask(mask),
            labels=labels,
        ).loss
    loss.backward()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    missing = [n for n, p in trainable if p.grad is None]
    nonfinite = [n for n, p in trainable if p.grad is not None and not p.grad.isfinite().all()]
    model.zero_grad(set_to_none=True)
    model.eval()
    if missing:
        return False, f"{len(missing)} trainable parameters got no gradient, e.g. {missing[:4]}"
    if nonfinite:
        return False, f"{len(nonfinite)} parameters got a non-finite gradient, e.g. {nonfinite[:4]}"
    under = f" under {amp} autocast" if device.type == "cuda" else ""
    return True, f"every trainable parameter received a finite gradient{under}"


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
    ap.add_argument(
        "--amp-dtype",
        choices=sorted(AMP_DTYPES),
        default="bf16",
        help="autocast dtype on CUDA, as train.amp.dtype (fp16 on GPUs without native bf16)",
    )
    ap.add_argument("--max-ppl", type=float, default=200.0)
    ap.add_argument(
        "--max-ppl-drift",
        type=float,
        default=0.05,
        help="largest relative change of the perplexity under autocast from fp32",
    )
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
        verdict = check_perplexity(
            model, tokenizer, device, args.amp_dtype, args.max_ppl, args.max_ppl_drift
        )
        results.append(("perplexity", *verdict))
    results.append(("gradients", *check_gradients(model, device, args.amp_dtype)))
    results.append(
        ("batched generation", *check_batched_generation(model, device, int(pad_id or 0)))
    )
    results.append(("fast path", True, fast_path_report(model)))

    for name, ok, msg in results:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {msg}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
