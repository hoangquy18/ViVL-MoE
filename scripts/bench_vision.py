#!/usr/bin/env python
"""Time the frozen vision tower of a config without any data loading: images/s and the costliest ops.

    python scripts/bench_vision.py --config configs/v2/align.yaml [--batch 64] [--iters 5] [--device cuda]

Random pixels at the tower's resolution, weights in the dtype training gives them (``frozen_dtype``, under
autocast on CUDA: bf16, or fp16 on GPUs without native bf16 like ``scripts/gpu/env.sh`` picks). Next to ``scripts/bench_loader.py`` (images/s the DataLoader delivers) it tells a
GPU-bound run from a data-bound one: every optimizer step needs ``batch_size x grad_accum_steps`` images
from both.
"""

from __future__ import annotations

import argparse
import contextlib
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from vivlmoe.config import load_config  # noqa: E402
from vivlmoe.models.towers import build_vision_tower  # noqa: E402


def _top_ops(tower: torch.nn.Module, x: torch.Tensor, amp: Callable[[], Any], rows: int) -> str:
    """Profiler table of one forward, the ops with the most time first."""
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if x.is_cuda else [])
    with profile(activities=activities) as prof, torch.no_grad(), amp():
        tower(x)
        if x.is_cuda:
            torch.cuda.synchronize()
    keys = ("self_device_time_total", "self_cuda_time_total") if x.is_cuda else ()
    for key in (*keys, "self_cpu_time_total"):  # the key name changed across torch versions
        try:
            return prof.key_averages().table(sort_by=key, row_limit=rows)
        except (AttributeError, KeyError):
            continue
    return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", type=int, default=64, help="images per forward (C-RADIO runs 64)")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--dtype", choices=["auto", "bf16", "fp16"], default="auto", help="auto: bf16 if native"
    )
    ap.add_argument("--profile-rows", type=int, default=15, help="0 = no profiler table")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA GPU visible (--device cpu checks the script itself)")

    cfg = load_config(args.config, args.overrides)
    tower = build_vision_tower(cfg.model.vision).to(device)
    cuda = device.type == "cuda"
    amp_dtype = None
    if cuda:  # bf16 needs compute capability 8.0 (Ampere); before that it is emulated
        native_bf16 = torch.cuda.get_device_capability(device)[0] >= 8
        choice = args.dtype if args.dtype != "auto" else ("bf16" if native_bf16 else "fp16")
        amp_dtype = torch.bfloat16 if choice == "bf16" else torch.float16
    dtype = tower.apply_dtype_policy(amp_dtype)

    def amp() -> Any:
        return torch.autocast("cuda", dtype=amp_dtype) if cuda else contextlib.nullcontext()

    gpu = f"{torch.cuda.get_device_name(device)}, CUDA {torch.version.cuda}" if cuda else "CPU"
    print(f"torch {torch.__version__} on {gpu}; {cfg.model.vision.name} weights {dtype}")
    if cuda:
        b = torch.backends.cuda
        print(
            f"SDPA enabled: flash={b.flash_sdp_enabled()} mem_efficient={b.mem_efficient_sdp_enabled()}"
            f" math={b.math_sdp_enabled()}"
        )
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    x = torch.rand(args.batch, 3, tower.image_size, tower.image_size, device=device)
    times = []
    with torch.no_grad(), amp():
        tower(x)  # warm-up: kernel selection, lazy initialisation
        sync()
        for _ in range(args.iters):
            start = time.perf_counter()
            tower(x)
            sync()
            times.append(time.perf_counter() - start)
    best, median = min(times), statistics.median(times)
    print(
        f"vision tower {tower.image_size}px, {args.batch} images per forward: best {best * 1e3:.0f} ms, "
        f"median {median * 1e3:.0f} ms -> {args.batch / median:.1f} images/s"
    )
    if cuda:
        print(f"peak GPU memory {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB")
    if args.profile_rows > 0:
        print(_top_ops(tower, x, amp, args.profile_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
