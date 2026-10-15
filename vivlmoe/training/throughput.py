"""Throughput (samples / tokens per second) and peak-memory bookkeeping for the log line."""

from __future__ import annotations

import time

import torch


class ThroughputMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._t0 = time.perf_counter()
        self.samples = 0
        self.tokens = 0
        self.steps = 0

    def update(self, samples: int, tokens: int = 0, steps: int = 1) -> None:
        self.samples += samples
        self.tokens += tokens
        self.steps += steps

    def snapshot(self) -> dict[str, float]:
        elapsed = max(time.perf_counter() - self._t0, 1e-9)
        out = {"samples_per_s": self.samples / elapsed, "steps_per_s": self.steps / elapsed}
        if self.tokens:
            out["tokens_per_s"] = self.tokens / elapsed
        return out


def peak_memory_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1024**3


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


__all__ = ["ThroughputMeter", "peak_memory_gb", "reset_peak_memory"]
