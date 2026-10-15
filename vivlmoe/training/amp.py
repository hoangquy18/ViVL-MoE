"""Mixed precision: bf16 autocast by default, fp16 with a ``GradScaler``, disabled on CPU."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Literal

import torch

from vivlmoe.config.schema import AmpCfg

log = logging.getLogger(__name__)


def _make_scaler(enabled: bool) -> Any:
    if hasattr(torch.amp, "GradScaler"):  # torch >= 2.3
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


@dataclass
class AmpState:
    enabled: bool
    dtype: torch.dtype | None
    device_type: str
    scaler: Any  # GradScaler (disabled unless fp16 on CUDA); always safe to call
    nonfinite_policy: Literal["skip", "raise"]  # fp16: skip the step & back off; else hard error

    def autocast(self) -> contextlib.AbstractContextManager[Any]:
        if not self.enabled:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.dtype)

    def describe(self) -> str:
        if not self.enabled:
            return "amp disabled (fp32)"
        return f"amp {self.dtype} on {self.device_type} (scaler={'on' if self.scaler.is_enabled() else 'off'})"


def build_amp(cfg: AmpCfg, device: torch.device) -> AmpState:
    if cfg.enabled and device.type != "cuda":
        log.warning(
            "train.amp requested (%s) but device is %s: AMP disabled", cfg.dtype, device.type
        )
        return AmpState(False, None, device.type, _make_scaler(False), "raise")
    if not cfg.enabled:
        return AmpState(False, None, device.type, _make_scaler(False), "raise")
    if cfg.dtype == "fp16":
        return AmpState(True, torch.float16, "cuda", _make_scaler(True), "skip")
    return AmpState(True, torch.bfloat16, "cuda", _make_scaler(False), "raise")


__all__ = ["AmpState", "build_amp"]
