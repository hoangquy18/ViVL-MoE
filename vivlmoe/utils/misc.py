"""Small, dependency-free helpers shared by training, tasks and scripts."""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import nn


def now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).replace(microsecond=0).isoformat()


def format_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Plain-text table with left-aligned text columns (no external dependency)."""
    str_rows = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*["-" * w for w in widths])]
    lines += [fmt.format(*row) for row in str_rows]
    return "\n".join(lines)


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def flatten_scalars(prefix: str, values: dict[str, Any]) -> dict[str, float]:
    """``{'a': 1, 'b': {'c': 2}}`` -> ``{'p/a': 1.0, 'p/b/c': 2.0}`` (non-numeric values dropped)."""
    out: dict[str, float] = {}
    for k, v in values.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_scalars(key, v))
        elif isinstance(v, bool):
            continue
        elif isinstance(v, (int, float)):
            out[key] = float(v)
        elif isinstance(v, torch.Tensor) and v.numel() == 1:
            out[key] = float(v.item())
    return out


__all__ = ["count_parameters", "flatten_scalars", "format_table", "now_iso"]
