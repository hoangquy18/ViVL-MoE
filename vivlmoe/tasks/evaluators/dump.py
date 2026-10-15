"""Prediction / metrics dumps shared by the downstream evaluators and ``scripts/evaluate.py``.

Layout under a run directory (plan decision 17: tokenizer + generation settings travel with the
numbers):

* ``<run_dir>/predictions/<split>_step<N>.jsonl`` - one JSON object per sample, in ``uid`` order,
  written with ``ensure_ascii=False`` so Vietnamese text stays readable.
* ``<run_dir>/metrics/<split>_step<N>.json`` - the metric dict plus provenance (tokenizer
  signatures, generation config, sample count).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def predictions_path(run_dir: str | Path, split: str, step: int) -> Path:
    return Path(run_dir) / "predictions" / f"{split}_step{step}.jsonl"


def metrics_path(run_dir: str | Path, split: str, step: int) -> Path:
    return Path(run_dir) / "metrics" / f"{split}_step{step}.json"


def write_predictions_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write ``rows`` as JSONL (UTF-8, ``ensure_ascii=False``); returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            n += 1
    log.info("wrote %d predictions to %s", n, path)
    return path


def read_predictions_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_metrics_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, default=str)
        fh.write("\n")
    log.info("wrote metrics to %s", path)
    return path


__all__ = [
    "metrics_path",
    "predictions_path",
    "read_predictions_jsonl",
    "write_metrics_json",
    "write_predictions_jsonl",
]
