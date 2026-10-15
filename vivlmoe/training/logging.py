"""Rank-aware Python logging and the scalar :class:`MetricsWriter` (TensorBoard / W&B / Null).

Scalar naming: ``train/loss``, ``train/<task metric>``, ``train/grad_norm``, ``lr/<group>``,
``throughput/samples_per_s``, ``memory/peak_gb`` and ``<split>/<metric>`` for evaluations
(so ``eval.selection_metric`` names a logged scalar verbatim).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Protocol

from vivlmoe.config.schema import LoggingCfg
from vivlmoe.training.distributed import DistContext

_FORMAT = "%(asctime)s %(levelname)s [rank %(rank)s] %(name)s: %(message)s"


class _RankFilter(logging.Filter):
    def __init__(self, rank: int) -> None:
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        return True


def setup_logging(ctx: DistContext, run_dir: Path | None = None, level: int = logging.INFO) -> None:
    """Rank 0: INFO to stderr and ``<run_dir>/train.log``; other ranks: WARNING to stderr."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter(_FORMAT)
    rank_filter = _RankFilter(ctx.rank)
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level if ctx.is_main else logging.WARNING)
    console.setFormatter(formatter)
    console.addFilter(rank_filter)
    root.addHandler(console)
    if run_dir is not None and ctx.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        fh.addFilter(rank_filter)
        root.addHandler(fh)


class MetricsWriter(Protocol):
    def log_scalars(self, scalars: dict[str, float], step: int) -> None: ...

    def log_text(self, tag: str, text: str, step: int) -> None: ...

    def close(self) -> None: ...


class NullWriter:
    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        return None

    def log_text(self, tag: str, text: str, step: int) -> None:
        return None

    def close(self) -> None:
        return None


class TensorBoardWriter:
    def __init__(self, log_dir: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._w = SummaryWriter(log_dir=str(log_dir))

    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        for k, v in scalars.items():
            self._w.add_scalar(k, v, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        self._w.add_text(tag, text, step)

    def close(self) -> None:
        self._w.flush()
        self._w.close()


class WandbWriter:
    def __init__(
        self, cfg: LoggingCfg, run_name: str, run_dir: Path, config: dict[str, Any]
    ) -> None:
        try:
            import wandb
        except ImportError as exc:  # optional extra
            raise ImportError("logging.wandb.enabled requires `pip install wandb`") from exc
        self._wandb = wandb
        self._run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            tags=list(cfg.wandb.tags),
            name=run_name,
            dir=str(run_dir),
            config=config,
            resume="allow",
        )

    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        self._wandb.log(scalars, step=step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        self._wandb.log({tag: self._wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def close(self) -> None:
        self._run.finish()


class MultiWriter:
    def __init__(self, writers: list[MetricsWriter]) -> None:
        self.writers = writers

    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        for w in self.writers:
            w.log_scalars(scalars, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        for w in self.writers:
            w.log_text(tag, text, step)

    def close(self) -> None:
        for w in self.writers:
            w.close()


def build_metrics_writer(
    cfg: LoggingCfg, ctx: DistContext, run_dir: Path, run_name: str, config: dict[str, Any]
) -> MetricsWriter:
    """Rank 0 gets TensorBoard (+ W&B when enabled); other ranks a :class:`NullWriter`."""
    if not ctx.is_main:
        return NullWriter()
    writers: list[MetricsWriter] = []
    if cfg.tensorboard:
        writers.append(TensorBoardWriter(run_dir / "tb"))
    if cfg.wandb.enabled:
        writers.append(WandbWriter(cfg, run_name, run_dir, config))
    if not writers:
        return NullWriter()
    return writers[0] if len(writers) == 1 else MultiWriter(writers)


__all__ = [
    "MetricsWriter",
    "MultiWriter",
    "NullWriter",
    "TensorBoardWriter",
    "WandbWriter",
    "build_metrics_writer",
    "setup_logging",
]
