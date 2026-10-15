"""A tiny regression Task + dataset + samplers implementing the Trainer contracts (tests only)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from vivlmoe.config import config_from_dict, deep_merge
from vivlmoe.config.schema import RunConfig
from vivlmoe.data.loaders import move_to_device
from vivlmoe.tasks.base import LoaderBundle, Task
from vivlmoe.tasks.evaluators.base import Evaluator, gather_ordered_tensors
from vivlmoe.tasks.registry import TASKS
from vivlmoe.training.distributed import DistContext
from vivlmoe.training.state import EvalContext, StepContext, StepOutput


class SyntheticModel(nn.Module):
    def __init__(
        self, in_dim: int = 8, hidden: int = 16, dropout: float = 0.1, unused_param: bool = False
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.frozen = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Dropout(0.5)
        )
        self.head = nn.Linear(hidden, 1)
        self.scale = nn.Parameter(torch.zeros(()))
        if unused_param:
            self.unused = nn.Parameter(torch.zeros(3))

    def forward(self, batch: dict[str, Any], mode: str) -> dict[str, torch.Tensor]:
        h = self.frozen(self.body(batch["x"]))
        return {"pred": self.head(h).squeeze(-1) + self.scale}


class SyntheticDataset(Dataset):
    def __init__(self, n: int, in_dim: int, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, in_dim, generator=g)
        w = torch.randn(in_dim, generator=g)
        self.y = self.x @ w + 0.1 * torch.randn(n, generator=g)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> dict[str, Any]:
        return {"x": self.x[i], "y": self.y[i], "uid": i}


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x": torch.stack([it["x"] for it in items]),
        "y": torch.stack([it["y"] for it in items]),
        "uids": [int(it["uid"]) for it in items],
    }


class ShardSampler(Sampler[int]):
    """Seeded per-epoch shuffle, equal disjoint shards (drop_last), ``set_epoch`` + ``skip_first``."""

    def __init__(self, n: int, world: int, rank: int, seed: int, shuffle: bool = True) -> None:
        self.n, self.world, self.rank, self.seed, self.shuffle = n, world, rank, seed, shuffle
        self.epoch = 0
        self.skip_first = 0
        self.per_rank = n // world

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.per_rank

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch)
            order = torch.randperm(self.n, generator=g).tolist()
        else:
            order = list(range(self.n))
        shard = order[: self.per_rank * self.world][self.rank :: self.world]
        skip, self.skip_first = self.skip_first, 0
        return iter(shard[skip:])


class OrderedShardSampler(Sampler[int]):
    def __init__(self, n: int, world: int, rank: int) -> None:
        self.indices = list(range(rank, n, world))

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)


class SyntheticEvaluator(Evaluator):
    name = "synthetic"

    def __init__(self, ctx: DistContext, scripted_scores: list[float] | None = None) -> None:
        self.ctx = ctx
        self.scripted = list(scripted_scores or [])
        self.calls = 0

    def run(self, model: nn.Module, loader: DataLoader, ec: EvalContext) -> dict[str, float] | None:
        uids: list[int] = []
        preds: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        for batch in loader:
            batch = move_to_device(batch, ec.ctx.device)
            out = model(batch, "eval")
            uids += batch["uids"]
            preds.append(out["pred"])
            ys.append(batch["y"])
        ids = torch.tensor(uids, dtype=torch.int64)
        pred = gather_ordered_tensors(ids, torch.cat(preds) if preds else torch.zeros(0), ec.ctx)
        y = gather_ordered_tensors(ids, torch.cat(ys) if ys else torch.zeros(0), ec.ctx)
        self.calls += 1
        if not ec.ctx.is_main:
            return None
        mse = float(F.mse_loss(pred, y))
        score = self.scripted[self.calls - 1] if self.calls <= len(self.scripted) else -mse
        return {"mse": mse, "score": score}

    def state_dict(self) -> dict[str, Any]:
        return {"calls": self.calls}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.calls = int(state.get("calls", 0))


@TASKS.register("synthetic")
class SyntheticTask(Task):
    name = "synthetic"

    def __init__(
        self,
        cfg: RunConfig,
        ctx: DistContext,
        *,
        n_train: int = 64,
        n_dev: int = 16,
        in_dim: int = 8,
        dropout: float = 0.1,
        nan_micro_batches: tuple[int, ...] = (),
        inf_grad_micro_batches: tuple[int, ...] = (),
        scripted_scores: list[float] | None = None,
        unused_param: bool = False,
    ) -> None:
        super().__init__(cfg, ctx)
        self.n_train, self.n_dev, self.in_dim, self.dropout = n_train, n_dev, in_dim, dropout
        self.nan_micro_batches = set(nan_micro_batches)
        self.inf_grad_micro_batches = set(inf_grad_micro_batches)
        self.scripted_scores = scripted_scores
        self.unused_param = unused_param
        self.records: list[tuple[int, list[int], float]] = []  # (step, uids, loss) per micro-batch
        self._micro_counter = 0
        self._evaluator: SyntheticEvaluator | None = None
        self.train_mode_calls = 0

    def build_model(self) -> nn.Module:
        return SyntheticModel(self.in_dim, dropout=self.dropout, unused_param=self.unused_param)

    def always_frozen(self) -> list[str]:
        return ["frozen"]

    def build_loaders(self) -> LoaderBundle:
        d = self.cfg.data
        train = SyntheticDataset(self.n_train, self.in_dim, seed=100)
        sampler = ShardSampler(
            len(train), self.ctx.world_size, self.ctx.rank, seed=self.cfg.run.seed
        )
        train_loader = DataLoader(
            train, batch_size=d.batch_size, sampler=sampler, collate_fn=collate, num_workers=0
        )
        evals: dict[str, DataLoader] = {}
        if "dev" in d.evals:
            dev = SyntheticDataset(self.n_dev, self.in_dim, seed=200)
            evals["dev"] = DataLoader(
                dev,
                batch_size=d.eval_batch_size,
                sampler=OrderedShardSampler(len(dev), self.ctx.world_size, self.ctx.rank),
                collate_fn=collate,
                num_workers=0,
            )
        return LoaderBundle(train=train_loader, evals=evals)

    def train_step(self, model: nn.Module, batch: dict[str, Any], sc: StepContext) -> StepOutput:
        out = model(batch, "train")
        loss = F.mse_loss(out["pred"], batch["y"])
        self.records.append((sc.step, list(batch["uids"]), float(loss)))
        clean = float(loss)
        if self._micro_counter in self.nan_micro_batches:
            loss = loss * float("nan")
        if self._micro_counter in self.inf_grad_micro_batches:
            # finite loss (scale == 0) whose gradient norm overflows to inf (fp16-overflow stand-in)
            loss = loss + 3e38 * self._scale_of(model)
        self._micro_counter += 1
        return StepOutput(loss=loss, batch_size=len(batch["uids"]), metrics={"mse": clean})

    @staticmethod
    def _scale_of(model: nn.Module) -> torch.Tensor:
        raw = getattr(model, "module", model)
        return raw.scale

    def evaluators(self) -> dict[str, Evaluator]:
        if "dev" not in self.cfg.data.evals:
            return {}
        if self._evaluator is None:
            self._evaluator = SyntheticEvaluator(self.ctx, self.scripted_scores)
        return {"dev": self._evaluator}

    def selection_metric(self) -> tuple[str | None, str]:
        return ("dev/score", "max") if "dev" in self.cfg.data.evals else (None, "max")

    def on_train_mode(self, raw_model: nn.Module) -> None:
        raw_model.frozen.eval()
        self.train_mode_calls += 1

    def state_dict(self) -> dict[str, Any]:
        return {"micro_counter": self._micro_counter}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._micro_counter = int(state.get("micro_counter", 0))

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {"dev_split_sha1": "synthetic-dev"}


def make_cfg(run_dir: Path, **overrides: Any) -> RunConfig:
    """Minimal RunConfig for the synthetic task (AMP off, no TensorBoard, no cadence by default)."""
    raw: dict[str, Any] = {
        "run": {"name": "t", "dir": str(run_dir), "seed": 1},
        "task": {"name": "synthetic"},
        "model": {"vision": {"pretrained": False}, "text": {"pretrained": False}},
        "data": {
            "batch_size": 4,
            "eval_batch_size": 8,
            "num_workers": 0,
            "evals": {"dev": {"kind": "synthetic", "split": "dev"}},
        },
        "train": {"max_steps": 4, "grad_accum_steps": 1, "amp": {"enabled": False}, "log_every": 1},
        "schedule": {"kind": "constant", "warmup_ratio": 0.0},
        "eval": {"every_steps": None, "every_epochs": None, "selection_metric": "dev/score"},
        "checkpoint": {"every_steps": None, "every_epochs": None, "keep_last_n": 2},
        "logging": {"tensorboard": False},
    }
    return config_from_dict(deep_merge(raw, overrides))


__all__ = [
    "OrderedShardSampler",
    "ShardSampler",
    "SyntheticDataset",
    "SyntheticEvaluator",
    "SyntheticModel",
    "SyntheticTask",
    "collate",
    "make_cfg",
]
