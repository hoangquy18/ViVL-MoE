"""A Task around the real tiny ``ViVLBackbone`` (configs/model/tiny.yaml) for Trainer tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from tests.unit.training.synthetic_task import make_cfg
from vivlmoe.config.loader import load_yaml_with_includes
from vivlmoe.config.schema import RunConfig
from vivlmoe.tasks.base import LoaderBundle, Task
from vivlmoe.tasks.evaluators.base import Evaluator
from vivlmoe.tasks.registry import TASKS
from vivlmoe.training.distributed import DistContext
from vivlmoe.training.state import StepContext, StepOutput

TINY_YAML = Path(__file__).resolve().parents[3] / "configs" / "model" / "tiny.yaml"


class PairDataset(Dataset):
    def __init__(self, n: int, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.pixels = torch.randn(n, 3, 32, 32, generator=g)
        self.ids = torch.randint(3, 1000, (n, 6), generator=g)

    def __len__(self) -> int:
        return self.pixels.shape[0]

    def __getitem__(self, i: int) -> dict[str, Any]:
        return {"pixel_values": self.pixels[i], "input_ids": self.ids[i], "uid": i}


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    ids = torch.stack([it["input_ids"] for it in items])
    return {
        "pixel_values": torch.stack([it["pixel_values"] for it in items]),
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "uids": [int(it["uid"]) for it in items],
        "image_ids": [f"img{it['uid']}" for it in items],
    }


class SequentialSampler(Sampler[int]):
    def __init__(self, n: int) -> None:
        self.n, self.skip_first = n, 0

    def set_epoch(self, epoch: int) -> None:
        return None

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[int]:
        skip, self.skip_first = self.skip_first, 0
        return iter(range(skip, self.n))


@TASKS.register("tiny_backbone")
class TinyBackboneTask(Task):
    """Contrastive-style training of the tiny backbone; no eval splits."""

    name = "tiny_backbone"

    def __init__(self, cfg: RunConfig, ctx: DistContext, n_train: int = 8) -> None:
        super().__init__(cfg, ctx)
        self.n_train = n_train

    def build_model(self) -> nn.Module:
        from vivlmoe.models.factory import build_task_model

        return build_task_model("backbone", self.cfg.model)

    def build_loaders(self) -> LoaderBundle:
        ds = PairDataset(self.n_train)
        loader = DataLoader(
            ds,
            batch_size=self.cfg.data.batch_size,
            sampler=SequentialSampler(len(ds)),
            collate_fn=collate,
            num_workers=0,
        )
        return LoaderBundle(train=loader, evals={})

    def train_step(self, model: nn.Module, batch: dict[str, Any], sc: StepContext) -> StepOutput:
        out = model(batch, "contrastive")
        logits = out.logit_scale * out.image_embeds @ out.text_embeds.t()
        target = torch.arange(logits.shape[0], device=logits.device)
        loss = nn.functional.cross_entropy(logits, target)
        return StepOutput(loss=loss, batch_size=len(batch["uids"]))

    def evaluators(self) -> dict[str, Evaluator]:
        return {}

    def selection_metric(self) -> tuple[str | None, str]:
        return None, "max"


def make_backbone_cfg(run_dir: Path, frozen_dtype: str = "auto", **overrides: Any) -> RunConfig:
    tiny = load_yaml_with_includes(TINY_YAML)["model"]
    tiny["vision"]["frozen_dtype"] = frozen_dtype
    base: dict[str, Any] = {
        "task": {"name": "tiny_backbone"},
        "model": tiny,
        "data": {"batch_size": 4, "evals": {}},
        "eval": {"selection_metric": None},
        "optim": {"freeze_preset": "stage2"},
        "train": {"max_steps": 2},
    }
    from vivlmoe.config import deep_merge

    return make_cfg(run_dir, **deep_merge(base, overrides))


__all__ = ["TinyBackboneTask", "make_backbone_cfg"]
