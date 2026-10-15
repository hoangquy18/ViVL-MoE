from __future__ import annotations

from pathlib import Path

import torch

from tests.unit.data.conftest import write_images
from vivlmoe.config.schema import DataCfg, DataSourceCfg
from vivlmoe.data.collators import ContrastiveCollator
from vivlmoe.data.datasets import ImageTextDataset
from vivlmoe.data.loaders import build_eval_loader, build_train_loader, move_to_device
from vivlmoe.data.records import ColumnarImageText


def test_move_to_device_recursive_keeps_strings() -> None:
    batch = {
        "a": torch.zeros(2),
        "ids": ["x", "y"],
        "nested": {"t": (torch.ones(1), "s")},
        "lst": [torch.zeros(1), 3],
    }
    out = move_to_device(batch, "cpu")
    assert out["ids"] == ["x", "y"] and out["nested"]["t"][1] == "s" and out["lst"][1] == 3
    assert isinstance(out["nested"]["t"], tuple) and out["a"].device.type == "cpu"


def _dataset(tmp_path: Path, tiny_transform, n_images: int = 6, caps: int = 2) -> ImageTextDataset:
    from vivlmoe.data.image_store import LocalImageStore

    names = [f"{i}.jpg" for i in range(n_images)]
    write_images(tmp_path, names, size=20)
    recs = ColumnarImageText(
        [f"{i}.jpg" for i in range(n_images) for _ in range(caps)],
        [f"cap {i} {c}" for i in range(n_images) for c in range(caps)],
    )
    return ImageTextDataset(recs, LocalImageStore(tmp_path), tiny_transform)


def test_train_loader_grouping_and_resume(tmp_path: Path, tiny_transform, bundle) -> None:
    ds = _dataset(tmp_path, tiny_transform)
    cfg = DataCfg(batch_size=2, num_workers=0, pin_memory=False, persistent_workers=False)
    loader = build_train_loader(
        ds, ContrastiveCollator(bundle.vi, 8), cfg, world_size=2, rank=0, seed=1, groups=ds.groups()
    )
    assert len(loader) == 1  # 6 images / 2 ranks / batch 2 (drop_last)
    batches = list(loader)
    assert batches[0]["pixel_values"].shape == (2, 3, 32, 32)
    assert len(set(batches[0]["image_ids"])) == 2  # one caption per image within a batch
    loader.sampler.set_epoch(0)
    shard = list(loader.sampler)  # 3 samples per rank
    first = list(loader)[0]["uids"]
    assert first == shard[:2]
    loader.sampler.skip_first = 1
    assert [b["uids"] for b in loader] == [shard[1:3]]  # resume mid-epoch
    assert list(loader)[0]["uids"] == first  # skip consumed once, epoch reproducible
    cfg.one_caption_per_image = False
    plain = build_train_loader(
        ds, ContrastiveCollator(bundle.vi, 8), cfg, world_size=1, rank=0, seed=1, groups=ds.groups()
    )
    assert len(plain) == 6


def test_eval_loader_covers_all_uids_across_ranks(tmp_path: Path, tiny_transform, bundle) -> None:
    ds = _dataset(tmp_path, tiny_transform, n_images=5, caps=1)
    cfg = DataCfg(eval_batch_size=2, num_workers=0, pin_memory=False, persistent_workers=False)
    uids = []
    for rank in range(2):
        for batch in build_eval_loader(
            ds, ContrastiveCollator(bundle.vi, 8), cfg, world_size=2, rank=rank
        ):
            uids.extend(batch["uids"])
    assert sorted(uids) == list(range(5))


def test_worker_loader_spawn(tmp_path: Path, tiny_transform, bundle) -> None:
    ds = _dataset(tmp_path, tiny_transform, n_images=4, caps=1)
    cfg = DataCfg(
        batch_size=2, num_workers=1, pin_memory=False, persistent_workers=False, prefetch_factor=1
    )
    src = DataSourceCfg(kind="synthetic_pairs")
    assert src.kind == "synthetic_pairs"
    loader = build_train_loader(ds, ContrastiveCollator(bundle.vi, 8), cfg, seed=0)
    assert loader.num_workers == 1
    assert sum(b["pixel_values"].shape[0] for b in loader) == 4
