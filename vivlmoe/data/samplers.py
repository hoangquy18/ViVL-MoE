"""Distributed samplers.

:class:`OneCaptionPerImageDistributedSampler` draws one caption per image per epoch (so a batch never
contains two positives of the same image, which would be false negatives for InfoNCE, Eq. 13),
shards the images equally and disjointly across ranks, and supports mid-epoch resume through
``skip_first``. With singleton groups it has exactly ``DistributedSampler(drop_last=True)``
semantics. :class:`OrderedShardSampler` is the padding-free evaluation sharding
(``rank::world``); evaluators gather by ``uid`` and assert coverage.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class OneCaptionPerImageDistributedSampler(Sampler[int]):
    """``groups[i]`` are the dataset indices (captions) of image *i*.

    Epoch *e* uses ``torch.Generator().manual_seed(seed * 1_000_003 + e)`` identically on all
    ranks: permute the groups (``randperm`` is the first draw, so with singleton groups the order
    equals ``DistributedSampler(seed=seed * 1_000_003, epoch=e)`` index for index), pick one index
    per group, truncate to a multiple of ``num_replicas`` (``drop_last=True``) or pad by wrapping
    around (``drop_last=False``), then rank *r* takes ``picks[r::num_replicas]``. ``skip_first``
    indices are dropped from the next ``__iter__`` only.
    """

    def __init__(
        self,
        groups: Sequence[Sequence[int]],
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError(f"invalid rank/num_replicas: {rank}/{num_replicas}")
        if any(len(g) == 0 for g in groups):
            raise ValueError("every group must contain at least one index")
        self.groups = [list(g) for g in groups]
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.skip_first = 0
        n = len(self.groups)
        if drop_last:
            self.total_size = (n // num_replicas) * num_replicas
        else:
            self.total_size = -(-n // num_replicas) * num_replicas
        self.num_samples = self.total_size // num_replicas
        if self.num_samples == 0:
            raise ValueError(
                f"{n} image groups cannot be sharded across {num_replicas} ranks with drop_last"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _global_order(self) -> list[int]:
        n = len(self.groups)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed * 1_000_003 + self.epoch)
            order = torch.randperm(n, generator=g).tolist()  # first draw == DistributedSampler
            within = torch.randint(0, 2**31 - 1, (n,), generator=g)
            picks = [self.groups[i][int(within[i]) % len(self.groups[i])] for i in order]
        else:
            picks = [g[0] for g in self.groups]
        if self.drop_last:
            picks = picks[: self.total_size]
        elif len(picks) < self.total_size:
            picks = picks + picks[: self.total_size - len(picks)]
        return picks

    def __iter__(self) -> Iterator[int]:
        shard = self._global_order()[self.rank :: self.num_replicas]
        skip, self.skip_first = self.skip_first, 0
        if skip:
            if skip > len(shard):
                raise ValueError(f"skip_first={skip} exceeds per-rank samples {len(shard)}")
            shard = shard[skip:]
        return iter(shard)

    def __len__(self) -> int:
        return self.num_samples


class OrderedShardSampler(Sampler[int]):
    """Indices ``rank, rank + world, ...`` without padding (shards may differ by one)."""

    def __init__(self, num_samples: int, num_replicas: int = 1, rank: int = 0) -> None:
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError(f"invalid rank/num_replicas: {rank}/{num_replicas}")
        self.total = int(num_samples)
        self.num_replicas = num_replicas
        self.rank = rank
        self.skip_first = 0

    def set_epoch(self, epoch: int) -> None:  # protocol compatibility
        pass

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.total, self.num_replicas))

    def __len__(self) -> int:
        return len(range(self.rank, self.total, self.num_replicas))


def build_train_sampler(
    num_samples: int,
    world_size: int,
    rank: int,
    seed: int,
    groups: Sequence[Sequence[int]] | None = None,
    one_caption_per_image: bool = True,
    shuffle: bool = True,
) -> OneCaptionPerImageDistributedSampler:
    """Image-grouped sampler when ``groups`` are given and enabled; else singleton groups."""
    if groups is None or not one_caption_per_image:
        groups = [[i] for i in range(num_samples)]
    else:
        covered = sorted(i for g in groups for i in g)
        if covered != list(range(num_samples)):
            raise ValueError("groups must partition range(num_samples)")
    return OneCaptionPerImageDistributedSampler(
        groups, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed, drop_last=True
    )


__all__ = ["OneCaptionPerImageDistributedSampler", "OrderedShardSampler", "build_train_sampler"]
