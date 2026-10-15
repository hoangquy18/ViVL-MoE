from __future__ import annotations

import pytest
from torch.utils.data import DistributedSampler

from vivlmoe.data.samplers import (
    OneCaptionPerImageDistributedSampler,
    OrderedShardSampler,
    build_train_sampler,
)


def _groups(n_images: int, caps: int) -> list[list[int]]:
    return [[i * caps + c for c in range(caps)] for i in range(n_images)]


def test_one_caption_per_image_invariants() -> None:
    groups = _groups(11, 3)
    owner = {idx: g for g, grp in enumerate(groups) for idx in grp}
    world = 4
    samplers = [
        OneCaptionPerImageDistributedSampler(groups, world, r, seed=3) for r in range(world)
    ]
    shards = [list(s) for s in samplers]
    assert all(len(sh) == 2 == len(samplers[0]) for sh in shards)  # 11 // 4
    flat = [i for sh in shards for i in sh]
    assert len(set(flat)) == len(flat)  # disjoint ranks
    assert len({owner[i] for i in flat}) == len(flat)  # one caption per image
    for s in samplers:
        s.set_epoch(1)
    epoch1 = [i for s in samplers for i in s]
    assert epoch1 != flat
    for s in samplers:
        s.set_epoch(0)
    assert [i for s in samplers for i in s] == flat  # reproducible
    no_drop = OneCaptionPerImageDistributedSampler(groups, world, 0, seed=3, drop_last=False)
    assert len(no_drop) == 3


def test_skip_first_consumed_once() -> None:
    s = OneCaptionPerImageDistributedSampler(_groups(8, 2), 2, 1, seed=0)
    full = list(s)
    s.skip_first = 3
    assert list(s) == full[3:] and s.skip_first == 0
    assert list(s) == full
    s.skip_first = 99
    with pytest.raises(ValueError):
        list(s)
    with pytest.raises(ValueError):
        OneCaptionPerImageDistributedSampler([[0], []], 1, 0)
    with pytest.raises(ValueError):
        OneCaptionPerImageDistributedSampler([[0]], 2, 0)


def test_singleton_groups_match_distributed_sampler_semantics() -> None:
    """Singleton groups yield exactly ``DistributedSampler``'s per-rank index sequences (our
    generator is seeded ``seed * 1_000_003 + epoch``, so the reference uses ``seed * 1_000_003``)."""
    n, world, seed = 13, 3, 5
    ours = [build_train_sampler(n, world, r, seed=seed, groups=None) for r in range(world)]
    ref = [
        DistributedSampler(range(n), world, r, shuffle=True, seed=seed * 1_000_003, drop_last=True)
        for r in range(world)
    ]
    for epoch in (0, 1, 7):
        for o, r in zip(ours, ref, strict=True):
            o.set_epoch(epoch)
            r.set_epoch(epoch)
            assert len(o) == len(r) and list(o) == list(r)
    ours_all = [i for s in ours for i in s]
    assert len(ours_all) == (n // world) * world and len(set(ours_all)) == len(ours_all)
    for o, r in zip(
        [
            OneCaptionPerImageDistributedSampler([[i] for i in range(n)], world, r, drop_last=False)
            for r in range(world)
        ],
        [
            DistributedSampler(range(n), world, r, shuffle=True, seed=0, drop_last=False)
            for r in range(world)
        ],
        strict=True,
    ):
        assert list(o) == list(r)  # wrap-around padding matches too
    no_group = build_train_sampler(6, 1, 0, 0, groups=_groups(3, 2), one_caption_per_image=False)
    assert sorted(no_group) == list(range(6))
    with pytest.raises(ValueError):
        build_train_sampler(5, 1, 0, 0, groups=_groups(2, 2))


def test_ordered_shard_sampler_covers_without_padding() -> None:
    shards = [list(OrderedShardSampler(10, 3, r)) for r in range(3)]
    assert shards == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]
    assert [len(OrderedShardSampler(10, 3, r)) for r in range(3)] == [4, 3, 3]
    assert list(OrderedShardSampler(0, 2, 1)) == []
