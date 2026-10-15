from __future__ import annotations

import pytest
import torch

from vivlmoe.metrics.retrieval import recall_at_k

# 3 images / 7 captions; image 0 -> captions {0,1}, image 1 -> {2,3,4}, image 2 -> {5,6}.
SIM = torch.tensor(
    [
        [0.9, 0.2, 0.8, 0.1, 0.0, 0.3, 0.4],  # best caption 0 (positive): rank 0
        [0.7, 0.6, 0.5, 0.15, 0.2, 0.0, 0.3],  # c0, c1 then c2 (positive): rank 2
        [0.1, 0.15, 0.3, 0.4, 0.5, 0.01, 0.05],  # c4,c3,c2,c1,c0 then c6 (positive): rank 5
    ]
)
IMG2CAPS = [[0, 1], [2, 3, 4], [5, 6]]
CAP2IMG = [0, 0, 1, 1, 1, 2, 2]
# t2i ranks per caption column: c0 -> 0, c1 -> 1, c2 -> 1, c3 -> 1, c4 -> 1, c5 -> 1, c6 -> 2.


def test_hand_built_values() -> None:
    out = recall_at_k(SIM, IMG2CAPS, CAP2IMG)
    assert out["i2t_r1"] == pytest.approx(100 / 3)
    assert out["i2t_r5"] == pytest.approx(200 / 3)
    assert out["i2t_r10"] == pytest.approx(100.0)
    assert out["t2i_r1"] == pytest.approx(100 / 7)
    assert out["t2i_r5"] == pytest.approx(100.0)
    assert out["t2i_r10"] == pytest.approx(100.0)
    assert out["rsum"] == pytest.approx(100 / 3 + 200 / 3 + 100 + 100 / 7 + 200)
    assert set(out) == {"i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10", "rsum"}

    out3 = recall_at_k(SIM, IMG2CAPS, torch.tensor(CAP2IMG), ks=(1, 3))
    assert out3["i2t_r3"] == pytest.approx(200 / 3)
    assert out3["t2i_r3"] == pytest.approx(100.0)
    assert set(out3) == {"i2t_r1", "i2t_r3", "t2i_r1", "t2i_r3", "rsum"}


def test_permutation_invariance() -> None:
    base = recall_at_k(SIM, IMG2CAPS, CAP2IMG)
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        cap_perm = torch.randperm(7, generator=g)
        img_perm = torch.randperm(3, generator=g)
        # new caption j = old caption cap_perm[j]; new image i = old image img_perm[i]
        old2new_img = torch.empty(3, dtype=torch.long)
        old2new_img[img_perm] = torch.arange(3)
        sim = SIM[img_perm][:, cap_perm]
        cap2img = [int(old2new_img[CAP2IMG[int(c)]]) for c in cap_perm]
        img2caps: list[list[int]] = [[] for _ in range(3)]
        for c, i in enumerate(cap2img):
            img2caps[i].append(c)
        out = recall_at_k(sim, img2caps, cap2img)
        for k, v in base.items():
            assert out[k] == pytest.approx(v)


def test_accepts_other_dtypes_and_devices() -> None:
    out = recall_at_k(SIM.to(torch.float16), IMG2CAPS, CAP2IMG)
    assert out["rsum"] == pytest.approx(recall_at_k(SIM, IMG2CAPS, CAP2IMG)["rsum"])


def test_inconsistent_maps_raise() -> None:
    with pytest.raises(ValueError, match="disagrees"):
        recall_at_k(SIM, [[0, 2], [1, 3, 4], [5, 6]], CAP2IMG)
    with pytest.raises(ValueError, match="no captions"):
        recall_at_k(SIM, [[0, 1], [], [5, 6]], CAP2IMG)
    with pytest.raises(ValueError, match="cap2img has"):
        recall_at_k(SIM, IMG2CAPS, CAP2IMG[:-1])
    with pytest.raises(ValueError, match="exactly one image"):
        recall_at_k(SIM, [[0, 1], [2, 3], [5, 6]], CAP2IMG)
    with pytest.raises(ValueError, match="positive"):
        recall_at_k(SIM, IMG2CAPS, CAP2IMG, ks=(0,))
