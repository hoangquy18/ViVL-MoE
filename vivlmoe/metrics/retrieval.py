"""Image-text retrieval Recall@K (paper Appendix A model selection: R@K on KTVIC / UIT-ViIC dev).

Pure CPU fp32 torch: ``sim[n_img, n_cap]`` is the cosine-similarity matrix between image and
caption embeddings; ``img2caps[i]`` lists the caption indices of image ``i`` and
``cap2img[c]`` the image index of caption ``c`` (1-to-many). A query counts as a hit at K
when any of its positives ranks in the top K. Values are percentages; ``rsum`` is their sum.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def recall_at_k(
    sim: torch.Tensor,
    img2caps: Sequence[Sequence[int]],
    cap2img: Sequence[int] | torch.Tensor,
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Return ``{'i2t_r{k}', 't2i_r{k}' for k in ks, 'rsum'}`` in percent."""
    if sim.ndim != 2:
        raise ValueError(f"sim must be [n_img, n_cap], got shape {tuple(sim.shape)}")
    sim = sim.detach().to(device="cpu", dtype=torch.float32)
    n_img, n_cap = sim.shape
    if n_img == 0 or n_cap == 0:
        raise ValueError("sim must be non-empty")
    if len(img2caps) != n_img:
        raise ValueError(f"img2caps has {len(img2caps)} entries but sim has {n_img} images")
    cap2img_t = torch.as_tensor(cap2img, dtype=torch.long).reshape(-1)
    if cap2img_t.numel() != n_cap:
        raise ValueError(f"cap2img has {cap2img_t.numel()} entries but sim has {n_cap} captions")
    if cap2img_t.numel() and not (0 <= int(cap2img_t.min()) and int(cap2img_t.max()) < n_img):
        raise ValueError("cap2img contains image indices outside [0, n_img)")

    positives = torch.zeros(n_img, n_cap, dtype=torch.bool)
    for i, caps in enumerate(img2caps):
        if len(caps) == 0:
            raise ValueError(f"image {i} has no captions")
        caps_t = torch.as_tensor(list(caps), dtype=torch.long)
        if not bool((cap2img_t[caps_t] == i).all()):
            raise ValueError(f"img2caps[{i}] disagrees with cap2img")
        positives[i, caps_t] = True
    if int(positives.sum()) != n_cap:
        raise ValueError("every caption must belong to exactly one image (img2caps covers cap2img)")

    # image -> text: rank of the best-ranked positive caption per image.
    order_i2t = sim.argsort(dim=1, descending=True, stable=True)  # [n_img, n_cap]
    ranked_pos = positives.gather(1, order_i2t)
    rank_i2t = ranked_pos.to(torch.int64).argmax(dim=1)  # first True (exists by construction)

    # text -> image: rank of the single ground-truth image per caption.
    order_t2i = sim.t().argsort(dim=1, descending=True, stable=True)  # [n_cap, n_img]
    rank_t2i = (order_t2i == cap2img_t[:, None]).to(torch.int64).argmax(dim=1)

    out: dict[str, float] = {}
    for k in ks:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        out[f"i2t_r{k}"] = float((rank_i2t < k).double().mean() * 100.0)
        out[f"t2i_r{k}"] = float((rank_t2i < k).double().mean() * 100.0)
    out["rsum"] = float(sum(out.values()))
    return out
