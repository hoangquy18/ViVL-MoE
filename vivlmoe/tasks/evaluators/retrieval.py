"""Image-text retrieval evaluators (paper Appendix A: model selection by R@K on KTVIC / UIT-ViIC dev).

Both evaluators share one protocol: the image side and the text side of a
:class:`~vivlmoe.data.datasets.RetrievalEvalDataset` are encoded on every rank over ordered,
unpadded shards, reassembled with :func:`gather_ordered_tensors` (called unconditionally, even on
an empty shard), and rank 0 computes ``sim = img @ txt.T`` on the CPU in fp32 and
:func:`~vivlmoe.metrics.retrieval.recall_at_k` with the dataset's ``img2caps`` / ``cap2img``
(keys ``i2t_r1..t2i_r10`` and ``rsum``).

* :class:`RetrievalEvaluator` - the dual-encoder backbone (Stages 2/3, sparse training):
  image side ``model(batch, 'encode_image')``, text side ``model(batch, 'encode_text')``.
* :class:`ClipSpaceRetrievalEvaluator` - Stage 1: image embeddings from the frozen CLIP image
  teacher (computed **once** per split and cached on the CPU), text side = the student output
  ``FC1(XLM-R <s>)`` L2-normalised in the CLIP joint space.

An eval "loader" for a retrieval split is a :class:`RetrievalLoaders` pair (image loader + text
loader + the dataset), which is what the pretraining tasks put into ``LoaderBundle.evals``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from vivlmoe.config.schema import DataCfg
from vivlmoe.data.collators import RetrievalImageCollator, RetrievalTextCollator
from vivlmoe.data.datasets import RetrievalEvalDataset
from vivlmoe.data.loaders import build_eval_loader, move_to_device
from vivlmoe.metrics.retrieval import recall_at_k
from vivlmoe.tasks.evaluators.base import Evaluator, gather_ordered_tensors
from vivlmoe.tasks.evaluators.dump import metrics_path, write_metrics_json
from vivlmoe.training.distributed import DistContext
from vivlmoe.training.state import EvalContext

log = logging.getLogger(__name__)

Batch = dict[str, Any]
EncodeFn = Callable[[nn.Module, Batch], torch.Tensor]


@dataclass
class RetrievalLoaders:
    """The two ordered-shard loaders of one retrieval split (what ``LoaderBundle.evals`` holds)."""

    dataset: RetrievalEvalDataset
    images: DataLoader
    texts: DataLoader

    def __len__(self) -> int:
        return len(self.images) + len(self.texts)


def build_retrieval_loaders(
    dataset: RetrievalEvalDataset,
    text_collate: Callable[..., Batch],
    data_cfg: DataCfg,
    ctx: DistContext,
    image_collate: Callable[..., Batch] | None = None,
) -> RetrievalLoaders:
    """Ordered ``rank::world`` loaders over ``dataset.images`` / ``dataset.texts``."""
    images = build_eval_loader(
        dataset.images,
        image_collate or RetrievalImageCollator(),
        data_cfg,
        ctx.world_size,
        ctx.rank,
    )
    texts = build_eval_loader(dataset.texts, text_collate, data_cfg, ctx.world_size, ctx.rank)
    return RetrievalLoaders(dataset=dataset, images=images, texts=texts)


def text_collator(tokenizer: Any, max_len: int) -> RetrievalTextCollator:
    return RetrievalTextCollator(tokenizer, max_len)


def encode_side(
    model: nn.Module,
    loader: DataLoader,
    encode: EncodeFn,
    embed_dim: int,
    ctx: DistContext,
) -> torch.Tensor:
    """Encode every batch of the local shard and gather ``[N, embed_dim]`` fp32 in uid order.

    The gather is a collective and runs on every rank; an empty shard contributes ``[0, embed_dim]``
    (the trailing dimension must match across ranks).
    """
    uids: list[int] = []
    parts: list[torch.Tensor] = []
    for batch in loader:
        batch = move_to_device(batch, ctx.device)
        emb = encode(model, batch).detach().float()
        if emb.ndim != 2 or emb.shape[1] != embed_dim:
            raise RuntimeError(
                f"encoder returned {tuple(emb.shape)}, expected [B, {embed_dim}] embeddings"
            )
        parts.append(emb)
        uids.extend(int(u) for u in batch["uids"])
    ids = torch.tensor(uids, dtype=torch.int64)
    local = torch.cat(parts) if parts else torch.zeros(0, embed_dim)
    return gather_ordered_tensors(ids, local.cpu(), ctx).cpu()


def retrieval_metrics(
    image_embeds: torch.Tensor, text_embeds: torch.Tensor, dataset: RetrievalEvalDataset
) -> dict[str, float]:
    """``recall_at_k`` on the cosine-similarity matrix of L2-normalised embeddings (CPU fp32)."""
    img = F.normalize(image_embeds.float().cpu(), dim=-1)
    txt = F.normalize(text_embeds.float().cpu(), dim=-1)
    if img.shape[0] != len(dataset.image_ids) or txt.shape[0] != len(dataset.captions):
        raise RuntimeError(
            f"gathered {img.shape[0]} images / {txt.shape[0]} captions but the split has "
            f"{len(dataset.image_ids)} / {len(dataset.captions)}"
        )
    sim = img @ txt.t()
    return recall_at_k(sim, dataset.img2caps, dataset.cap2img)


def dump_retrieval_metrics(
    ec: EvalContext, evaluator: str, metrics: dict[str, float], dataset: RetrievalEvalDataset
) -> None:
    """Rank-0 ``<run_dir>/metrics/<split>_step<N>.json`` with the R@K dict and split sizes."""
    if not ec.ctx.is_main:
        return
    write_metrics_json(
        metrics_path(ec.run_dir, ec.split_name, ec.step),
        {
            "split": ec.split_name,
            "step": ec.step,
            "evaluator": evaluator,
            "num_images": len(dataset.image_ids),
            "num_captions": len(dataset.captions),
            "metrics": metrics,
            "scales": {"i2t_r*/t2i_r*": "recall in percent", "rsum": "sum of the six recalls"},
        },
    )


def _backbone_of(model: nn.Module) -> nn.Module:
    return getattr(model, "backbone", model)


def _encode_image(model: nn.Module, batch: Batch) -> torch.Tensor:
    return model(batch, "encode_image")["embeds"]


def _encode_text(model: nn.Module, batch: Batch) -> torch.Tensor:
    return model(batch, "encode_text")["embeds"]


class RetrievalEvaluator(Evaluator):
    """R@K for the ViVL backbone (``encode_image`` / ``encode_text`` modes)."""

    name = "retrieval"

    def __init__(self, ctx: DistContext) -> None:
        self.ctx = ctx

    def run(self, model: nn.Module, loader: Any, ec: EvalContext) -> dict[str, float] | None:
        if not isinstance(loader, RetrievalLoaders):
            raise TypeError(f"RetrievalEvaluator expects RetrievalLoaders, got {type(loader)}")
        dim = int(_backbone_of(model).dim)
        img = encode_side(model, loader.images, _encode_image, dim, ec.ctx)
        txt = encode_side(model, loader.texts, _encode_text, dim, ec.ctx)
        if not ec.ctx.is_main:
            return None
        metrics = retrieval_metrics(img, txt, loader.dataset)
        dump_retrieval_metrics(ec, self.name, metrics, loader.dataset)
        return metrics


class ClipSpaceRetrievalEvaluator(Evaluator):
    """Stage-1 retrieval in the CLIP joint space (plan decision 11).

    ``image_teacher`` is a zero-argument callable returning the (device-resident) CLIP image
    teacher; it is invoked only while a split's image embeddings are not cached yet. ``student``
    turns a text batch into ``[B, P]`` student embeddings.
    """

    name = "clip_space_retrieval"

    def __init__(
        self,
        ctx: DistContext,
        image_teacher: Callable[[], nn.Module],
        student: EncodeFn,
        embed_dim: int,
    ) -> None:
        self.ctx = ctx
        self._image_teacher = image_teacher
        self._student = student
        self.embed_dim = embed_dim
        self._image_cache: dict[str, torch.Tensor] = {}  # split name -> [n_img, P] CPU fp32

    def image_embeddings(self, split: str, loader: DataLoader, ctx: DistContext) -> torch.Tensor:
        """CLIP image embeddings of ``split`` (computed once on all ranks, then cached on CPU)."""
        cached = self._image_cache.get(split)
        if cached is not None:
            return cached
        teacher = self._image_teacher()
        teacher.eval()

        def encode(_model: nn.Module, batch: Batch) -> torch.Tensor:
            # fp32 like the text teacher in train_step, even if evaluation ever runs under autocast
            with torch.autocast(device_type=ctx.device.type, enabled=False):
                return teacher(batch["pixel_values"]).float()

        emb = encode_side(teacher, loader, encode, self.embed_dim, ctx)
        self._image_cache[split] = emb
        log.info(
            "cached CLIP image embeddings for %s: %s (rank %d)", split, tuple(emb.shape), ctx.rank
        )
        return emb

    def run(self, model: nn.Module, loader: Any, ec: EvalContext) -> dict[str, float] | None:
        if not isinstance(loader, RetrievalLoaders):
            raise TypeError(
                f"ClipSpaceRetrievalEvaluator expects RetrievalLoaders, got {type(loader)}"
            )
        img = self.image_embeddings(ec.split_name, loader.images, ec.ctx)
        txt = encode_side(model, loader.texts, self._student, self.embed_dim, ec.ctx)
        if not ec.ctx.is_main:
            return None
        metrics = retrieval_metrics(img, txt, loader.dataset)
        dump_retrieval_metrics(ec, self.name, metrics, loader.dataset)
        return metrics


__all__ = [
    "ClipSpaceRetrievalEvaluator",
    "RetrievalEvaluator",
    "RetrievalLoaders",
    "build_retrieval_loaders",
    "dump_retrieval_metrics",
    "encode_side",
    "retrieval_metrics",
    "text_collator",
]
