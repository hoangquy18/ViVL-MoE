"""Convenience glue used by the Tasks: config -> adapter -> records -> dataset (+ groups)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from vivlmoe.config.schema import DataSourceCfg, RunConfig
from vivlmoe.data.datasets import (
    CaptionEvalDataset,
    ImageTextDataset,
    MultiImageClsDataset,
    ParallelTextDataset,
    RetrievalEvalDataset,
    VQADataset,
)
from vivlmoe.data.records import RecordSet
from vivlmoe.data.registry import build_adapter
from vivlmoe.data.text_normalize import build_normalizer
from vivlmoe.data.transforms import ImageTransformSpec, build_transform, build_transform_spec

log = logging.getLogger(__name__)


def build_transform_for(cfg: RunConfig) -> tuple[ImageTransformSpec, Any, Any]:
    """``(spec, train_transform, eval_transform)`` for the run's vision tower."""
    spec = build_transform_spec(cfg.model.vision, cfg.data.transform)
    return spec, build_transform(spec, train=True), build_transform(spec, train=False)


def draft_size_for(cfg: RunConfig, spec: ImageTransformSpec) -> int | None:
    return spec.size if cfg.data.transform.draft_decode else None


def build_records(
    source_cfg: DataSourceCfg,
    split: str | None = None,
    run_dir: str | Path | None = None,
    write_dropped: bool = True,
) -> tuple[RecordSet, Any]:
    """Load ``split`` (default ``source_cfg.split``) through the adapter pipeline; returns
    ``(records, adapter)``. ``write_dropped`` should be true on rank 0 only."""
    adapter = build_adapter(source_cfg, run_dir)
    split = split or source_cfg.split
    records = adapter.load_split(split)
    if write_dropped:
        adapter.write_dropped(split, run_dir)
    return records, adapter


def build_image_text_train(
    source_cfg: DataSourceCfg,
    transform: Any,
    draft_size: int | None = None,
    run_dir: str | Path | None = None,
    write_dropped: bool = True,
) -> tuple[ImageTextDataset, list[list[int]]]:
    """Contrastive training dataset + image groups for the one-caption-per-image sampler."""
    records, adapter = build_records(source_cfg, None, run_dir, write_dropped)
    ds = ImageTextDataset(records, adapter.image_store(source_cfg.split), transform, draft_size)
    return ds, records.groups()


def build_retrieval_eval(
    source_cfg: DataSourceCfg,
    transform: Any,
    draft_size: int | None = None,
    run_dir: str | Path | None = None,
    write_dropped: bool = True,
) -> RetrievalEvalDataset:
    records, adapter = build_records(source_cfg, None, run_dir, write_dropped)
    return RetrievalEvalDataset(
        records, adapter.image_store(source_cfg.split), transform, draft_size
    )


def build_caption_eval(
    source_cfg: DataSourceCfg,
    transform: Any,
    draft_size: int | None = None,
    run_dir: str | Path | None = None,
    write_dropped: bool = True,
) -> CaptionEvalDataset:
    records, adapter = build_records(source_cfg, None, run_dir, write_dropped)
    return CaptionEvalDataset(records, adapter.image_store(source_cfg.split), transform, draft_size)


def build_vqa(
    source_cfg: DataSourceCfg,
    transform: Any,
    draft_size: int | None = None,
    run_dir: str | Path | None = None,
    write_dropped: bool = True,
    expand_answers: bool = False,
) -> tuple[VQADataset, str]:
    """``(dataset, prompt_template)`` — the template comes from the adapter options when the
    adapter defines one (``vqa_generic``), else ``"{question}"``."""
    records, adapter = build_records(source_cfg, None, run_dir, write_dropped)
    ds = VQADataset(
        records, adapter.image_store(source_cfg.split), transform, draft_size, expand_answers
    )
    template = getattr(adapter.options, "prompt_template", "{question}")
    return ds, template


def build_vimacsa(
    source_cfg: DataSourceCfg,
    transform: Any,
    image_hw: tuple[int, int],
    max_images: int = 6,
    draft_size: int | None = None,
    run_dir: str | Path | None = None,
    write_dropped: bool = True,
) -> MultiImageClsDataset:
    records, adapter = build_records(source_cfg, None, run_dir, write_dropped)
    return MultiImageClsDataset(
        records, adapter.image_store(source_cfg.split), transform, image_hw, max_images, draft_size
    )


def build_parallel_text(
    source_cfg: DataSourceCfg, run_dir: str | Path | None = None
) -> ParallelTextDataset:
    records, _ = build_records(source_cfg, None, run_dir, write_dropped=False)
    return ParallelTextDataset(records)


__all__ = [
    "build_caption_eval",
    "build_image_text_train",
    "build_normalizer",
    "build_parallel_text",
    "build_records",
    "build_retrieval_eval",
    "build_transform_for",
    "build_vimacsa",
    "build_vqa",
    "draft_size_for",
]
