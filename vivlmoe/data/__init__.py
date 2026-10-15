"""Data pipeline: adapters -> records -> datasets -> collators -> samplers -> loaders."""

import vivlmoe.data.adapters  # noqa: F401  (registers the built-in adapters)
from vivlmoe.data.collators import (
    ContrastiveCollator,
    MultiImageClsCollator,
    RetrievalImageCollator,
    RetrievalTextCollator,
    Seq2SeqCollator,
    TeacherCollator,
)
from vivlmoe.data.datasets import (
    CaptionEvalDataset,
    ImageTextDataset,
    MultiImageClsDataset,
    ParallelTextDataset,
    RetrievalEvalDataset,
    VQADataset,
)
from vivlmoe.data.image_store import (
    CompositeImageStore,
    HFImageStore,
    ImageLoadError,
    ImageStore,
    LocalImageStore,
)
from vivlmoe.data.loaders import build_eval_loader, build_train_loader, move_to_device
from vivlmoe.data.records import (
    ColumnarImageText,
    ImageRef,
    ImageTextRecord,
    ListRecordSet,
    MultiImageClsRecord,
    ParallelTextRecord,
    RecordSet,
    VQARecord,
)
from vivlmoe.data.registry import DATASETS, DatasetAdapter, build_adapter, register_dataset
from vivlmoe.data.samplers import (
    OneCaptionPerImageDistributedSampler,
    OrderedShardSampler,
    build_train_sampler,
)
from vivlmoe.data.text_normalize import NormalizeOptions, VietnameseNormalizer, build_normalizer
from vivlmoe.data.tokenization import TokenizerBundle, build_tokenizers
from vivlmoe.data.transforms import ImageTransformSpec, build_transform, build_transform_spec

__all__ = [
    "DATASETS",
    "CaptionEvalDataset",
    "ColumnarImageText",
    "CompositeImageStore",
    "ContrastiveCollator",
    "DatasetAdapter",
    "HFImageStore",
    "ImageLoadError",
    "ImageRef",
    "ImageStore",
    "ImageTextDataset",
    "ImageTextRecord",
    "ImageTransformSpec",
    "ListRecordSet",
    "LocalImageStore",
    "MultiImageClsCollator",
    "MultiImageClsDataset",
    "MultiImageClsRecord",
    "NormalizeOptions",
    "OneCaptionPerImageDistributedSampler",
    "OrderedShardSampler",
    "ParallelTextDataset",
    "ParallelTextRecord",
    "RecordSet",
    "RetrievalEvalDataset",
    "RetrievalImageCollator",
    "RetrievalTextCollator",
    "Seq2SeqCollator",
    "TeacherCollator",
    "TokenizerBundle",
    "VQADataset",
    "VQARecord",
    "VietnameseNormalizer",
    "build_adapter",
    "build_eval_loader",
    "build_normalizer",
    "build_tokenizers",
    "build_train_loader",
    "build_train_sampler",
    "build_transform",
    "build_transform_spec",
    "move_to_device",
    "register_dataset",
]
