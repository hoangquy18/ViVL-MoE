"""Built-in dataset adapters; importing this package registers them in ``DATASETS``."""

from vivlmoe.data.adapters import (  # noqa: F401
    caption_text,
    contrastive_corpus,
    ktvic,
    phomt,
    synthetic,
    uitviic,
    vimacsa,
    vqa_generic,
)
from vivlmoe.data.adapters.base import BaseAdapter

__all__ = ["BaseAdapter"]
