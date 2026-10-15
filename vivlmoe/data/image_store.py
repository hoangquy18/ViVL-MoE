"""Image sources: local directories, Hugging Face ``datasets`` image columns, and their union.

Every store resolves an :class:`~vivlmoe.data.records.ImageRef` to a PIL RGB image. A missing or
undecodable image raises :class:`ImageLoadError`; nothing ever substitutes a placeholder (the
legacy grey-image fallback silently trained on garbage). Missing images are removed at index time
by :mod:`vivlmoe.data.manifest`, never at ``__getitem__`` time.
"""

from __future__ import annotations

import io
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from vivlmoe.data.records import ImageRef

log = logging.getLogger(__name__)


class ImageLoadError(RuntimeError):
    """An image referenced by a record does not exist or cannot be decoded."""


def _to_rgb(img: Image.Image, draft_size: int | None) -> Image.Image:
    if draft_size is not None and img.format == "JPEG":
        # JPEG DCT downscale-on-decode: cheap for the 4-8 MP crawled corpus images. PIL picks the
        # largest scale >= requested size, so quality of the later bicubic resize is preserved.
        img.draft("RGB", (draft_size, draft_size))
    return img.convert("RGB")


class ImageStore:
    """Base interface. ``source`` of the reference must be 0 for single stores."""

    def exists(self, ref: ImageRef) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def open(self, ref: ImageRef, draft_size: int | None = None) -> Image.Image:
        """Decode the image as RGB (raises :class:`ImageLoadError`)."""
        raise NotImplementedError  # pragma: no cover - abstract

    def describe(self) -> str:
        """Stable description used as part of the manifest-cache key."""
        raise NotImplementedError  # pragma: no cover - abstract

    def _check_source(self, ref: ImageRef) -> None:
        if ref.source != 0:
            raise ImageLoadError(
                f"{type(self).__name__} only serves source 0, got ImageRef(source={ref.source})"
            )


class LocalImageStore(ImageStore):
    """Images under ``root``; ``ref.key`` is a path relative to ``root``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, ref: ImageRef) -> Path:
        return self.root / str(ref.key)

    def exists(self, ref: ImageRef) -> bool:
        self._check_source(ref)
        return self.path(ref).is_file()

    def open(self, ref: ImageRef, draft_size: int | None = None) -> Image.Image:
        self._check_source(ref)
        path = self.path(ref)
        try:
            with Image.open(path) as img:
                return _to_rgb(img, draft_size)
        except FileNotFoundError:
            raise ImageLoadError(f"image not found: {path}") from None
        except Exception as exc:  # PIL raises UnidentifiedImageError, OSError, ValueError, ...
            raise ImageLoadError(f"cannot decode image {path}: {exc}") from exc

    def describe(self) -> str:
        return f"local:{self.root.resolve()}"


class HFImageStore(ImageStore):
    """Image column of a Hugging Face ``datasets.Dataset``; ``ref.key`` is the row index.

    The dataset handle is created lazily in every process (DataLoader workers get their own
    memory-mapped Arrow reader) with ``Image(decode=False)`` so only the raw bytes are read.
    ``datasets`` is an optional dependency (``pip install vivlmoe[hfdata]``).
    """

    def __init__(
        self,
        dataset: str,
        split: str = "train",
        image_column: str = "image",
        load_kwargs: dict[str, Any] | None = None,
        _dataset_obj: Any = None,
    ) -> None:
        self.dataset = dataset
        self.split = split
        self.image_column = image_column
        self.load_kwargs = dict(load_kwargs or {})
        self._ds = _dataset_obj
        self._n: int | None = len(_dataset_obj) if _dataset_obj is not None else None

    @classmethod
    def from_dataset(cls, ds: Any, image_column: str = "image") -> HFImageStore:
        """Wrap an in-memory ``datasets.Dataset`` (tests). Pickled by value into workers."""
        return cls(dataset="<in-memory>", image_column=image_column, _dataset_obj=ds)

    @staticmethod
    def load_hf_dataset(name_or_dir: str, split: str, **kwargs: Any) -> Any:
        try:
            import datasets
        except ImportError as exc:
            raise ImportError(
                "HFImageStore needs the optional 'datasets' package: pip install 'vivlmoe[hfdata]'"
            ) from exc
        if os.path.isdir(name_or_dir):
            ds = datasets.load_from_disk(name_or_dir, **kwargs)
            if isinstance(ds, datasets.DatasetDict):
                ds = ds[split]
            return ds
        return datasets.load_dataset(name_or_dir, split=split, **kwargs)

    def _handle(self) -> Any:
        if self._ds is None:
            self._ds = self.load_hf_dataset(self.dataset, self.split, **self.load_kwargs)
        return self._ds

    def _undecoded(self) -> Any:
        ds = self._handle()
        if getattr(self, "_raw", None) is None:
            import datasets

            feats = ds.features
            if self.image_column not in feats:
                raise KeyError(
                    f"HF dataset {self.dataset!r} has no column {self.image_column!r}; "
                    f"columns: {list(feats)}"
                )
            self._raw = ds.cast_column(self.image_column, datasets.Image(decode=False))
        return self._raw

    def __len__(self) -> int:
        if self._n is None:
            self._n = len(self._handle())
        return self._n

    def exists(self, ref: ImageRef) -> bool:
        self._check_source(ref)
        return 0 <= int(ref.key) < len(self)

    def open(self, ref: ImageRef, draft_size: int | None = None) -> Image.Image:
        self._check_source(ref)
        idx = int(ref.key)
        if not 0 <= idx < len(self):
            raise ImageLoadError(f"row {idx} out of range for HF dataset {self.dataset!r}")
        cell = self._undecoded()[idx][self.image_column]
        try:
            if isinstance(cell, dict) and cell.get("bytes") is not None:
                with Image.open(io.BytesIO(cell["bytes"])) as img:
                    return _to_rgb(img, draft_size)
            if isinstance(cell, dict) and cell.get("path"):
                with Image.open(cell["path"]) as img:
                    return _to_rgb(img, draft_size)
            if isinstance(cell, Image.Image):
                return _to_rgb(cell, draft_size)
        except ImageLoadError:
            raise
        except Exception as exc:
            raise ImageLoadError(
                f"cannot decode row {idx} of HF dataset {self.dataset!r}: {exc}"
            ) from exc
        raise ImageLoadError(f"row {idx} of HF dataset {self.dataset!r} holds no image bytes")

    def describe(self) -> str:
        return f"hf:{self.dataset}:{self.split}:{self.image_column}"

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_raw"] = None
        if self.dataset != "<in-memory>":
            state["_ds"] = None  # re-opened lazily in the worker process
        return state


class CompositeImageStore(ImageStore):
    """Union of stores addressed by ``ImageRef.source`` (used by ``mixture`` corpora)."""

    def __init__(self, stores: Sequence[ImageStore]) -> None:
        if not stores:
            raise ValueError("CompositeImageStore needs at least one store")
        self.stores = list(stores)

    def _sub(self, ref: ImageRef) -> tuple[ImageStore, ImageRef]:
        if not 0 <= ref.source < len(self.stores):
            raise ImageLoadError(
                f"ImageRef source {ref.source} out of range (stores={len(self.stores)})"
            )
        return self.stores[ref.source], ImageRef(0, ref.key)

    def exists(self, ref: ImageRef) -> bool:
        store, sub = self._sub(ref)
        return store.exists(sub)

    def open(self, ref: ImageRef, draft_size: int | None = None) -> Image.Image:
        store, sub = self._sub(ref)
        return store.open(sub, draft_size)

    def describe(self) -> str:
        return "composite[" + ";".join(s.describe() for s in self.stores) + "]"


__all__ = [
    "CompositeImageStore",
    "HFImageStore",
    "ImageLoadError",
    "ImageStore",
    "LocalImageStore",
]
