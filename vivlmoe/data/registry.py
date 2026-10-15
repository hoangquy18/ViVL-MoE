"""Dataset adapter registry: ``kind`` (YAML) -> adapter class, with strictly validated options.

An adapter turns one local/remote dataset into immutable records (:mod:`vivlmoe.data.records`)
plus an image store; the Tasks never see file formats. New datasets are added by subclassing
:class:`vivlmoe.data.adapters.base.BaseAdapter` and decorating with :func:`register_dataset`
(see ``docs/tasks.md``, "Adding a dataset").
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from vivlmoe.config.loader import from_dict
from vivlmoe.config.registry import Registry
from vivlmoe.config.schema import DataSourceCfg
from vivlmoe.data.image_store import ImageStore
from vivlmoe.data.records import RecordSet


@runtime_checkable
class DatasetAdapter(Protocol):
    kind: str

    def load_split(self, split: str) -> RecordSet: ...

    def image_store(self, split: str) -> ImageStore: ...


DATASETS: Registry[type[Any]] = Registry("datasets")


def register_dataset(key: str) -> Callable[[type[Any]], type[Any]]:
    """Class decorator: register ``cls`` under ``key`` and stamp ``cls.kind``."""

    def decorator(cls: type[Any]) -> type[Any]:
        if not hasattr(cls, "Options"):
            raise TypeError(f"{cls.__name__} must define a nested 'Options' dataclass")
        cls.kind = key
        return DATASETS.register(key)(cls)

    return decorator


def build_adapter(cfg: DataSourceCfg, run_dir: str | Path | None = None) -> Any:
    """Instantiate the adapter for ``cfg.kind`` after validating ``cfg.options`` strictly against
    its ``Options`` dataclass (unknown keys / wrong types raise :class:`ConfigError`)."""
    import vivlmoe.data.adapters  # noqa: F401  (registers the built-in adapters)

    cls = DATASETS.get(cfg.kind)
    options = from_dict(cls.Options, cfg.options, path=f"data.<{cfg.kind}>.options")
    return cls(cfg, options, run_dir=run_dir)


__all__ = ["DATASETS", "DatasetAdapter", "build_adapter", "register_dataset"]
