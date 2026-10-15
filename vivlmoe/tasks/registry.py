"""``TASKS`` registry (``task.name`` -> Task class) and :func:`build_task`."""

from __future__ import annotations

import importlib
import logging

from vivlmoe.config.registry import Registry
from vivlmoe.config.schema import ConfigError, RunConfig
from vivlmoe.tasks.base import Task
from vivlmoe.training.distributed import DistContext

log = logging.getLogger(__name__)

TASKS: Registry[type[Task]] = Registry("tasks")

BUILTIN_TASK_MODULES = (
    "vivlmoe.tasks.stage1_teacher",
    "vivlmoe.tasks.contrastive",
    "vivlmoe.tasks.captioning",
    "vivlmoe.tasks.vqa",
    "vivlmoe.tasks.vimacsa",
)


def import_builtin_tasks() -> list[str]:
    """Import the built-in task modules so they register; a missing module logs a warning."""
    loaded = []
    for mod in BUILTIN_TASK_MODULES:
        try:
            importlib.import_module(mod)
            loaded.append(mod)
        except ModuleNotFoundError as exc:
            if exc.name != mod:
                raise
            log.warning("task module %s is not available yet (%s)", mod, exc)
    return loaded


def build_task(cfg: RunConfig, ctx: DistContext) -> Task:
    if cfg.task.name not in TASKS:
        raise ConfigError(f"unknown task.name '{cfg.task.name}'; registered: {TASKS.keys()}")
    return TASKS.get(cfg.task.name)(cfg, ctx)


__all__ = ["BUILTIN_TASK_MODULES", "TASKS", "build_task", "import_builtin_tasks"]
