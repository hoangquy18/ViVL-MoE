from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.unit.training.synthetic_task import SyntheticTask, make_cfg
from vivlmoe.config.schema import ConfigError
from vivlmoe.tasks import registry
from vivlmoe.tasks.registry import TASKS, build_task, import_builtin_tasks
from vivlmoe.training.distributed import DistContext


def test_build_task_and_unknown_name(tmp_path: Path) -> None:
    assert "synthetic" in TASKS
    cfg = make_cfg(tmp_path)
    task = build_task(cfg, DistContext.single_process("cpu"))
    assert isinstance(task, SyntheticTask) and task.name == "synthetic"
    cfg.task.name = "does_not_exist"
    with pytest.raises(ConfigError, match="unknown task.name"):
        build_task(cfg, DistContext.single_process("cpu"))


def test_import_builtin_tasks_tolerates_missing_modules_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry, "BUILTIN_TASK_MODULES", ("vivlmoe.tasks._not_written_yet",))
    assert import_builtin_tasks() == []
    (tmp_path / "broken_task_mod.py").write_text("import dependency_that_is_missing_xyz\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("broken_task_mod", None)
    monkeypatch.setattr(registry, "BUILTIN_TASK_MODULES", ("broken_task_mod",))
    with pytest.raises(ModuleNotFoundError, match="dependency_that_is_missing_xyz"):
        import_builtin_tasks()
