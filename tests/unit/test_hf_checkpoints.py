"""scripts/hf_checkpoints.py: file selection, private-only pushes, resumable uploads (fake Hub API)."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "hf_checkpoints.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("hf_checkpoints_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _runs(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    ck = runs / "v2_align" / "checkpoints"
    ck.mkdir(parents=True)
    for name in ("best.pt", "best.pt.meta.json", "step_0000100.pt", "step_0000100.pt.meta.json"):
        (ck / name).write_bytes(b"x" * (300 if name.endswith(".pt") else 10))
    os.symlink("step_0000100.pt", ck / "last.pt")
    os.symlink("step_0000100.pt.meta.json", ck / "last.pt.meta.json")
    for name in ("train.log", "eval_history.jsonl", "config.resolved.yaml", "tb/events.out"):
        (runs / "v2_align" / name).parent.mkdir(parents=True, exist_ok=True)
        (runs / "v2_align" / name).write_text("log", encoding="utf-8")
    (runs / "v2_moe_upcycled").mkdir()
    (runs / "v2_moe_upcycled" / "upcycled.pt").write_bytes(b"u" * 300)
    (runs / "stage2_contrastive").mkdir()  # a v1 run: not matched by v2_*
    (runs / "stage2_contrastive" / "train.log").write_text("v1", encoding="utf-8")
    (runs / ".cache").mkdir()
    (runs / ".cache" / "x").write_text("hidden", encoding="utf-8")
    return runs


def test_selection_skips_step_checkpoints_and_last_unless_asked(tmp_path: Path) -> None:
    mod = _load()
    runs = _runs(tmp_path)
    got = {p.as_posix() for p in mod.select_files(runs, "v2_*", with_last=False)}
    assert got == {
        "v2_align/checkpoints/best.pt",
        "v2_align/checkpoints/best.pt.meta.json",
        "v2_align/config.resolved.yaml",
        "v2_align/eval_history.jsonl",
        "v2_align/tb/events.out",
        "v2_align/train.log",
        "v2_moe_upcycled/upcycled.pt",
    }
    with_last = {p.as_posix() for p in mod.select_files(runs, "v2_*", with_last=True)}
    assert with_last - got == {
        "v2_align/checkpoints/last.pt",
        "v2_align/checkpoints/last.pt.meta.json",
    }


class _FakeApi:
    def __init__(self, private: bool = True, remote: dict[str, int] | None = None) -> None:
        self.private, self.remote = private, dict(remote or {})
        self.commits: list[list[str]] = []
        self.uploads: list[str] = []

    def create_repo(self, repo_id: str, **kw: Any) -> None:
        assert kw["private"] is True and kw["repo_type"] == "model"

    def repo_info(self, repo_id: str, **kw: Any) -> Any:
        return SimpleNamespace(private=self.private)

    def list_repo_tree(self, repo_id: str, **kw: Any) -> list[Any]:
        return [SimpleNamespace(path=p, size=s) for p, s in self.remote.items()]

    def create_commit(self, repo_id: str, operations: list[Any], **kw: Any) -> None:
        self.commits.append([op.path_in_repo for op in operations])

    def upload_file(self, *, path_or_fileobj: str, path_in_repo: str, **kw: Any) -> None:
        self.uploads.append(path_in_repo)


def test_push_uploads_what_the_repo_lacks(tmp_path: Path) -> None:
    mod = _load()
    runs = _runs(tmp_path)
    files = mod.select_files(runs, "v2_*", with_last=True)
    api = _FakeApi(remote={"v2_align/checkpoints/best.pt": 300})  # already uploaded
    result = mod.push(api, "me/ckpt", runs, files, big_bytes=100)
    assert set(api.uploads) == {"v2_moe_upcycled/upcycled.pt", "v2_align/checkpoints/last.pt"}
    assert len(api.commits) == 1 and "README.md" in api.commits[0]
    assert (
        "v2_align/train.log" in api.commits[0]
        and "v2_align/checkpoints/best.pt" not in api.commits[0]
    )
    assert result == {"selected": len(files), "uploaded": len(files) - 1}


def test_push_refuses_a_public_repo(tmp_path: Path) -> None:
    mod = _load()
    runs = _runs(tmp_path)
    with pytest.raises(SystemExit, match="public"):
        mod.push(_FakeApi(private=False), "me/ckpt", runs, mod.select_files(runs, "v2_*", False))


def test_dry_run_needs_no_network(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load()
    runs = _runs(tmp_path)
    assert mod.main(["push", "--repo", "me/ckpt", "--runs", str(runs), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "v2_moe_upcycled/upcycled.pt" in out and "7 files" in out and "step_" not in out
