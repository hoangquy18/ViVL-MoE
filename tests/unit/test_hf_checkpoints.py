"""scripts/hf_checkpoints.py: file selection, private-only pushes, version tracking, the watcher (fake Hub)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "hf_checkpoints.py"
LOGS = ["v2_align/config.resolved.yaml", "v2_align/eval_history.jsonl", "v2_align/tb/events.out",
        "v2_align/train.log"]  # fmt: skip
LAST = ["v2_align/checkpoints/last.pt", "v2_align/checkpoints/last.pt.meta.json"]
BEST = ["v2_align/checkpoints/best.pt", "v2_align/checkpoints/best.pt.meta.json"]
UPCYCLED = ["v2_moe_upcycled/upcycled.pt", "v2_moe_upcycled/upcycled.pt.meta.json"]


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("hf_checkpoints_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- the trainer's side (CheckpointManager)
def _replace(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _save(ckpt_dir: Path, name: str, data: bytes, step: int) -> None:
    _replace(ckpt_dir / name, data)  # atomically, then the sidecar
    (ckpt_dir / f"{name}.meta.json").write_text(json.dumps({"step": step}), encoding="utf-8")


def _save_step(ckpt_dir: Path, step: int, data: bytes) -> None:
    """step_N.pt, last.pt -> it, older steps pruned (keep_last_n: 1)."""
    name = f"step_{step:07d}.pt"
    _save(ckpt_dir, name, data, step)
    for link, target in (("last.pt", name), ("last.pt.meta.json", f"{name}.meta.json")):
        (ckpt_dir / link).unlink(missing_ok=True)
        os.symlink(target, ckpt_dir / link)
    for old in ckpt_dir.glob("step_*.pt"):
        if old.name != name:
            old.unlink()
            old.with_name(old.name + ".meta.json").unlink()


def _runs(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    ck = runs / "v2_align" / "checkpoints"
    ck.mkdir(parents=True)
    _save(ck, "best.pt", b"b" * 300, 100)
    _save_step(ck, 100, b"s" * 300)
    for name in LOGS:
        (runs / name).parent.mkdir(parents=True, exist_ok=True)
        (runs / name).write_text("log", encoding="utf-8")
    (runs / "v2_moe_upcycled").mkdir()
    _save(runs / "v2_moe_upcycled", "upcycled.pt", b"u" * 300, 0)
    (runs / "stage2_contrastive").mkdir()  # a v1 run: not matched by v2_*
    (runs / "stage2_contrastive" / "train.log").write_text("v1", encoding="utf-8")
    (runs / ".cache").mkdir()
    (runs / ".cache" / "x").write_text("hidden", encoding="utf-8")
    return runs


class _FakeHub:
    def __init__(self, private: bool = True, exists: bool = True, role: str = "write") -> None:
        self.private, self.exists, self.role = private, exists, role
        self.files: dict[str, bytes] = {}
        self.commits: list[list[str]] = []
        self.squashes = 0
        self.during_commit: Any = None  # what the trainer does while an upload runs

    def whoami(self) -> dict:
        return {"name": "me", "auth": {"accessToken": {"role": self.role}}}

    def repo_exists(self, repo_id: str, **kw: Any) -> bool:
        return self.exists

    def create_repo(self, repo_id: str, **kw: Any) -> None:
        assert kw["private"] is True and kw["repo_type"] == "model"
        self.exists = True

    def repo_info(self, repo_id: str, **kw: Any) -> Any:
        return SimpleNamespace(private=self.private)

    def list_repo_tree(self, repo_id: str, **kw: Any) -> list[Any]:
        return [SimpleNamespace(path=p, size=len(d)) for p, d in self.files.items()]

    def create_commit(self, repo_id: str, operations: list[Any], **kw: Any) -> None:
        if self.during_commit is not None:
            self.during_commit()
        for op in operations:
            src = op.path_or_fileobj
            self.files[op.path_in_repo] = src if isinstance(src, bytes) else Path(src).read_bytes()
        self.commits.append(sorted(op.path_in_repo for op in operations))

    def super_squash_history(self, repo_id: str, **kw: Any) -> None:
        self.squashes += 1


def _push(mod: Any, hub: _FakeHub, runs: Path) -> dict[str, int]:
    return mod.Uploader(hub, "me/ckpt", runs).push(mod.select_files(runs, "v2_*", with_last=True))


# ---------------------------------------------------------------- tests
def test_selection_skips_step_checkpoints_and_last_unless_asked(tmp_path: Path) -> None:
    mod = _load()
    runs = _runs(tmp_path)
    got = {p.as_posix() for p in mod.select_files(runs, "v2_*", with_last=False)}
    assert got == {*BEST, *LOGS, *UPCYCLED}
    with_last = {p.as_posix() for p in mod.select_files(runs, "v2_*", with_last=True)}
    assert with_last - got == set(LAST)


def test_checkpoints_go_up_with_their_sidecars_then_only_what_changed(tmp_path: Path) -> None:
    mod = _load()
    runs = _runs(tmp_path)
    hub = _FakeHub()
    assert _push(mod, hub, runs) == {"selected": 10, "uploaded": 10}
    # each checkpoint with its sidecar, the resumable state first, then the logs in one commit
    assert hub.commits == [LAST, BEST, UPCYCLED, sorted([*LOGS, "README.md"])]
    assert hub.files["v2_align/checkpoints/last.pt"] == b"s" * 300  # the step file, not a link
    assert hub.squashes == 0 and not any((runs / mod.PINS).iterdir())
    assert _push(mod, hub, runs)["uploaded"] == 0 and len(hub.commits) == 4  # read from the record


def test_a_checkpoint_rewritten_with_the_same_size_goes_up_again(tmp_path: Path) -> None:
    mod = _load()
    runs = _runs(tmp_path)
    hub = _FakeHub()
    _push(mod, hub, runs)
    _save(runs / "v2_align" / "checkpoints", "best.pt", b"B" * 300, 200)  # a new best, same size
    (runs / "v2_align" / "train.log").write_text("log, then more", encoding="utf-8")
    _push(mod, hub, runs)
    assert hub.commits[-2:] == [BEST, ["v2_align/train.log"]]
    assert hub.files[BEST[0]] == b"B" * 300 and json.loads(hub.files[BEST[1]]) == {"step": 200}
    assert hub.squashes == 1  # the replaced version would keep using quota


def test_last_follows_new_steps_and_a_pinned_upload_survives_pruning(tmp_path: Path) -> None:
    mod = _load()
    runs = _runs(tmp_path)
    ck = runs / "v2_align" / "checkpoints"
    hub = _FakeHub()
    _push(mod, hub, runs)
    _save_step(ck, 200, b"t" * 300)
    # while step 200 uploads, the trainer saves step 300 and prunes step 200
    hub.during_commit = lambda: _save_step(ck, 300, b"v" * 300)
    _push(mod, hub, runs)
    assert not (ck / "step_0000200.pt").exists()
    assert hub.files[LAST[0]] == b"t" * 300 and json.loads(hub.files[LAST[1]]) == {"step": 200}
    assert not any((runs / mod.PINS).iterdir())
    hub.during_commit = None
    _push(mod, hub, runs)
    assert hub.files[LAST[0]] == b"v" * 300 and json.loads(hub.files[LAST[1]]) == {"step": 300}
    assert hub.squashes == 2


def test_a_checkpoint_waits_for_its_sidecar(tmp_path: Path) -> None:
    mod = _load()
    runs = _runs(tmp_path)
    ck = runs / "v2_align" / "checkpoints"
    hub = _FakeHub()
    _push(mod, hub, runs)
    _replace(ck / "best.pt", b"N" * 300)  # the trainer has not written the new sidecar yet
    _push(mod, hub, runs)
    assert hub.files[BEST[0]] == b"b" * 300  # the old pair stays until the new one is complete
    (ck / "best.pt.meta.json").write_text(json.dumps({"step": 300}), encoding="utf-8")
    _push(mod, hub, runs)
    assert hub.files[BEST[0]] == b"N" * 300 and json.loads(hub.files[BEST[1]]) == {"step": 300}


def test_push_refuses_a_public_repo(tmp_path: Path) -> None:
    mod = _load()
    runs = _runs(tmp_path)
    with pytest.raises(SystemExit, match="public"):
        _push(mod, _FakeHub(private=False), runs)


def test_check_creates_a_missing_repo_private_and_refuses_read_only_tokens() -> None:
    mod = _load()
    hub = _FakeHub(exists=False)
    assert mod.check(hub, "me/ckpt") == "me" and hub.exists
    with pytest.raises(SystemExit, match="read-only"):
        mod.check(_FakeHub(role="read"), "me/ckpt")


def test_watch_retries_failed_passes_and_runs_one_last_pass_on_stop(tmp_path: Path) -> None:
    mod = _load()
    runs = _runs(tmp_path)
    hub = _FakeHub()
    stop = mod.Stop()
    calls = []

    def flaky(**kw: Any) -> None:
        calls.append(1)
        if len(calls) == 1:
            stop.requested = True  # SIGTERM arrives during the first pass, which then fails
            raise ConnectionError("Hub unreachable")

    hub.during_commit = lambda: flaky()
    assert mod.watch(mod.Uploader(hub, "me/ckpt", runs), "v2_*", 3600.0, stop) == 0
    assert hub.commits == [LAST, BEST, UPCYCLED, sorted([*LOGS, "README.md"])]  # the last pass


def test_one_uploader_per_runs_directory(tmp_path: Path) -> None:
    mod = _load()
    runs = _runs(tmp_path)
    with mod._exclusive(runs), pytest.raises(SystemExit, match="another backup"):
        with mod._exclusive(runs):
            pass


def test_dry_run_needs_no_network(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load()
    runs = _runs(tmp_path)
    assert mod.main(["push", "--repo", "me/ckpt", "--runs", str(runs), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "v2_moe_upcycled/upcycled.pt" in out and "8 files" in out and "step_" not in out
    assert "8 new or changed" in out
