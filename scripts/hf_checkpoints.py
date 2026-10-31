#!/usr/bin/env python
"""Back up pretraining runs to a private Hugging Face model repo, and restore them on another machine.

    python scripts/hf_checkpoints.py check --repo USER/NAME
    python scripts/hf_checkpoints.py push  --repo USER/NAME [--runs runs] [--pattern 'v2_*'] [--with-last] [--dry-run]
    python scripts/hf_checkpoints.py watch --repo USER/NAME [--runs runs] [--interval 300] [--log-file F]
    python scripts/hf_checkpoints.py pull  --repo USER/NAME [--runs runs]

``push`` uploads, from every run directory matching ``--pattern``, everything but the step checkpoints:
``checkpoints/best.pt``, ``upcycled.pt``, configs, logs, metrics, routing reports and TensorBoard events.
``--with-last`` adds ``checkpoints/last.pt`` (the newest step, optimizer state included) so training can
resume elsewhere. ``watch`` runs alongside training (``scripts/gpu/5c_autobackup.sh``) and pushes what
changed every ``--interval`` seconds, ``last.pt`` included, so a dead machine loses at most the
checkpoint being uploaded; SIGTERM, SIGHUP or Ctrl-C makes it run one last pass and exit, and a failed
pass (network, Hub) is retried at the next one. ``check`` verifies, before a long run, that the login can
upload to the repo. The repo layout mirrors ``runs/``, so ``pull`` restores it in place (``last.pt``
comes back as a plain file, which ``--resume auto`` takes).

Each checkpoint goes up in a commit of its own together with its ``.meta.json`` sidecar, from a hard link
that pins the version being uploaded (the trainer may prune or replace the file meanwhile); the other
files go in one commit, read at once. The uploaded file versions (target, inode, size, mtime) are kept in
``runs/.hf_backup.json``: only new or changed files go up, and an interrupted push is rerun as is. When a
checkpoint replaces one already on the Hub, the repo history is squashed, or every old version would keep
counting against the storage quota (the Hub frees it within ~36 h).

The repo is created private and its visibility is checked: a public repo is refused (the runs derive from
data such as PhoMT that must not be redistributed). The token is the one of ``hf auth login``; pushing
needs write access to the repo.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

log = logging.getLogger("hf_checkpoints")
REPO_TYPE = "model"
STATE = ".hf_backup.json"  # file versions already uploaded (hidden: never uploaded itself)
LOCK = ".hf_backup.lock"  # one uploader per runs/ directory
PINS = ".hf_backup_pins"  # hard links to the checkpoint versions being uploaded
SIDECAR = ".meta.json"
SIDECAR_GRACE_S = 60  # the trainer writes a sidecar right after its checkpoint
README = """# ViVL-MoE v2 pretraining runs (private backup)

Mirror of `runs/` on the training machine: `best.pt` of every stage, `upcycled.pt`, configs, logs and
metrics, and `last.pt` (resumable state) when pushed with `--with-last` or by `watch`. Restore into
`runs/` with

    python scripts/hf_checkpoints.py pull --repo {repo}
"""


def select_files(runs: Path, pattern: str, with_last: bool) -> list[Path]:
    """Files to back up, as paths under ``runs`` (step checkpoints and hidden files are left out)."""
    out: list[Path] = []
    for run in sorted(p for p in runs.glob(pattern) if p.is_dir()):
        for f in sorted(run.rglob("*")):
            rel = f.relative_to(runs)
            if any(part.startswith(".") for part in rel.parts) or not f.is_file():
                continue
            if f.parent.name == "checkpoints":
                if f.name.startswith("step_"):
                    continue
                if f.name.startswith("last.pt") and not with_last:
                    continue
            out.append(rel)
    return out


def _version(name: str, st: os.stat_result) -> str:
    return f"{name}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}"


def version(path: Path) -> str:
    """Identity of a file version: the file a symlink points to (``last.pt`` -> its step), inode, size
    and mtime (the trainer replaces checkpoints by ``os.replace``: a new inode each time)."""
    return _version(path.resolve().name, path.stat())


def _complete_json(data: bytes) -> bool:
    try:
        json.loads(data)
    except ValueError:
        return False
    return True


def ensure_private(api: Any, repo_id: str) -> None:
    # create only when missing: a fine-grained token may write to the repo without creating repos
    if not api.repo_exists(repo_id, repo_type=REPO_TYPE):
        api.create_repo(repo_id, repo_type=REPO_TYPE, private=True)
    if not api.repo_info(repo_id, repo_type=REPO_TYPE).private:
        raise SystemExit(f"{repo_id} is a public repo: refusing to upload pretraining runs there")


def check(api: Any, repo_id: str) -> str:
    """Fail before a long run when the login cannot upload to the private ``repo_id``; the user name."""
    from huggingface_hub.errors import LocalTokenNotFoundError

    try:
        who = api.whoami()  # an invalid token raises here too
    except LocalTokenNotFoundError as e:
        raise SystemExit("not logged in to Hugging Face: run `hf auth login`") from e
    role = ((who.get("auth") or {}).get("accessToken") or {}).get("role")
    if role == "read":
        raise SystemExit(
            f"the Hugging Face token is read-only: log in with one that can write to {repo_id}"
        )
    ensure_private(api, repo_id)
    return str(who.get("name"))


def remote_sizes(api: Any, repo_id: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for item in api.list_repo_tree(repo_id, repo_type=REPO_TYPE, recursive=True):
        size = getattr(item, "size", None)
        if size is not None:
            sizes[item.path] = int(size)
    return sizes


def _sidecar(sidecar: Path, ckpt: os.stat_result) -> tuple[bool, tuple[bytes, str] | None]:
    """(ready, (content, version) or None) for the sidecar of a pinned checkpoint.

    A sidecar older than its checkpoint belongs to the version being replaced (``best.pt``), and an
    unparsable one is being written: wait for the next pass, unless the checkpoint is old enough that its
    sidecar will not come (then it goes up as it is, or without one).
    """
    settled = time.time() - ckpt.st_mtime > SIDECAR_GRACE_S
    try:
        st = sidecar.stat()
        data = sidecar.read_bytes()
    except FileNotFoundError:
        return settled, None
    if not _complete_json(data):
        return settled, None
    if st.st_mtime_ns < ckpt.st_mtime_ns and not settled:
        return False, None
    return True, (data, _version(sidecar.name, st))


@contextmanager
def _exclusive(runs: Path) -> Iterator[None]:
    """One uploader per ``runs`` directory (``watch`` holds it for its lifetime)."""
    import fcntl

    runs.mkdir(parents=True, exist_ok=True)
    with (runs / LOCK).open("a+", encoding="utf-8") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.seek(0)
            holder = fh.read().strip() or "?"
            raise SystemExit(f"another backup of {runs} is running (pid {holder})") from None
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()}\n")
        fh.flush()
        shutil.rmtree(runs / PINS, ignore_errors=True)  # links left by a killed uploader
        yield


class Uploader:
    """Pushes the new or changed files of ``runs`` to a private repo (see the module docstring)."""

    def __init__(self, api: Any, repo_id: str, runs: Path) -> None:
        self.api, self.repo_id, self.runs = api, repo_id, runs
        self.state = self._load_state()

    def _load_state(self) -> dict[str, str]:
        try:
            data = json.loads((self.runs / STATE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return dict(data.get("files", {})) if data.get("repo") == self.repo_id else {}

    def _save_state(self) -> None:
        path = self.runs / STATE
        tmp = path.with_name(path.name + ".tmp")
        payload = {"repo": self.repo_id, "files": self.state}
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def pending(self, files: list[Path]) -> list[Path]:
        """The files whose current version has not been uploaded yet."""
        out = []
        for rel in files:
            try:
                current = version(self.runs / rel)
            except FileNotFoundError:  # pruned since it was listed
                continue
            if self.state.get(rel.as_posix()) != current:
                out.append(rel)
        return out

    def push(self, files: list[Path]) -> dict[str, int]:
        todo = self.pending(files)
        if not todo:
            return {"selected": len(files), "uploaded": 0}
        ensure_private(self.api, self.repo_id)
        remote = remote_sizes(self.api, self.repo_id)
        # checkpoints first, last.pt leading: the resumable state is what a dead machine would cost
        ckpts = sorted(
            (rel for rel in todo if rel.suffix == ".pt"),
            key=lambda rel: (rel.name != "last.pt", rel.as_posix()),
        )
        sidecars = {rel.with_name(rel.name + SIDECAR) for rel in ckpts}
        rest = [rel for rel in todo if rel.suffix != ".pt" and rel not in sidecars]
        log.info(
            "%d files selected, %d new or changed (%d checkpoints) -> %s",
            len(files),
            len(todo),
            len(ckpts),
            self.repo_id,
        )
        uploaded, replaced = 0, False
        for rel in ckpts:  # before the logs: a failing commit must not hold them back
            n = self._push_checkpoint(rel)
            uploaded += n
            replaced |= n > 0 and rel.as_posix() in remote
        if replaced:
            self._squash()
        uploaded += self._push_files(rest, readme="README.md" not in remote)
        return {"selected": len(files), "uploaded": uploaded}

    def _commit(self, ops: list[Any], message: str, versions: dict[str, str]) -> None:
        self.api.create_commit(
            self.repo_id, operations=ops, commit_message=message, repo_type=REPO_TYPE
        )
        self.state.update(versions)
        self._save_state()

    def _push_files(self, rels: list[Path], readme: bool) -> int:
        """Logs, metrics, configs: one commit, each file read at once (training keeps appending)."""
        from huggingface_hub import CommitOperationAdd

        ops, versions = [], {}
        for rel in rels:
            path = self.runs / rel
            try:
                current, data = version(path), path.read_bytes()
            except FileNotFoundError:
                continue
            if path.suffix == ".json" and not _complete_json(data):  # being written: next pass
                continue
            ops.append(CommitOperationAdd(rel.as_posix(), data))
            versions[rel.as_posix()] = current
        if readme:
            text = README.format(repo=self.repo_id).encode("utf-8")
            ops.append(CommitOperationAdd("README.md", text))
        if ops:
            self._commit(ops, f"logs and metrics ({len(versions)} files)", versions)
        return len(versions)

    @contextmanager
    def _pin(self, path: Path) -> Iterator[tuple[Path, Path] | None]:
        """(hard link to the current version of ``path``, its target), or None once it is gone."""
        try:
            target = path.resolve(strict=True)
        except FileNotFoundError:  # pruned since it was listed: the next pass takes its successor
            yield None
            return
        pins = self.runs / PINS
        pins.mkdir(exist_ok=True)
        link = pins / target.name
        link.unlink(missing_ok=True)
        try:
            os.link(target, link)
        except FileNotFoundError:
            yield None
            return
        except OSError as e:  # no hard links on this filesystem
            log.warning("cannot pin %s (%s): uploading it in place", target, e)
            yield target, target
            return
        try:
            yield link, target
        finally:
            link.unlink(missing_ok=True)

    def _push_checkpoint(self, rel: Path) -> int:
        """One commit: the pinned checkpoint and its sidecar. Number of files uploaded (0 = later)."""
        from huggingface_hub import CommitOperationAdd

        with self._pin(self.runs / rel) as pin:
            if pin is None:
                return 0
            pinned, target = pin
            st = pinned.stat()
            ready, meta = _sidecar(target.with_name(target.name + SIDECAR), st)
            if not ready:
                log.info("%s: sidecar not written yet, next pass", rel)
                return 0
            ops = [CommitOperationAdd(rel.as_posix(), str(pinned))]
            versions = {rel.as_posix(): _version(target.name, st)}
            if meta is not None:
                ops.append(CommitOperationAdd(rel.as_posix() + SIDECAR, meta[0]))
                versions[rel.as_posix() + SIDECAR] = meta[1]
            log.info("uploading %s (%s, %.1f GB)", rel, target.name, st.st_size / 1e9)
            start = time.monotonic()
            self._commit(ops, f"{rel.as_posix()} ({target.name})", versions)
        log.info("uploaded %s in %.0f s", rel, time.monotonic() - start)
        return len(ops)

    def _squash(self) -> None:
        try:
            self.api.super_squash_history(self.repo_id, repo_type=REPO_TYPE)
        except Exception as e:  # the backup itself went through
            log.warning(
                "could not squash %s (%s): old checkpoints still use quota", self.repo_id, e
            )
        else:
            log.info(
                "squashed %s: replaced checkpoints are freed (quota within ~36 h)", self.repo_id
            )


class Stop:
    """Set on SIGTERM / SIGHUP / Ctrl-C: finish the current pass, run one last pass, exit."""

    requested = False


def watch(uploader: Uploader, pattern: str, interval: float, stop: Stop) -> int:
    """Push what changed every ``interval`` seconds until ``stop``; 1 when the last pass failed."""
    log.info(
        "watching %s/%s every %.0f s -> %s (private)",
        uploader.runs,
        pattern,
        interval,
        uploader.repo_id,
    )
    while True:
        final = stop.requested
        try:
            files = select_files(uploader.runs, pattern, with_last=True)
            if files:
                uploader.push(files)
            failed = False
        except Exception as e:  # network / Hub hiccup: the next pass retries, training goes on
            failed = True
            if final:
                log.error("last backup pass failed (%s: %s)", type(e).__name__, e)
            else:
                log.warning(
                    "backup pass failed (%s: %s); retry in %.0f s", type(e).__name__, e, interval
                )
        if final:
            return int(failed)
        deadline = time.monotonic() + interval
        while not stop.requested and time.monotonic() < deadline:
            time.sleep(min(1.0, max(deadline - time.monotonic(), 0.0)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="verify that the login can upload to the private repo")
    c.add_argument("--repo", required=True, help="USER/NAME of the private model repo")
    p = sub.add_parser("push", help="upload what changed since the last push (resumable)")
    p.add_argument("--repo", required=True, help="USER/NAME of the private model repo")
    p.add_argument("--runs", default="runs")
    p.add_argument(
        "--pattern", default="v2_*", help="run directories to back up (glob under --runs)"
    )
    p.add_argument(
        "--with-last", action="store_true", help="also checkpoints/last.pt (resumable state)"
    )
    p.add_argument("--dry-run", action="store_true", help="list what would be uploaded; no network")
    w = sub.add_parser("watch", help="push what changes (last.pt included) until SIGTERM / Ctrl-C")
    w.add_argument("--repo", required=True, help="USER/NAME of the private model repo")
    w.add_argument("--runs", default="runs")
    w.add_argument("--pattern", default="v2_*")
    w.add_argument("--interval", type=float, default=300.0, help="seconds between two passes")
    w.add_argument("--log-file", help="also append the log to this file")
    q = sub.add_parser("pull", help="restore the runs from the repo")
    q.add_argument("--repo", required=True)
    q.add_argument("--runs", default="runs")
    args = ap.parse_args(argv)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if getattr(args, "log_file", None):
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers
    )

    if args.cmd == "pull":
        from huggingface_hub import snapshot_download

        path = snapshot_download(args.repo, repo_type=REPO_TYPE, local_dir=args.runs)
        log.info("restored %s into %s", args.repo, path)
        return 0
    if args.cmd == "check":
        from huggingface_hub import HfApi

        log.info("ok: %s can upload to %s (private)", check(HfApi(), args.repo), args.repo)
        return 0

    runs = Path(args.runs)
    if args.cmd == "push":
        files = select_files(runs, args.pattern, args.with_last)
        if not files:
            raise SystemExit(f"no run directory matching {args.pattern!r} under {runs}")
        total = sum((runs / rel).stat().st_size for rel in files)
        if args.dry_run:
            pending = set(Uploader(None, args.repo, runs).pending(files))
            for rel in files:
                mark = "*" if rel in pending else " "
                print(f"{mark} {(runs / rel).stat().st_size / 1e6:12.1f} MB  {rel.as_posix()}")
            print(
                f"{len(files)} files, {total / 1e9:.1f} GB -> {args.repo} (private); "
                f"{len(pending)} new or changed (*)"
            )
            return 0
        from huggingface_hub import HfApi

        with _exclusive(runs):
            result = Uploader(HfApi(), args.repo, runs).push(files)
        log.info("done: %s (%.1f GB selected)", result, total / 1e9)
        return 0

    from huggingface_hub import HfApi
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()  # a background log, not a terminal
    stop = Stop()

    def on_signal(signum: int, frame: Any) -> None:
        if stop.requested and signum == signal.SIGINT:
            raise KeyboardInterrupt  # second Ctrl-C: give up the last pass
        log.info("signal %d: one last pass, then exit", signum)
        stop.requested = True

    signal.signal(signal.SIGTERM, on_signal)
    # started in the background by a script, SIGINT is ignored: leave it so
    for signum in (signal.SIGINT, signal.SIGHUP):
        if signal.getsignal(signum) is not signal.SIG_IGN:
            signal.signal(signum, on_signal)
    # no `check` here (the scripts run it first): a Hub hiccup at start must not end the backup
    with _exclusive(runs):
        return watch(Uploader(HfApi(), args.repo, runs), args.pattern, args.interval, stop)


if __name__ == "__main__":
    sys.exit(main())
