#!/usr/bin/env python
"""Back up pretraining runs to a private Hugging Face model repo, and restore them on another machine.

    python scripts/hf_checkpoints.py push --repo USER/NAME [--runs runs] [--pattern 'v2_*'] [--with-last] [--dry-run]
    python scripts/hf_checkpoints.py pull --repo USER/NAME [--runs runs]

``push`` uploads, from every run directory matching ``--pattern``, everything but the step checkpoints:
``checkpoints/best.pt`` (+ sidecar), ``upcycled.pt``, configs, logs, metrics, routing reports and
TensorBoard events. ``--with-last`` adds ``checkpoints/last.pt`` (the newest step, optimizer state included)
so training can resume elsewhere. The repo layout mirrors ``runs/``, so ``pull`` restores it in place.

The repo is created private and its visibility is checked: a public repo is refused (the runs derive from
data such as PhoMT that must not be redistributed). Files already in the repo with the same size are
skipped, so an interrupted push is rerun as is. Small files go in one commit, each checkpoint in its own.
The token is the one of ``hf auth login``; pushing needs write access to the repo.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("hf_checkpoints")
REPO_TYPE = "model"
BIG_BYTES = 100 << 20  # files above this get a commit of their own
README = """# ViVL-MoE v2 pretraining runs (private backup)

Mirror of `runs/` on the training machine: `best.pt` of every stage, `upcycled.pt`, configs, logs and
metrics (and `last.pt` when pushed with `--with-last`). Restore into `runs/` with

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


def ensure_private(api: Any, repo_id: str) -> None:
    api.create_repo(repo_id, repo_type=REPO_TYPE, private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type=REPO_TYPE).private:
        raise SystemExit(f"{repo_id} is a public repo: refusing to upload pretraining runs there")


def remote_sizes(api: Any, repo_id: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for item in api.list_repo_tree(repo_id, repo_type=REPO_TYPE, recursive=True):
        size = getattr(item, "size", None)
        if size is not None:
            sizes[item.path] = int(size)
    return sizes


def push(
    api: Any, repo_id: str, runs: Path, files: list[Path], big_bytes: int = BIG_BYTES
) -> dict[str, int]:
    from huggingface_hub import CommitOperationAdd

    ensure_private(api, repo_id)
    remote = remote_sizes(api, repo_id)
    todo = [rel for rel in files if remote.get(rel.as_posix()) != (runs / rel).stat().st_size]
    small = [rel for rel in todo if (runs / rel).stat().st_size < big_bytes]
    big = [rel for rel in todo if (runs / rel).stat().st_size >= big_bytes]
    log.info(
        "%d files selected, %d already in %s, %d to upload (%d checkpoints)",
        len(files),
        len(files) - len(todo),
        repo_id,
        len(todo),
        len(big),
    )
    ops = [CommitOperationAdd(rel.as_posix(), str(runs / rel)) for rel in small]
    if "README.md" not in remote:
        ops.append(CommitOperationAdd("README.md", README.format(repo=repo_id).encode("utf-8")))
    if ops:
        api.create_commit(
            repo_id,
            operations=ops,
            commit_message=f"logs and metrics ({len(small)} files)",
            repo_type=REPO_TYPE,
        )
    for rel in big:
        size_gb = (runs / rel).stat().st_size / 1e9
        log.info("uploading %s (%.1f GB)", rel, size_gb)
        api.upload_file(
            path_or_fileobj=str(runs / rel),  # a last.pt symlink uploads the step file it points to
            path_in_repo=rel.as_posix(),
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            commit_message=f"{rel.as_posix()}",
        )
    return {"selected": len(files), "uploaded": len(todo)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push", help="upload the runs (resumable)")
    p.add_argument("--repo", required=True, help="USER/NAME of the private model repo")
    p.add_argument("--runs", default="runs")
    p.add_argument(
        "--pattern", default="v2_*", help="run directories to back up (glob under --runs)"
    )
    p.add_argument(
        "--with-last", action="store_true", help="also checkpoints/last.pt (resumable state)"
    )
    p.add_argument("--dry-run", action="store_true", help="list what would be uploaded; no network")
    q = sub.add_parser("pull", help="restore the runs from the repo")
    q.add_argument("--repo", required=True)
    q.add_argument("--runs", default="runs")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runs = Path(args.runs)

    if args.cmd == "pull":
        from huggingface_hub import snapshot_download

        path = snapshot_download(args.repo, repo_type=REPO_TYPE, local_dir=str(runs))
        log.info("restored %s into %s", args.repo, path)
        return 0

    files = select_files(runs, args.pattern, args.with_last)
    if not files:
        raise SystemExit(f"no run directory matching {args.pattern!r} under {runs}")
    total = sum((runs / rel).stat().st_size for rel in files)
    if args.dry_run:
        for rel in files:
            print(f"{(runs / rel).stat().st_size / 1e6:12.1f} MB  {rel.as_posix()}")
        print(f"{len(files)} files, {total / 1e9:.1f} GB -> {args.repo} (private)")
        return 0
    from huggingface_hub import HfApi

    result = push(HfApi(), args.repo, runs, files)
    log.info("done: %s (%.1f GB selected)", result, total / 1e9)
    return 0


if __name__ == "__main__":
    sys.exit(main())
