"""Ship the pretraining corpus to and from Hugging Face dataset repos as tar shards.

Hundreds of thousands of loose files are a poor fit for the Hub (one request per file, slow
listings), so the images the corpus uses are packed into ~2 GB uncompressed tars. :func:`push`
uploads one shard while the next one is packed (constant RAM, two shards of scratch disk) and
skips shards already in the repo, so an interrupted upload resumes. :func:`pull` downloads and
unpacks shards in parallel with strict member checks, skipping shards whose images are present.

Extra folders (PhoMT, KTVIC, UIT-ViIC) carry their own licenses - PhoMT forbids redistribution -
so :func:`push_extras` only ever writes to a private repo, verified after creation.

Repo layout::

    shards.json                    the plan: shard -> image paths, corpus sha1
    images/shard-00000.tar ...     members images/<name>
    corpus/pretrain_corpus.jsonl   + pretrain_corpus.report.json
    README.md
    extras/<name>.tar              members <name>/... (extras repo, always private)

``api`` is a :class:`huggingface_hub.HfApi` or a test double with its ``create_repo``,
``repo_info``, ``list_repo_files``, ``upload_file`` and ``hf_hub_download`` methods. Hidden files
and folders (``.env``, ``.git``, ``.DS_Store``) are never packed.
"""

from __future__ import annotations

import json
import logging
import os
import posixpath
import shutil
import tarfile
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vivlmoe.data.corpus_prep import file_sha1

log = logging.getLogger(__name__)

INDEX = "shards.json"
REPO_TYPE = "dataset"
CORPUS_FILES = ("corpus/pretrain_corpus.jsonl", "corpus/pretrain_corpus.report.json")
_INDEX_VERSION = 1


@dataclass(frozen=True)
class Shard:
    path_in_repo: str
    files: tuple[str, ...]  # paths relative to the dataset root = tar member names
    bytes: int


def corpus_images(corpus: Path, image_key: str = "image") -> list[str]:
    """Sorted unique image paths (relative to the dataset root) of a corpus JSONL."""
    seen: set[str] = set()
    with corpus.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                seen.add(json.loads(line)[image_key])
    return sorted(seen)


def plan_shards(
    root: Path, rel_paths: Sequence[str], target_bytes: int, prefix: str = "images/shard"
) -> list[Shard]:
    """Cut ``rel_paths`` (in order) into shards of about ``target_bytes``; a file never straddles."""
    shards: list[Shard] = []
    cur: list[str] = []
    size = 0
    for rel in rel_paths:
        n = (root / rel).stat().st_size
        if cur and size + n > target_bytes:
            shards.append(Shard(f"{prefix}-{len(shards):05d}.tar", tuple(cur), size))
            cur, size = [], 0
        cur.append(rel)
        size += n
    if cur:
        shards.append(Shard(f"{prefix}-{len(shards):05d}.tar", tuple(cur), size))
    return shards


def build_index(corpus: Path, shards: Sequence[Shard]) -> dict[str, Any]:
    return {
        "version": _INDEX_VERSION,
        "corpus_sha1": file_sha1(corpus),
        "images": sum(len(s.files) for s in shards),
        "bytes": sum(s.bytes for s in shards),
        "shards": [
            {"path": s.path_in_repo, "bytes": s.bytes, "files": list(s.files)} for s in shards
        ],
    }


def _member(name: str, size: int, mtime: float) -> tarfile.TarInfo:
    """A regular-file header without owner information."""
    ti = tarfile.TarInfo(name)
    ti.size, ti.mtime, ti.mode, ti.type = size, int(mtime), 0o644, tarfile.REGTYPE
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    return ti


def write_tar(out: Path, items: Iterable[tuple[Path, str]]) -> int:
    """Stream ``(source file, member name)`` pairs into an uncompressed tar; returns the count."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    n = 0
    with tarfile.open(tmp, "w", format=tarfile.PAX_FORMAT) as tar:
        for src, name in items:
            st = src.stat()
            with src.open("rb") as fh:
                tar.addfile(_member(name, st.st_size, st.st_mtime), fh)
            n += 1
    os.replace(tmp, out)
    return n


def dir_items(src: Path, prefix: str) -> list[tuple[Path, str]]:
    """Regular files under ``src`` as ``<prefix>/<relative path>``, skipping hidden entries."""
    items: list[tuple[Path, str]] = []
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for f in sorted(filenames):
            p = Path(dirpath) / f
            if f.startswith(".") or p.is_symlink() or not p.is_file():
                continue
            items.append((p, posixpath.join(prefix, p.relative_to(src).as_posix())))
    return items


def safe_extract(tar_path: Path, dest: Path, prefix: str) -> list[str]:
    """Extract regular files named ``<prefix>/...``; any other member (links, absolute or ``..``
    paths, other folders) aborts the extraction."""
    names: list[str] = []
    with tarfile.open(tar_path, "r:") as tar:
        for m in tar:
            name = m.name
            if (
                not m.isreg()
                or name.startswith("/")
                or posixpath.normpath(name) != name
                or ".." in name.split("/")
                or not name.startswith(prefix + "/")
            ):
                raise ValueError(f"{tar_path}: refusing member {name!r} (type {m.type!r})")
            src = tar.extractfile(m)
            assert src is not None  # regular file
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            with tmp.open("wb") as fh:
                shutil.copyfileobj(src, fh, 1 << 20)
            os.replace(tmp, target)
            names.append(name)
    return names


def _upload(
    api: Any, repo_id: str, path_or_bytes: Path | bytes, path_in_repo: str, msg: str
) -> None:
    obj = path_or_bytes if isinstance(path_or_bytes, bytes) else str(path_or_bytes)
    api.upload_file(
        path_or_fileobj=obj,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        commit_message=msg,
    )


def ensure_repo(api: Any, repo_id: str, private: bool) -> None:
    """Create the dataset repo if needed and refuse one whose visibility differs."""
    api.create_repo(repo_id, repo_type=REPO_TYPE, private=private, exist_ok=True)
    actual = bool(api.repo_info(repo_id, repo_type=REPO_TYPE).private)
    if actual != private:
        want, got = ("private", "public") if private else ("public", "private")
        raise RuntimeError(f"{repo_id} exists as a {got} repo, expected {want}")


def push(
    api: Any,
    repo_id: str,
    root: Path,
    shards: Sequence[Shard],
    work_dir: Path,
    readme: str | None = None,
    private: bool = True,
) -> dict[str, int]:
    """Upload what the repo lacks: index, image shards, corpus files, README."""
    index = build_index(root / CORPUS_FILES[0], shards)
    ensure_repo(api, repo_id, private)
    remote = set(api.list_repo_files(repo_id, repo_type=REPO_TYPE))
    if INDEX in remote:
        with open(api.hf_hub_download(repo_id, INDEX, repo_type=REPO_TYPE)) as fh:
            old = json.load(fh)
        if old.get("corpus_sha1") != index["corpus_sha1"] or old.get("shards") != index["shards"]:
            raise RuntimeError(
                f"{repo_id} holds a different corpus / shard plan; push to a new repo instead"
            )
    else:
        _upload(api, repo_id, json.dumps(index).encode("utf-8"), INDEX, "shard plan")
    todo = [s for s in shards if s.path_in_repo not in remote]
    done = {"shards_uploaded": 0, "shards_skipped": len(shards) - len(todo), "bytes_uploaded": 0}
    work_dir.mkdir(parents=True, exist_ok=True)

    def pack(s: Shard) -> Path:
        tar = work_dir / posixpath.basename(s.path_in_repo)
        write_tar(tar, [(root / f, f) for f in s.files])
        return tar

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=1) as packer:  # packs shard i+1 while shard i uploads
        nxt: Future[Path] | None = packer.submit(pack, todo[0]) if todo else None
        try:
            for i, s in enumerate(todo):
                assert nxt is not None
                tar = nxt.result()
                nxt = packer.submit(pack, todo[i + 1]) if i + 1 < len(todo) else None
                _upload(
                    api, repo_id, tar, s.path_in_repo, f"{s.path_in_repo} ({len(s.files)} images)"
                )
                tar.unlink()
                done["shards_uploaded"] += 1
                done["bytes_uploaded"] += s.bytes
                rate = done["bytes_uploaded"] / max(1e-9, time.time() - t0) / 1e6
                log.info(
                    "uploaded %s (%d/%d done, %.1f MB/s)", s.path_in_repo, i + 1, len(todo), rate
                )
        except BaseException:
            if nxt is not None:
                nxt.cancel()
            packer.shutdown(wait=True)
            for leftover in work_dir.glob("shard-*.tar*"):  # this function's own scratch shards
                leftover.unlink(missing_ok=True)
            raise
    for rel in CORPUS_FILES:
        if rel not in remote:
            _upload(api, repo_id, root / rel, rel, rel)
    if readme is not None and "README.md" not in remote:
        _upload(api, repo_id, readme.encode("utf-8"), "README.md", "dataset card")
    return done


def push_extras(api: Any, repo_id: str, extras: dict[str, Path], work_dir: Path) -> list[str]:
    """Upload each folder as ``extras/<name>.tar`` to a repo that is, and stays, private."""
    ensure_repo(api, repo_id, private=True)
    remote = set(api.list_repo_files(repo_id, repo_type=REPO_TYPE))
    work_dir.mkdir(parents=True, exist_ok=True)
    uploaded: list[str] = []
    for name, src in sorted(extras.items()):
        path = f"extras/{name}.tar"
        if path in remote:
            continue
        tar = work_dir / f"{name}.tar"
        try:
            write_tar(tar, dir_items(src, name))
            _upload(api, repo_id, tar, path, f"extra {name}")
        finally:
            tar.unlink(missing_ok=True)
        uploaded.append(path)
        log.info("uploaded %s to %s", path, repo_id)
    return uploaded


def pull(api: Any, repo_id: str, root: Path, workers: int = 4) -> dict[str, int]:
    """Restore ``<root>/images`` and ``<root>/corpus`` from an image repo."""
    root.mkdir(parents=True, exist_ok=True)
    dl = root / ".hf_download"
    files = set(api.list_repo_files(repo_id, repo_type=REPO_TYPE))
    with open(api.hf_hub_download(repo_id, INDEX, repo_type=REPO_TYPE, local_dir=dl)) as fh:
        index = json.load(fh)
    for rel in CORPUS_FILES:
        if rel in files:
            got = api.hf_hub_download(repo_id, rel, repo_type=REPO_TYPE, local_dir=dl)
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            os.replace(got, root / rel)
    todo = [s for s in index["shards"] if not all((root / f).is_file() for f in s["files"])]
    log.info("%d / %d image shards to fetch", len(todo), len(index["shards"]))

    def fetch(shard: dict[str, Any]) -> int:
        tar = Path(api.hf_hub_download(repo_id, shard["path"], repo_type=REPO_TYPE, local_dir=dl))
        names = safe_extract(tar, root, "images")
        missing = set(shard["files"]) - set(names)
        if missing:
            raise RuntimeError(
                f"{shard['path']} lacks {len(missing)} images, e.g. {sorted(missing)[:3]}"
            )
        tar.unlink()
        log.info("restored %s (%d images)", shard["path"], len(names))
        return len(names)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        restored = sum(ex.map(fetch, todo))
    shutil.rmtree(dl, ignore_errors=True)  # this function's own download scratch
    return {"shards_fetched": len(todo), "images_restored": restored}


def pull_extras(api: Any, repo_id: str, extras_dir: Path) -> list[str]:
    """Unpack every ``extras/<name>.tar`` of ``repo_id`` into ``extras_dir/<name>`` (skip existing)."""
    files = set(api.list_repo_files(repo_id, repo_type=REPO_TYPE))
    dl = extras_dir / ".hf_download"
    restored: list[str] = []
    for path in sorted(f for f in files if f.startswith("extras/") and f.endswith(".tar")):
        name = posixpath.basename(path)[: -len(".tar")]
        if (extras_dir / name).exists():
            continue
        tar = Path(api.hf_hub_download(repo_id, path, repo_type=REPO_TYPE, local_dir=dl))
        safe_extract(tar, extras_dir, name)
        tar.unlink()
        restored.append(name)
        log.info("restored %s -> %s", path, extras_dir / name)
    shutil.rmtree(dl, ignore_errors=True)  # this function's own download scratch
    return restored


__all__ = [
    "CORPUS_FILES",
    "INDEX",
    "Shard",
    "build_index",
    "corpus_images",
    "dir_items",
    "ensure_repo",
    "plan_shards",
    "pull",
    "pull_extras",
    "push",
    "push_extras",
    "safe_extract",
    "write_tar",
]
