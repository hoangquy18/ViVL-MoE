from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from vivlmoe.data.corpus_shards import (
    CORPUS_FILES,
    INDEX,
    corpus_images,
    plan_shards,
    pull,
    pull_extras,
    push,
    push_extras,
    safe_extract,
)


class FakeApi:
    """In-memory stand-in for the ``HfApi`` methods the module uses (one dict per repo)."""

    def __init__(self, cache: Path, fail_on: str | None = None) -> None:
        self.cache = cache
        self.repos: dict[str, dict[str, bytes]] = {}
        self.private: dict[str, bool] = {}
        self.uploads: list[str] = []
        self.downloads: list[str] = []
        self.fail_on = fail_on

    def create_repo(self, repo_id: str, repo_type: str, private: bool, exist_ok: bool) -> None:
        if repo_id not in self.repos:  # exist_ok keeps the existing visibility, like the Hub
            self.repos[repo_id], self.private[repo_id] = {}, private

    def repo_info(self, repo_id: str, repo_type: str) -> SimpleNamespace:
        return SimpleNamespace(private=self.private[repo_id])

    def list_repo_files(self, repo_id: str, repo_type: str) -> list[str]:
        return sorted(self.repos[repo_id])

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        if path_in_repo == self.fail_on:
            raise ConnectionError("upload interrupted")
        data = (
            path_or_fileobj
            if isinstance(path_or_fileobj, bytes)
            else Path(path_or_fileobj).read_bytes()
        )
        self.repos[repo_id][path_in_repo] = data
        self.uploads.append(path_in_repo)

    def hf_hub_download(self, repo_id, filename, repo_type, local_dir=None) -> str:
        out = Path(local_dir or self.cache) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.repos[repo_id][filename])
        self.downloads.append(filename)
        return str(out)


def _dataset(root: Path, sizes: dict[str, int]) -> None:
    (root / "images").mkdir(parents=True)
    lines = []
    for i, (name, size) in enumerate(sizes.items()):
        (root / "images" / name).write_bytes(bytes([i % 251]) * size)
        lines.append(json.dumps({"image": f"images/{name}", "caption": f"c{i}", "source": "x"}))
        lines.append(json.dumps({"image": f"images/{name}", "caption": f"d{i}", "source": "x"}))
    (root / "corpus").mkdir()
    (root / CORPUS_FILES[0]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / CORPUS_FILES[1]).write_text(json.dumps({"totals": {}}), encoding="utf-8")


def _extra(d: Path) -> Path:
    (d / "sub").mkdir(parents=True)
    (d / "a.txt").write_text("a", encoding="utf-8")
    (d / "sub" / "b.json").write_text("{}", encoding="utf-8")
    (d / ".env").write_text("SECRET=1", encoding="utf-8")
    (d / ".git").mkdir()
    (d / ".git" / "config").write_text("x", encoding="utf-8")
    return d


SIZES = {"a_1.jpg": 300, "b_2.jpg": 500, "c_3.png": 400, "d_4.jpg": 100}


def _shards(root: Path, target: int = 800):
    return plan_shards(root, corpus_images(root / CORPUS_FILES[0]), target_bytes=target)


def test_plan_shards_cuts_by_size_and_never_splits_a_file(tmp_path: Path) -> None:
    _dataset(tmp_path, SIZES)
    rel = corpus_images(tmp_path / CORPUS_FILES[0])
    assert rel == [f"images/{n}" for n in sorted(SIZES)]  # unique, sorted
    shards = plan_shards(tmp_path, rel, target_bytes=800)
    assert [s.files for s in shards] == [
        ("images/a_1.jpg", "images/b_2.jpg"),
        ("images/c_3.png", "images/d_4.jpg"),
    ]
    assert [s.bytes for s in shards] == [800, 500]
    assert shards[1].path_in_repo == "images/shard-00001.tar"
    assert len(plan_shards(tmp_path, rel, target_bytes=10)) == 4  # every file above the target


def test_public_images_private_extras_round_trip(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    _dataset(src, SIZES)
    api = FakeApi(tmp_path / "cache")
    assert push_extras(api, "u/ext", {"PhoMT": _extra(tmp_path / "PhoMT")}, tmp_path / "work") == [
        "extras/PhoMT.tar"
    ]
    done = push(api, "u/img", src, _shards(src), tmp_path / "work", readme="# card", private=False)
    assert done["shards_uploaded"] == 2
    assert api.private == {"u/ext": True, "u/img": False}
    assert set(api.repos["u/img"]) == {
        INDEX,
        "images/shard-00000.tar",
        "images/shard-00001.tar",
        *CORPUS_FILES,
        "README.md",
    }
    with tarfile.open(fileobj=io.BytesIO(api.repos["u/ext"]["extras/PhoMT.tar"])) as tar:
        assert sorted(tar.getnames()) == ["PhoMT/a.txt", "PhoMT/sub/b.json"]  # no .env / .git
        assert all(m.uname == "" and m.uid == 0 for m in tar.getmembers())
    assert not list((tmp_path / "work").iterdir())  # scratch shards deleted after upload

    assert pull(api, "u/img", dst, workers=2) == {"shards_fetched": 2, "images_restored": 4}
    for name in SIZES:
        assert (dst / "images" / name).read_bytes() == (src / "images" / name).read_bytes()
    assert (dst / CORPUS_FILES[0]).read_bytes() == (src / CORPUS_FILES[0]).read_bytes()
    assert not (dst / ".hf_download").exists()
    assert pull_extras(api, "u/ext", tmp_path / "extras") == ["PhoMT"]
    assert (tmp_path / "extras" / "PhoMT" / "sub" / "b.json").read_text(encoding="utf-8") == "{}"

    api.downloads.clear()
    assert pull(api, "u/img", dst)["shards_fetched"] == 0  # everything already on disk
    assert pull_extras(api, "u/ext", tmp_path / "extras") == []
    assert not any(d.endswith(".tar") for d in api.downloads)


def test_extras_never_go_to_a_public_repo(tmp_path: Path) -> None:
    api = FakeApi(tmp_path / "cache")
    api.create_repo("u/pub", repo_type="dataset", private=False, exist_ok=True)
    with pytest.raises(RuntimeError, match="public repo, expected private"):
        push_extras(api, "u/pub", {"PhoMT": _extra(tmp_path / "PhoMT")}, tmp_path / "work")
    assert api.repos["u/pub"] == {}


def test_push_resumes_and_refuses_a_different_plan(tmp_path: Path) -> None:
    _dataset(tmp_path, SIZES)
    api = FakeApi(tmp_path / "cache")
    shards = _shards(tmp_path)
    push(api, "u/r", tmp_path, shards, tmp_path / "work")
    assert api.private["u/r"] is True  # private unless asked otherwise
    del api.repos["u/r"]["images/shard-00001.tar"]  # interrupted before the last shard
    api.uploads.clear()
    done = push(api, "u/r", tmp_path, shards, tmp_path / "work")
    assert api.uploads == ["images/shard-00001.tar"] and done["shards_skipped"] == 1

    with (tmp_path / CORPUS_FILES[0]).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"image": "images/a_1.jpg", "caption": "new", "source": "x"}) + "\n")
    with pytest.raises(RuntimeError, match="different corpus"):
        push(api, "u/r", tmp_path, shards, tmp_path / "work")


def test_failed_upload_leaves_no_scratch_shards(tmp_path: Path) -> None:
    _dataset(tmp_path, SIZES)
    api = FakeApi(tmp_path / "cache", fail_on="images/shard-00000.tar")
    with pytest.raises(ConnectionError):
        push(api, "u/r", tmp_path, _shards(tmp_path, target=300), tmp_path / "work")
    assert not list((tmp_path / "work").glob("shard-*"))  # the shard packed ahead is gone too


@pytest.mark.parametrize(
    "name, kind",
    [
        ("images/../evil.jpg", tarfile.REGTYPE),
        ("/etc/evil", tarfile.REGTYPE),
        ("other/x.jpg", tarfile.REGTYPE),
        ("images/./x.jpg", tarfile.REGTYPE),
        ("images/link.jpg", tarfile.SYMTYPE),
    ],
)
def test_safe_extract_rejects_unsafe_members(tmp_path: Path, name: str, kind: bytes) -> None:
    tar_path = tmp_path / "bad.tar"
    with tarfile.open(tar_path, "w") as tar:
        ti = tarfile.TarInfo(name)
        ti.type = kind
        if kind == tarfile.SYMTYPE:
            ti.linkname = "/etc/passwd"
            tar.addfile(ti)
        else:
            ti.size = 1
            tar.addfile(ti, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="refusing member"):
        safe_extract(tar_path, tmp_path / "out", "images")
    assert not (tmp_path / "evil.jpg").exists()
