from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.data.conftest import write_images
from vivlmoe.data.image_store import CompositeImageStore, ImageLoadError, LocalImageStore
from vivlmoe.data.records import ImageRef


def test_local_store_errors_and_draft(tmp_path: Path) -> None:
    write_images(tmp_path, ["a.jpg", "sub/b.png"], size=64)
    (tmp_path / "junk.jpg").write_bytes(b"nope")
    st = LocalImageStore(tmp_path)
    assert st.exists(ImageRef(0, "sub/b.png")) and not st.exists(ImageRef(0, "zzz.jpg"))
    img = st.open(ImageRef(0, "a.jpg"), draft_size=16)
    assert img.mode == "RGB" and max(img.size) <= 72
    with pytest.raises(ImageLoadError, match="not found"):
        st.open(ImageRef(0, "zzz.jpg"))
    with pytest.raises(ImageLoadError, match="cannot decode"):
        st.open(ImageRef(0, "junk.jpg"))
    with pytest.raises(ImageLoadError):
        st.open(ImageRef(1, "a.jpg"))
    assert st.describe().startswith("local:")


def test_composite_dispatch(tmp_path: Path) -> None:
    write_images(tmp_path / "x", ["1.jpg"], size=16)
    write_images(tmp_path / "y", ["2.jpg"], size=24)
    comp = CompositeImageStore([LocalImageStore(tmp_path / "x"), LocalImageStore(tmp_path / "y")])
    assert comp.exists(ImageRef(1, "2.jpg")) and not comp.exists(ImageRef(0, "2.jpg"))
    assert comp.open(ImageRef(1, "2.jpg")).size == (32, 24)
    with pytest.raises(ImageLoadError):
        comp.open(ImageRef(2, "2.jpg"))
    with pytest.raises(ValueError):
        CompositeImageStore([])
