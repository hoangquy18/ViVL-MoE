from __future__ import annotations

from pathlib import Path

from tests.unit.data.conftest import TINY_SPEC
from vivlmoe.config.schema import DEFAULT_ASPECTS, DataSourceCfg
from vivlmoe.data.builders import (
    build_caption_eval,
    build_image_text_train,
    build_parallel_text,
    build_retrieval_eval,
    build_vimacsa,
    build_vqa,
)
from vivlmoe.data.collators import (
    ContrastiveCollator,
    MultiImageClsCollator,
    RetrievalImageCollator,
    RetrievalTextCollator,
    Seq2SeqCollator,
    TeacherCollator,
)
from vivlmoe.data.registry import build_adapter


def test_synthetic_kinds_end_to_end(tmp_path: Path, tiny_transform, bundle) -> None:
    root = str(tmp_path / "syn")
    pairs = DataSourceCfg(
        kind="synthetic_pairs", root=root, options={"num_samples": 8, "captions_per_image": 2}
    )
    ds, groups = build_image_text_train(pairs, tiny_transform)
    assert len(ds) == 8 and len(groups) == 4
    batch = ContrastiveCollator(bundle.vi, 16)([ds[0], ds[1]])
    assert batch["pixel_values"].shape == (2, 3, 32, 32)
    a1 = build_adapter(pairs).load_split("train")
    a2 = build_adapter(pairs).load_split("train")
    assert [r.text for r in a1] == [r.text for r in a2]  # deterministic
    dev = build_adapter(pairs).load_split("dev")
    assert [r.text for r in dev] != [r.text for r in a1]
    ret = build_retrieval_eval(pairs, tiny_transform)
    assert (
        len(ret.images) == 4
        and len(ret.texts) == 8
        and ret.img2caps[1] == [2, 3]
        and ret.cap2img[3] == 1
    )
    RetrievalImageCollator()([ret.images[0]])
    RetrievalTextCollator(bundle.vi, 8)([ret.texts[0]])
    cap = build_caption_eval(
        DataSourceCfg(
            kind="synthetic_caption", root=root, options={"num_samples": 6, "captions_per_image": 3}
        ),
        tiny_transform,
    )
    assert len(cap) == 2 and len(cap[0]["refs"]) == 3
    Seq2SeqCollator(None, bundle.decoder, with_text=False, train=False)([cap[0]])
    par = build_parallel_text(DataSourceCfg(kind="synthetic_parallel", options={"num_samples": 5}))
    tb = TeacherCollator(bundle.clip_en, bundle.vi)([par[0], par[4]])
    assert tb["en_input_ids"].shape == (2, 77)
    vqa, template = build_vqa(
        DataSourceCfg(kind="synthetic_vqa", root=root, options={"num_samples": 4}), tiny_transform
    )
    assert template == "{question}" and len(vqa) == 4 and len(vqa[0]["refs"]) == 2
    vb = Seq2SeqCollator(bundle.vi, bundle.decoder, with_text=True, train=True)([vqa[0], vqa[1]])
    assert vb["labels"].shape[0] == 2
    cls = build_vimacsa(
        DataSourceCfg(
            kind="synthetic_vimacsa", root=root, options={"num_samples": 5, "max_images": 2}
        ),
        tiny_transform,
        (TINY_SPEC.crop_size,) * 2,
        max_images=3,
    )
    cb = MultiImageClsCollator(bundle.vi, DEFAULT_ASPECTS, cls_max_len=32)(
        [cls[i] for i in range(5)]
    )
    assert cb["pixel_values"].shape == (5, 3, 3, 32, 32) and cb["labels"].shape == (5, 6)
    assert cb["input_ids"].shape[:2] == (5, 6)
