from __future__ import annotations

from dataclasses import dataclass

import pytest

from vivlmoe.config.schema import ConfigError, DataSourceCfg
from vivlmoe.data.adapters.base import BaseAdapter
from vivlmoe.data.registry import DATASETS, DatasetAdapter, build_adapter, register_dataset


def test_builtin_kinds_registered() -> None:
    import vivlmoe.data.adapters  # noqa: F401

    for kind in [
        "phomt",
        "json_dict",
        "jsonl",
        "hf_image_caption",
        "mixture",
        "ktvic",
        "uitviic",
        "vimacsa",
        "vqa_generic",
        "synthetic_pairs",
        "synthetic_parallel",
        "synthetic_caption",
        "synthetic_vqa",
        "synthetic_vimacsa",
    ]:
        assert kind in DATASETS
    assert DATASETS.get("ktvic").kind == "ktvic"
    assert isinstance(build_adapter(DataSourceCfg(kind="synthetic_pairs")), DatasetAdapter)


def test_register_requires_options_dataclass() -> None:
    with pytest.raises(TypeError):

        @register_dataset("_no_options")
        class Bad:  # noqa: D401
            pass


def test_build_adapter_validates_options_strictly() -> None:
    cfg = DataSourceCfg(
        kind="ktvic", root="/nowhere", options={"caption_key": "caption", "bogus": 1}
    )
    with pytest.raises(ConfigError) as exc:
        build_adapter(cfg)
    assert "bogus" in str(exc.value) and "options" in str(exc.value)
    with pytest.raises(ConfigError):
        build_adapter(DataSourceCfg(kind="ktvic", options={"caption_key": "nope"}))
    with pytest.raises(KeyError):
        build_adapter(DataSourceCfg(kind="does_not_exist"))


def test_custom_adapter_pipeline_uid_renumber_and_limit(tmp_path) -> None:
    from vivlmoe.data.records import ImageRef, ImageTextRecord, ListRecordSet

    @register_dataset("_test_custom")
    class Custom(BaseAdapter):
        @dataclass
        class Options:
            n: int = 5

        def _image_check(self):
            return "none"

        def _raw_records(self, split):
            return ListRecordSet(
                [
                    ImageTextRecord(
                        uid=100 + i, image_id=f"i{i}", image=ImageRef(0, f"{i}.jpg"), text=f"t{i}"
                    )
                    for i in range(self.options.n)
                ]
            )

    ad = build_adapter(DataSourceCfg(kind="_test_custom", limit=3, options={"n": 6}))
    recs = ad.load_split("train")
    assert [r.uid for r in recs] == [0, 1, 2]
    with pytest.raises(KeyError):
        ad.split_spec("train")  # no files configured
