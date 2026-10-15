from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.data.conftest import write_images
from vivlmoe.config.schema import ConfigError, DataSourceCfg, SplitSpec
from vivlmoe.data.datasets import VQADataset
from vivlmoe.data.image_store import ImageLoadError
from vivlmoe.data.registry import build_adapter


def _cfg(root: Path, file: str, **options) -> DataSourceCfg:
    return DataSourceCfg(
        kind="vqa_generic",
        root=str(root),
        splits={"train": SplitSpec(file=file, image_root="imgs")},
        options=options,
    )


def test_openvivqa_dict_of_dicts_with_image_lookup(tmp_path: Path) -> None:
    write_images(tmp_path / "imgs", ["a.jpg", "b.jpg"], size=20)
    doc = {
        "images": {"7": "a.jpg", "8": "b.jpg"},
        "annotations": {
            "1": {"image_id": 7, "question": "Cái gì?", "answers": ["con chó", "chó"]},
            "2": {"image_id": 8, "question": "Ở đâu?", "answers": ["sân"]},
        },
    }
    (tmp_path / "open.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    ad = build_adapter(
        _cfg(
            tmp_path,
            "open.json",
            records_path="annotations",
            answers_key="answers",
            image_lookup={"table_path": "images"},
            prompt_template="Câu hỏi: {question}",
        )
    )
    recs = ad.load_split("train")
    assert len(recs) == 2 and recs[0].answers == ("con chó", "chó") and recs[0].image.key == "a.jpg"
    assert recs[0].image_id == "7" and recs[0].meta["id"] == "1" and recs[1].uid == 1
    assert ad.options.prompt_template == "Câu hỏi: {question}"


def test_record_list_with_single_answer_key(tmp_path: Path) -> None:
    write_images(tmp_path / "imgs", ["c.png"], size=20)
    rows = [{"image": "c.png", "question": "Có mấy con chó?", "answer": "hai", "id": 42}]
    (tmp_path / "qa.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    recs = build_adapter(
        _cfg(tmp_path, "qa.json", image_key="image", answer_key="answer", id_key="id")
    ).load_split("train")
    assert (
        recs[0].answers == ("hai",) and recs[0].meta["id"] == "42" and recs[0].image.key == "c.png"
    )


def test_vipope_jsonl_with_list_lookup_and_template(tmp_path: Path, tiny_transform) -> None:
    write_images(tmp_path / "imgs", ["COCO_000000000011.jpg"], size=20)
    doc = {"images": [{"id": 11, "file_name": "COCO_000000000011.jpg"}]}
    (tmp_path / "images.json").write_text(json.dumps(doc), encoding="utf-8")
    rows = [
        {
            "image_id": 11,
            "text": "Có con mèo trong ảnh không?",
            "label": "không",
            "extra": ["có", "không"],
        }
    ]
    (tmp_path / "pope.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    recs = build_adapter(
        _cfg(
            tmp_path,
            "pope.jsonl",
            format="jsonl",
            question_key="text",
            answer_key="label",
            answers_key="extra",
            image_filename_template="COCO_{image_id:0>12}.jpg",
        )
    ).load_split("train")
    assert recs[0].image.key == "COCO_000000000011.jpg"
    assert recs[0].answers == ("không", "có")  # answer_key first, duplicates removed
    ad = build_adapter(
        _cfg(
            tmp_path,
            "pope.jsonl",
            format="jsonl",
            question_key="text",
            answer_key="label",
            image_filename_template="{image_id}.jpg",
        )
    )
    with pytest.raises(ImageLoadError):  # missing image under policy=error
        ad.load_split("train")
    ds = VQADataset(
        recs,
        build_adapter(
            _cfg(
                tmp_path,
                "pope.jsonl",
                format="jsonl",
                question_key="text",
                answer_key="label",
                answers_key="extra",
                image_filename_template="COCO_{image_id:0>12}.jpg",
            )
        ).image_store("train"),
        tiny_transform,
        expand_answers=True,
    )
    assert (
        len(ds) == 2
        and ds[1]["answer"] == "có"
        and ds[1]["uid"] == 1
        and ds[1]["refs"] == ["không", "có"]
    )
    assert ds.groups() == [[0, 1]]


def test_vqa_generic_option_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        build_adapter(_cfg(tmp_path, "x.json"))  # no answer key
    with pytest.raises(ConfigError):
        build_adapter(_cfg(tmp_path, "x.json", answer_key="a", format="csv"))
    (tmp_path / "x.json").write_text(
        json.dumps([{"question": "q", "answer": "a"}]), encoding="utf-8"
    )
    with pytest.raises(KeyError, match="image_id"):
        build_adapter(_cfg(tmp_path, "x.json", answer_key="answer")).load_split("train")
