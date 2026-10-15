"""Generic YAML-configured VQA adapter (OpenViVQA, ViTextVQA, ViOCR-VQA, ViPOPE ...).

Layouts covered by ``options`` (plan 8.2):

* ``format: json`` with ``records_path`` pointing (dotted) at a **list** of records or a **dict**
  of ``id -> record`` (OpenViVQA ``annotations``); ``format: jsonl`` one record per line.
* the image is ``record[image_key]``, mapped through ``image_lookup`` (``table_path`` to a dict
  ``id -> filename`` / ``id -> {filename_key: ..}`` or a list of ``{id_key, filename_key}``) and/or
  ``image_filename_template`` (``"{image_id}.jpg"``), else used verbatim as the file name.
* answers from ``answers_key`` (list) and/or ``answer_key`` (single; becomes ``answers[0]``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vivlmoe.config.schema import ConfigError
from vivlmoe.data.adapters.base import BaseAdapter
from vivlmoe.data.records import ImageRef, ListRecordSet, RecordSet, VQARecord
from vivlmoe.data.registry import register_dataset

log = logging.getLogger(__name__)


def navigate(data: Any, path: str | None, where: str) -> Any:
    """Follow a dotted path (``"a.b.0"``: dict keys or list indices)."""
    node = data
    if not path:
        return node
    for seg in path.split("."):
        if isinstance(node, dict):
            if seg not in node:
                raise KeyError(f"{where}: key {seg!r} of records_path {path!r} not found")
            node = node[seg]
        elif isinstance(node, list) and seg.isdigit():
            node = node[int(seg)]
        else:
            raise KeyError(f"{where}: cannot descend into {type(node).__name__} with {seg!r}")
    return node


def iter_container(container: Any, where: str) -> Iterator[tuple[str, dict[str, Any]]]:
    if isinstance(container, list):
        items: Iterator[tuple[str, Any]] = ((str(i), e) for i, e in enumerate(container))
    elif isinstance(container, dict):
        items = ((str(k), v) for k, v in container.items())
    else:
        raise ValueError(
            f"{where}: records container must be a list or dict, got {type(container).__name__}"
        )
    for key, entry in items:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{where}[{key}]: record must be a mapping, got {type(entry).__name__}"
            )
        yield key, entry


@register_dataset("vqa_generic")
class GenericVQAAdapter(BaseAdapter):
    @dataclass
    class Options:
        format: str = "json"  # json | jsonl
        records_path: str | None = None  # dotted path to the record container (json only)
        id_key: str | None = None  # None: dict key / row index
        image_key: str = "image_id"
        question_key: str = "question"
        answer_key: str | None = None
        answers_key: str | None = None
        image_lookup: dict[str, Any] | None = None  # {table_path, id_key, filename_key}
        image_filename_template: str | None = None  # "{image_id}.jpg"
        prompt_template: str = "{question}"  # used by builders.build_vqa

        def __post_init__(self) -> None:
            if self.format not in ("json", "jsonl"):
                raise ConfigError(f"vqa_generic: format must be json|jsonl, got {self.format!r}")
            if self.answer_key is None and self.answers_key is None:
                raise ConfigError("vqa_generic: set answer_key and/or answers_key")
            if self.format == "jsonl" and self.records_path:
                raise ConfigError("vqa_generic: records_path is only valid with format=json")
            if "{question}" not in self.prompt_template:
                raise ConfigError("vqa_generic: prompt_template must contain '{question}'")

    def _load(self, path: Path) -> tuple[Any, Any]:
        """Return ``(records container, whole document)``."""
        if self.options.format == "jsonl":
            rows = []
            with path.open("r", encoding="utf-8") as fh:
                for ln, line in enumerate(fh):
                    if line.strip():
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"{path}:{ln + 1}: invalid JSON ({exc})") from None
            return rows, rows
        with path.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        return navigate(doc, self.options.records_path, str(path)), doc

    def _lookup(self, doc: Any, where: str) -> dict[str, str] | None:
        lk = self.options.image_lookup
        if lk is None:
            return None
        unknown = set(lk) - {"table_path", "id_key", "filename_key"}
        if unknown:
            raise ValueError(f"vqa_generic: image_lookup has unknown keys {sorted(unknown)}")
        table = navigate(doc, lk.get("table_path"), where)
        fk, ik = lk.get("filename_key"), lk.get("id_key")
        out: dict[str, str] = {}
        if isinstance(table, dict):
            for k, v in table.items():
                out[str(k)] = str(v[fk]) if isinstance(v, dict) else str(v)
        elif isinstance(table, list):
            if not ik or not fk:
                raise ValueError(
                    "vqa_generic: a list image_lookup table needs id_key and filename_key"
                )
            for e in table:
                out[str(e[ik])] = str(e[fk])
        else:
            raise ValueError(f"{where}: image_lookup table must be a dict or list")
        return out

    def _answers(self, entry: dict[str, Any], where: str) -> tuple[str, ...]:
        o = self.options
        answers: list[str] = []
        if o.answer_key is not None:
            if o.answer_key not in entry:
                raise KeyError(f"{where}: missing {o.answer_key!r}")
            answers.append(str(entry[o.answer_key]))
        if o.answers_key is not None:
            if o.answers_key not in entry:
                raise KeyError(f"{where}: missing {o.answers_key!r}")
            more = entry[o.answers_key]
            if isinstance(more, str):
                more = [more]
            answers.extend(str(a) for a in more if str(a) not in answers)
        answers = [a.strip() for a in answers if a and a.strip()]
        if not answers:
            raise ValueError(f"{where}: no non-empty answer")
        return tuple(answers)

    def _raw_records(self, split: str) -> RecordSet:
        o = self.options
        path = self.split_file(split)
        container, doc = self._load(path)
        lookup = self._lookup(doc, str(path))
        records: list[VQARecord] = []
        for key, entry in iter_container(container, str(path)):
            where = f"{path}[{key}]"
            for k in (o.image_key, o.question_key):
                if k not in entry:
                    raise KeyError(f"{where}: missing {k!r}")
            raw_image = str(entry[o.image_key])
            filename = raw_image
            if lookup is not None:
                if raw_image not in lookup:
                    raise KeyError(f"{where}: image id {raw_image!r} not in image_lookup table")
                filename = lookup[raw_image]
            if o.image_filename_template:
                filename = o.image_filename_template.format(image_id=filename)
            sample_id = str(entry[o.id_key]) if o.id_key else key
            records.append(
                VQARecord(
                    uid=len(records),
                    image_id=raw_image,
                    image=ImageRef(0, filename),
                    question=str(entry[o.question_key]).strip(),
                    answers=self._answers(entry, where),
                    meta={"id": sample_id},
                )
            )
        log.info("vqa_generic/%s: %d records from %s", split, len(records), path)
        return ListRecordSet(records)


__all__ = ["GenericVQAAdapter", "iter_container", "navigate"]
