"""GenerationEvaluator / ClassificationEvaluator with stub models: ordering, dumps, gloo gather."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from tests.fakes import tiny_t5_tokenizer
from vivlmoe.config.schema import DistCfg, GenerationCfg, TaskCfg
from vivlmoe.data.collators import Seq2SeqCollator
from vivlmoe.data.samplers import OrderedShardSampler
from vivlmoe.metrics import aspect_macro_f1
from vivlmoe.models.outputs import ClassifierOutput
from vivlmoe.tasks.evaluators.classification import ClassificationEvaluator
from vivlmoe.tasks.evaluators.dump import read_predictions_jsonl
from vivlmoe.tasks.evaluators.generation import GenerationEvaluator, generation_kwargs
from vivlmoe.training.distributed import DistContext, destroy_distributed, init_distributed
from vivlmoe.training.state import EvalContext

REFS = ["một con chó đang chạy", "cô gái đẹp", "chiếc xe đạp màu đỏ", "bãi cỏ xanh", "ngôi nhà"]
WORD_OFFSET = 10  # vocab ids of TINY_WORDS start after the 3 T5 specials; keep clear of them


class EchoDataset(Dataset):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, Any]:
        return {
            "uid": i,
            "image_id": f"img{i}",
            "pixel_values": torch.zeros(3, 4, 4),
            "refs": [REFS[i % len(REFS)], REFS[(i + 1) % len(REFS)]],
        }


class EchoModel(nn.Module):
    """``generate`` returns ``[WORD_OFFSET + uid, eos]`` so the decoded text identifies the uid."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []
        self.dummy = nn.Linear(1, 1)

    def forward(self, batch: dict, mode: str) -> Any:
        raise AssertionError("the generation evaluator must call generate(), not forward()")

    def generate(self, batch: dict, mode: str | None = None, **kw: Any) -> torch.Tensor:
        self.calls.append({"mode": mode, **kw})
        uids = torch.tensor(batch["uids"], dtype=torch.long)
        return torch.stack([WORD_OFFSET + uids, torch.ones_like(uids)], dim=1)


class ListSampler(Sampler[int]):
    def __init__(self, order: list[int]) -> None:
        self.order = order

    def __len__(self) -> int:
        return len(self.order)

    def __iter__(self) -> Iterator[int]:
        return iter(self.order)


def _task_cfg(**kw: Any) -> TaskCfg:
    return TaskCfg(
        name="captioning",
        metrics=kw.pop("metrics", ["bleu4", "rougeL"]),
        generation=GenerationCfg(num_beams=3, max_new_tokens=7),
        **kw,
    )


def _loader(n: int, order: list[int] | Sampler[int], batch_size: int = 2) -> DataLoader:
    tok = tiny_t5_tokenizer()
    collate = Seq2SeqCollator(None, tok, with_text=False, train=False)
    sampler = order if isinstance(order, Sampler) else ListSampler(order)
    return DataLoader(EchoDataset(n), batch_size=batch_size, sampler=sampler, collate_fn=collate)


def test_generation_evaluator_orders_by_uid_and_dumps(tmp_path: Path) -> None:
    tok = tiny_t5_tokenizer()
    ctx = DistContext.single_process("cpu")
    ev = GenerationEvaluator(ctx, _task_cfg(), tok, mode="joint_caption")
    model = EchoModel().eval()
    ec = EvalContext(ctx=ctx, step=7, run_dir=tmp_path, split_name="dev")
    metrics = ev.run(model, _loader(5, [4, 2, 0, 1, 3]), ec)
    assert metrics is not None and metrics["num_samples"] == 5.0
    assert {"bleu4", "rougeL", "rougeL_paper_compat", "pred_len_mean"} <= set(metrics)
    assert model.calls[0]["mode"] == "joint_caption" and model.calls[0]["num_beams"] == 3
    assert model.calls[0] == {"mode": "joint_caption", **generation_kwargs(ev.task_cfg.generation)}
    rows = read_predictions_jsonl(tmp_path / "predictions" / "dev_step7.jsonl")
    assert [r["uid"] for r in rows] == [0, 1, 2, 3, 4]
    expected = [tok.decode([WORD_OFFSET + i], skip_special_tokens=True) for i in range(5)]
    assert [r["prediction"] for r in rows] == expected
    assert rows[3]["references"] == [REFS[3], REFS[4]] and rows[3]["image_id"] == "img3"
    text = (tmp_path / "predictions" / "dev_step7.jsonl").read_text(encoding="utf-8")
    assert "một con chó" in text and "\\u" not in text
    assert (tmp_path / "metrics" / "dev_step7.json").exists()
    # save_predictions=False writes nothing new
    ec2 = EvalContext(ctx=ctx, step=8, run_dir=tmp_path, split_name="dev", save_predictions=False)
    ev.run(model, _loader(5, [0, 1, 2, 3, 4]), ec2)
    assert not (tmp_path / "predictions" / "dev_step8.jsonl").exists()


def test_generation_evaluator_rejects_bad_metric_names() -> None:
    with pytest.raises(ValueError, match="unknown metrics"):
        GenerationEvaluator(
            DistContext.single_process("cpu"), _task_cfg(metrics=["rsum"]), tiny_t5_tokenizer()
        )
    with pytest.raises(ValueError, match="at least one"):
        GenerationEvaluator(
            DistContext.single_process("cpu"), _task_cfg(metrics=[]), tiny_t5_tokenizer()
        )


# ------------------------------------------------------------------ classification
ASPECTS = ["Location", "Food", "Room"]


class ClsDataset(Dataset):
    def __init__(self, labels: torch.Tensor) -> None:
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, i: int) -> dict[str, Any]:
        return {"uid": i, "labels": self.labels[i]}


def _cls_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "uids": [it["uid"] for it in items],
        "labels": torch.stack([it["labels"] for it in items]),
        "attention_mask": torch.ones(len(items), 1),
    }


class LogitsModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("table", logits)

    def forward(self, batch: dict, mode: str) -> ClassifierOutput:
        assert mode == "joint_cls"
        return ClassifierOutput(logits=self.table[batch["uids"]], loss=None)


def test_classification_evaluator_matches_metric_and_dumps(tmp_path: Path) -> None:
    g = torch.Generator().manual_seed(0)
    n, a, c = 9, len(ASPECTS), 4
    logits = torch.randn(n, a, c, generator=g)
    labels = torch.randint(0, c, (n, a), generator=g)
    ctx = DistContext.single_process("cpu")
    ev = ClassificationEvaluator(ctx, ASPECTS, exclude_none=False)
    loader = DataLoader(
        ClsDataset(labels),
        batch_size=4,
        sampler=ListSampler([8, 3, 0, 1, 2, 4, 5, 6, 7]),
        collate_fn=_cls_collate,
    )
    ec = EvalContext(ctx=ctx, step=3, run_dir=tmp_path, split_name="dev")
    metrics = ev.run(LogitsModel(logits).eval(), loader, ec)
    assert metrics is not None
    ref = aspect_macro_f1(labels.numpy(), logits.argmax(-1).numpy(), ASPECTS)
    assert metrics["macro_f1"] == pytest.approx(ref["macro_f1"])
    assert metrics["macro_p"] == pytest.approx(ref["macro_p"])
    assert metrics["f1/Food"] == pytest.approx(ref["per_aspect"]["Food"]["f1"])
    assert metrics["accuracy"] == pytest.approx(
        float(np.mean(logits.argmax(-1).numpy() == labels.numpy()))
    )
    rows = read_predictions_jsonl(tmp_path / "predictions" / "dev_step3.jsonl")
    assert [r["uid"] for r in rows] == list(range(n))
    assert (
        rows[5]["gold_ids"] == labels[5].tolist()
        and rows[5]["pred_ids"] == logits[5].argmax(-1).tolist()
    )
    assert set(rows[0]["pred"]) == set(ASPECTS)
    # exclude_none changes the macro average
    ev2 = ClassificationEvaluator(ctx, ASPECTS, exclude_none=True)
    m2 = ev2.run(
        LogitsModel(logits).eval(),
        loader,
        EvalContext(ctx=ctx, step=4, run_dir=tmp_path, split_name="dev", save_predictions=False),
    )
    assert m2 is not None and m2["macro_f1"] == pytest.approx(
        aspect_macro_f1(labels.numpy(), logits.argmax(-1).numpy(), ASPECTS, exclude_none=True)[
            "macro_f1"
        ]
    )


# ------------------------------------------------------------------ gloo (2 processes)
WORLD = 2


def _gen_worker(rank: int, tmpdir: str) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(WORLD)
    ctx = init_distributed(
        DistCfg(backend="gloo", timeout_minutes=1), init_method=f"file://{tmpdir}/gen"
    )
    try:
        tok = tiny_t5_tokenizer()
        ev = GenerationEvaluator(
            ctx, _task_cfg(metrics=["rougeL", "f1"]), tok, mode="joint_caption"
        )
        # N = 1 -> rank 1 receives an empty shard and must still take part in the gather
        loader = _loader(1, OrderedShardSampler(1, WORLD, rank))
        assert len(loader) == (1 if rank == 0 else 0)
        ec = EvalContext(ctx=ctx, step=1, run_dir=Path(tmpdir) / f"rank{rank}", split_name="dev")
        metrics = ev.run(EchoModel().eval(), loader, ec)
        if rank == 0:
            assert metrics is not None and metrics["num_samples"] == 1.0
            rows = read_predictions_jsonl(
                Path(tmpdir) / "rank0" / "predictions" / "dev_step1.jsonl"
            )
            assert [r["uid"] for r in rows] == [0]
        else:
            assert metrics is None
            assert not (Path(tmpdir) / "rank1" / "predictions").exists()
        # N = 5 split 3 / 2: predictions come back in uid order on rank 0
        loader5 = _loader(5, OrderedShardSampler(5, WORLD, rank))
        m5 = ev.run(
            EchoModel().eval(),
            loader5,
            EvalContext(ctx=ctx, step=2, run_dir=Path(tmpdir) / f"rank{rank}", split_name="dev"),
        )
        if rank == 0:
            assert m5 is not None and m5["num_samples"] == 5.0
            rows = read_predictions_jsonl(
                Path(tmpdir) / "rank0" / "predictions" / "dev_step2.jsonl"
            )
            assert [r["prediction"] for r in rows] == [
                tok.decode([WORD_OFFSET + i], skip_special_tokens=True) for i in range(5)
            ]
    finally:
        destroy_distributed()


@pytest.mark.gloo
@pytest.mark.timeout(120)
def test_generation_evaluator_gather_with_empty_shard(tmp_path: Path) -> None:
    mp.spawn(_gen_worker, args=(str(tmp_path),), nprocs=WORLD, join=True)
