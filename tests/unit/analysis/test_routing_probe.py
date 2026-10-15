"""RoutingProbeEvaluator, CompositeEvaluator and maybe_add_routing_probe on tiny MoE models."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tests.unit.models.conftest import TINY_YAML
from vivlmoe.config.loader import config_from_dict, load_yaml_with_includes
from vivlmoe.config.schema import AnalysisCfg, ConfigError, MoECfg, RoutingProbeCfg
from vivlmoe.models.factory import build_task_model, load_model_cfg
from vivlmoe.tasks.evaluators.base import Evaluator
from vivlmoe.tasks.evaluators.composite import CompositeEvaluator
from vivlmoe.tasks.evaluators.routing import (
    RoutingProbeEvaluator,
    maybe_add_routing_probe,
    warn_probe_cadence,
)
from vivlmoe.training.distributed import DistContext
from vivlmoe.training.state import EvalContext

CTX = DistContext.single_process("cpu")
MOE = MoECfg(num_shared=1, num_routed=4, top_k=2)
VOCAB, LEN, N = 1000, 7, 6


class PairDataset(Dataset):
    def __init__(self, n: int = N, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.px = torch.randn(n, 3, 32, 32, generator=g)
        self.ids = torch.randint(3, VOCAB, (n, LEN), generator=g)
        self.lengths = torch.randint(3, LEN + 1, (n,), generator=g)

    def __len__(self) -> int:
        return self.px.shape[0]

    def __getitem__(self, i: int) -> dict[str, Any]:
        mask = torch.zeros(LEN, dtype=torch.long)
        mask[: int(self.lengths[i])] = 1
        ids = self.ids[i].clone()
        ids[mask == 0] = 1
        return {"pixel_values": self.px[i], "input_ids": ids, "attention_mask": mask, "uid": i}


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        k: torch.stack([it[k] for it in items])
        for k in ("pixel_values", "input_ids", "attention_mask")
    }
    out["uids"] = [int(it["uid"]) for it in items]
    return out


def _loader(batch_size: int = 4) -> DataLoader:
    return DataLoader(PairDataset(), batch_size=batch_size, collate_fn=collate, shuffle=False)


def _side_loaders() -> Any:
    """A RetrievalLoaders-like object: separate image / text loaders with their own uid spaces."""
    ds = PairDataset()
    images = DataLoader(
        [{"pixel_values": ds.px[i], "uid": i} for i in range(N)],
        batch_size=4,
        collate_fn=lambda items: {
            "pixel_values": torch.stack([it["pixel_values"] for it in items]),
            "uids": [it["uid"] for it in items],
        },
    )
    texts = DataLoader(
        [ds[i] for i in range(N)],
        batch_size=4,
        collate_fn=lambda items: {
            "input_ids": torch.stack([it["input_ids"] for it in items]),
            "attention_mask": torch.stack([it["attention_mask"] for it in items]),
            "uids": [it["uid"] for it in items],
        },
    )
    return SimpleNamespace(images=images, texts=texts)


@pytest.fixture(scope="module")
def moe_backbone() -> nn.Module:
    torch.manual_seed(0)
    return build_task_model(
        "backbone", dataclasses.replace(load_model_cfg(TINY_YAML), moe=MOE)
    ).eval()


def _ec(run_dir: Path, step: int) -> EvalContext:
    return EvalContext(ctx=CTX, step=step, run_dir=run_dir, split_name="dev")


def _probe(
    every: int = 2,
    max_steps: int | None = None,
    max_batches: int | None = None,
    mode: str = "contrastive",
) -> RoutingProbeEvaluator:
    return RoutingProbeEvaluator(
        "routing_probe", RoutingProbeCfg("dev", every, max_steps, max_batches), mode, num_layers=2
    )


def test_probe_contrastive_reports_files_and_change_rate(
    moe_backbone: nn.Module, tmp_path: Path
) -> None:
    probe = _probe(every=2, max_steps=4)
    model = moe_backbone.train()  # the probe must run in eval and leave the counters untouched
    first = probe.run(model, _loader(), _ec(tmp_path, 0))
    assert model.training and int(model.encoder.dispatch_stats().sum()) == 0
    assert first is not None and "routing/change_rate" not in first  # no previous snapshot yet
    for key in (
        "routing/sym_kl",
        "routing/sym_kl_layer_mean",
        "routing/margin_full",
        "routing/margin_topk",
        "routing/margin_full_visual",
        "routing/margin_full_text",
        "routing/expert_frac_max",
        "routing/expert_frac_min",
        "routing/bias_range",
        "routing/layer0/sym_kl",
        "routing/layer1/margin_full",
    ):
        assert key in first, key
    assert all(isinstance(v, float) for v in first.values())
    routing_dir = tmp_path / "routing"
    report = json.loads((routing_dir / "step_0000000.json").read_text())
    n_tokens = N * moe_backbone.num_visual_tokens + int(PairDataset().lengths.sum())
    assert report["num_tokens"] == n_tokens and report["num_layers"] == 2 and report["top_k"] == 2
    assert report["change_rate"] is None and report["forward_mode"] == "contrastive"
    heat = torch.tensor(report["modality"]["heatmap"])
    assert heat.shape == (4, 2) and torch.allclose(heat.sum(dim=0), torch.ones(2, dtype=heat.dtype))
    assert (routing_dir / "snapshot_step_0000000.pt").exists()
    assert (routing_dir / "step_0000000_heatmap.png").exists()  # matplotlib is installed in the env
    assert probe.run(model, _loader(), _ec(tmp_path, 1)) == {}  # not due
    # perturb the routers: the second due step reports a change rate against the stored snapshot
    with torch.no_grad():
        for _, ffn in model.encoder.moe_layers():
            ffn.router.proj.weight.mul_(-1.0)
    second = probe.run(model, _loader(), _ec(tmp_path, 2))
    assert second is not None and 0.0 < second["routing/change_rate"] <= 1.0
    assert {
        "routing/change_rate_visual",
        "routing/change_rate_text",
        "routing/layer0/change_rate",
    } <= set(second)
    report2 = json.loads((routing_dir / "step_0000002.json").read_text())
    assert report2["previous_step"] == 0 and report2["change_rate"]["steps"] == [0, 2]
    assert probe.run(model, _loader(), _ec(tmp_path, 6)) == {}  # beyond max_steps
    assert probe.history == [0, 2]
    with torch.no_grad():
        for _, ffn in model.encoder.moe_layers():
            ffn.router.proj.weight.mul_(-1.0)
    model.eval()


def test_probe_state_dict_survives_resume_and_batch_bound(
    moe_backbone: nn.Module, tmp_path: Path
) -> None:
    probe = _probe(every=1, max_batches=1)
    probe.run(moe_backbone, _loader(batch_size=2), _ec(tmp_path / "a", 0))
    report = json.loads((tmp_path / "a" / "routing" / "step_0000000.json").read_text())
    assert report["num_tokens"] == 2 * moe_backbone.num_visual_tokens + int(
        PairDataset().lengths[:2].sum()
    )
    state = probe.state_dict()
    assert state["history"] == [0] and state["previous"]["step"] == 0
    resumed = _probe(every=1, max_batches=1)
    resumed.load_state_dict(state)
    res = resumed.run(moe_backbone, _loader(batch_size=2), _ec(tmp_path / "b", 3))
    assert res["routing/change_rate"] == 0.0  # same model, same tokens
    assert resumed.history == [0, 3]
    fresh = _probe(every=1)
    fresh.load_state_dict({})
    assert fresh.previous is None and fresh.history == []


def test_probe_restarts_change_rate_on_a_different_token_stream(
    moe_backbone: nn.Module, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    probe = _probe(every=1, max_batches=1)
    probe.run(moe_backbone, _loader(batch_size=2), _ec(tmp_path, 0))
    state = probe.state_dict()
    assert state["previous"]["topk"].dtype == torch.int16  # compact snapshot in the checkpoint
    assert state["previous"]["modality"].dtype == torch.int8
    resumed = _probe(every=1, max_batches=2)  # resumed with a different bound -> other stream
    resumed.load_state_dict(state)
    with caplog.at_level(logging.WARNING, logger="vivlmoe.tasks.evaluators.routing"):
        res = resumed.run(moe_backbone, _loader(batch_size=2), _ec(tmp_path / "b", 1))
    assert "routing/change_rate" not in res and "restarts" in caplog.text
    assert resumed.previous is not None and resumed.previous.step == 1
    again = resumed.run(moe_backbone, _loader(batch_size=2), _ec(tmp_path / "b", 2))
    assert again["routing/change_rate"] == 0.0  # the series restarted from step 1


def test_probe_budget_is_world_size_independent(moe_backbone: nn.Module, tmp_path: Path) -> None:
    """``max_batches`` bounds the global uid range, not the first batches of each rank's shard."""
    budget = 1 * 4  # max_batches=1 x batch_size=4 -> uids 0..3 whatever the shard
    full = _probe(every=1, max_batches=1).run(moe_backbone, _loader(batch_size=4), _ec(tmp_path, 0))
    report = json.loads((tmp_path / "routing" / "step_0000000.json").read_text())
    assert report["num_tokens"] == budget * moe_backbone.num_visual_tokens + int(
        PairDataset().lengths[:budget].sum()
    )
    assert full is not None and "routing/sym_kl" in full
    # a "rank-1 shard" (uids 1, 3, 5) of a 2-process run keeps only uids 1 and 3 of its first batch
    ds = PairDataset()
    kept, exhausted = RoutingProbeEvaluator._select(collate([ds[1], ds[3], ds[5]]), budget)
    assert kept is not None and kept["uids"] == [1, 3] and exhausted
    assert kept["pixel_values"].shape[0] == 2 and torch.equal(
        kept["input_ids"][1], ds[3]["input_ids"]
    )
    assert RoutingProbeEvaluator._select(collate([ds[5]]), budget) == (None, True)
    whole = collate([ds[0], ds[2]])
    assert RoutingProbeEvaluator._select(whole, budget) == (whole, False)
    assert RoutingProbeEvaluator._select(whole, None) == (whole, False)
    with pytest.raises(ConfigError, match="batch_size"):
        _probe(every=1, max_batches=1).run(
            moe_backbone, DataLoader([collate([ds[0]])], batch_size=None), _ec(tmp_path, 1)
        )


def test_probe_with_separate_image_and_text_loaders(
    moe_backbone: nn.Module, tmp_path: Path
) -> None:
    probe = _probe(every=1)
    res = probe.run(moe_backbone, _side_loaders(), _ec(tmp_path, 0))
    assert res is not None and "routing/sym_kl" in res
    report = json.loads((tmp_path / "routing" / "step_0000000.json").read_text())
    assert report["num_tokens"] == N * moe_backbone.num_visual_tokens + int(
        PairDataset().lengths.sum()
    )
    same = probe.run(moe_backbone, _side_loaders(), _ec(tmp_path, 1))
    assert same["routing/change_rate"] == 0.0


def test_probe_rejects_dense_models_and_bad_modes(tmp_path: Path) -> None:
    torch.manual_seed(0)
    dense = build_task_model("backbone", load_model_cfg(TINY_YAML)).eval()
    with pytest.raises(ConfigError, match="MoE"):
        _probe(every=1).run(dense, _loader(), _ec(tmp_path, 0))
    with pytest.raises(ValueError, match="forward_mode"):
        _probe(mode="dual")


@pytest.mark.parametrize(
    "kind,mode",
    [("seq2seq", "joint_caption"), ("seq2seq", "joint_vqa"), ("classifier", "joint_cls")],
)
def test_probe_joint_modes_on_downstream_heads(kind: str, mode: str, tmp_path: Path) -> None:
    torch.manual_seed(0)
    cfg = dataclasses.replace(load_model_cfg(TINY_YAML), moe=MOE)
    model = build_task_model(kind, cfg).eval()
    ds = PairDataset(n=3)
    g = torch.Generator().manual_seed(1)
    if mode == "joint_cls":
        a, m = cfg.classifier.num_aspects, 2
        image_mask = torch.tensor([[True, True], [True, False], [False, True]])
        attn = torch.ones(3, a, LEN, dtype=torch.long)
        attn[:, :, 5:] = 0
        batch = {
            "pixel_values": torch.randn(3, m, 3, 32, 32, generator=g),
            "image_mask": image_mask,
            "input_ids": torch.randint(3, VOCAB, (3, a, LEN), generator=g),
            "attention_mask": attn,
            "uids": [0, 1, 2],
        }
        expected = int(image_mask.sum()) * cfg.resampler.num_latents * a + int(attn.sum())
    else:
        batch = collate([ds[i] for i in range(3)])
        if mode == "joint_caption":
            batch.pop("input_ids")
            batch.pop("attention_mask")
            expected = 3 * cfg.resampler.num_latents
        else:
            expected = 3 * cfg.resampler.num_latents + int(batch["attention_mask"].sum())
    loader = DataLoader([batch], batch_size=None)
    res = _probe(every=1, mode=mode).run(model, loader, _ec(tmp_path, 0))
    assert res is not None and "routing/margin_full" in res
    report = json.loads((tmp_path / "routing" / "step_0000000.json").read_text())
    assert report["num_tokens"] == expected
    if mode == "joint_caption":
        assert report["modality"]["sym_kl"] is None  # no text tokens: L_mi = 0 by design
    else:
        assert report["modality"]["sym_kl"] is not None


class _Const(Evaluator):
    def __init__(self, name: str, value: dict[str, float]) -> None:
        self.name, self.value, self.calls = name, value, 0

    def run(self, model: nn.Module, loader: Any, ec: EvalContext) -> dict[str, float] | None:
        self.calls += 1
        return dict(self.value) if ec.ctx.is_main else None

    def state_dict(self) -> dict[str, Any]:
        return {"calls": self.calls}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.calls = int(state["calls"])


def test_composite_evaluator_merges_and_checks() -> None:
    a, b = _Const("a", {"x": 1.0}), _Const("b", {"y": 2.0})
    comp = CompositeEvaluator([a, b])
    ec = EvalContext(ctx=CTX, step=0, run_dir=Path("."), split_name="dev")
    assert comp.run(nn.Identity(), None, ec) == {"x": 1.0, "y": 2.0}
    state = comp.state_dict()
    assert state == {"a": {"calls": 1}, "b": {"calls": 1}}
    other = CompositeEvaluator([_Const("a", {}), _Const("b", {})])
    other.load_state_dict(state)
    assert other.evaluators[0].calls == 1
    with pytest.raises(KeyError, match="re-defines"):
        CompositeEvaluator([a, _Const("c", {"x": 3.0})]).run(nn.Identity(), None, ec)
    with pytest.raises(ValueError, match="unique"):
        CompositeEvaluator([a, _Const("a", {})])
    with pytest.raises(ValueError, match="at least one"):
        CompositeEvaluator([])


def _run_cfg(moe: bool, split: str = "dev"):
    raw = {
        "task": {"name": "contrastive"},
        "model": {
            **load_yaml_with_includes(TINY_YAML)["model"],
            "moe": dataclasses.asdict(MOE) if moe else None,
        },
        "data": {"evals": {"dev": {"kind": "synthetic_pairs", "split": "dev"}}},
        "train": {"max_steps": 1},
        "analysis": {"routing_probe": {"split": split, "every_steps": 2, "max_steps": 10}},
    }
    return config_from_dict(raw)


def test_maybe_add_routing_probe_wraps_the_split_evaluator() -> None:
    base = _Const("retrieval", {"rsum": 1.0})
    evaluators: dict[str, Evaluator] = {"dev": base}
    maybe_add_routing_probe(evaluators, _run_cfg(moe=True), CTX, "contrastive")
    comp = evaluators["dev"]
    assert isinstance(comp, CompositeEvaluator) and comp.evaluators[0] is base
    probe = comp.evaluators[1]
    assert (
        isinstance(probe, RoutingProbeEvaluator)
        and probe.cfg.every_steps == 2
        and probe.num_layers == 2
    )
    assert probe.forward_mode == "contrastive"
    with pytest.raises(ValueError, match="unique"):  # a second probe on the same split is refused
        maybe_add_routing_probe(evaluators, _run_cfg(moe=True), CTX, "contrastive")
    dense_evals: dict[str, Evaluator] = {"dev": base}
    maybe_add_routing_probe(dense_evals, _run_cfg(moe=False), CTX, "contrastive")
    assert dense_evals["dev"] is base  # dense model: no-op
    with pytest.raises(ConfigError, match="has no evaluator"):
        maybe_add_routing_probe({"other": base}, _run_cfg(moe=True), CTX, "contrastive")


def test_probe_cadence_warning(caplog: pytest.LogCaptureFixture) -> None:
    cfg = _run_cfg(moe=True)  # eval.every_steps None (epoch cadence), probe every 2 steps
    with caplog.at_level(logging.WARNING, logger="vivlmoe.tasks.evaluators.routing"):
        assert warn_probe_cadence(cfg) and "own step cadence" in caplog.text
    aligned = dataclasses.replace(cfg, eval=dataclasses.replace(cfg.eval, every_steps=2))
    assert not warn_probe_cadence(aligned)
    off = dataclasses.replace(cfg, eval=dataclasses.replace(cfg.eval, every_steps=3))
    assert warn_probe_cadence(off)
    dense = _run_cfg(moe=False)
    assert warn_probe_cadence(dataclasses.replace(dense, analysis=AnalysisCfg())) is False
