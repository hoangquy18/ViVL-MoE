from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from vivlmoe.config.schema import CheckpointCfg, ConfigError, ModelCfg, TextCfg, VisionCfg
from vivlmoe.training.checkpoint import (
    CheckpointError,
    CheckpointManager,
    ProvenanceTracker,
    build_checkpoint,
    collect_rng_state,
    load_checkpoint_file,
    load_for_resume,
    load_model_state,
    load_partial,
    model_fingerprint,
    restore_rng_state,
)
from vivlmoe.training.distributed import DistContext
from vivlmoe.training.state import TrainState
from vivlmoe.utils.ckpt_meta import meta_path, read_meta
from vivlmoe.utils.io import atomic_torch_save


class Tower(nn.Module):
    def __init__(self, d: int = 4) -> None:
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
        self.register_buffer("position_ids", torch.arange(d))


class Toy(nn.Module):
    def __init__(self, d: int = 4) -> None:
        super().__init__()
        self.vision_encoder = Tower(d)
        self.text_encoder = Tower(d)
        self.encoder = nn.Linear(d, d)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))
        self.register_buffer("expert_bias", torch.zeros(3))  # persistent MoE-style buffer
        self.register_buffer("dispatch_counts", torch.zeros(3, dtype=torch.int64), persistent=False)


CTX = DistContext.single_process("cpu")


def _ckpt(model: nn.Module, excluded: list[str], prov: dict[str, str], **kw) -> dict:
    return build_checkpoint(
        model=model,
        excluded=excluded,
        provenance=prov,
        config={"model": {"x": 1}},
        train_state=TrainState(step=3, epoch=1).state_dict(),
        task_state={"k": 1},
        ctx=CTX,
        **kw,
    )


def _sd_equal(a: nn.Module, b: nn.Module, prefix: str = "") -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return all(torch.equal(sa[k], sb[k]) for k in sa if k.startswith(prefix))


def test_roundtrip_with_sidecar_and_buffers(tmp_path: Path) -> None:
    torch.manual_seed(0)
    m = Toy()
    m.expert_bias.fill_(0.5)
    m.dispatch_counts.fill_(9)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    ck = _ckpt(
        m,
        [],
        {"vision_encoder": "trained", "text_encoder": "trained"},
        optimizer=opt,
        metrics={"dev/x": 1.0},
    )
    assert "dispatch_counts" not in ck["model"] and "expert_bias" in ck["model"]
    path = atomic_torch_save(ck, tmp_path / "a.pt")
    from vivlmoe.utils.ckpt_meta import write_meta

    write_meta(path, ck)
    meta = read_meta(path)
    assert (
        meta["format_version"] == 1
        and meta["step"] == 3
        and meta["tower_provenance"]["text_encoder"] == "trained"
    )
    assert json.loads(meta_path(path).read_text())["metrics"] == {"dev/x": 1.0}
    loaded = load_checkpoint_file(path)
    fresh = Toy()
    report = load_model_state(
        fresh, loaded, {"vision_encoder": "trained", "text_encoder": "trained"}
    )
    assert _sd_equal(m, fresh) and report.rematerialised == []
    assert torch.equal(fresh.expert_bias, torch.full((3,), 0.5))
    assert (
        loaded["optimizer"] is not None
        and loaded["world_size"] == 1
        and loaded["rng"][0]["torch"] is not None
    )


def test_exclusion_and_rematerialisation(tmp_path: Path) -> None:
    m = Toy()
    for p in m.vision_encoder.parameters():
        p.requires_grad_(False)
    tracker = ProvenanceTracker({"vision_encoder": "hf:openai/clip", "text_encoder": "hf:xlmr"})
    tracker.mark_trainable_towers(m)
    assert tracker.as_dict() == {"vision_encoder": "hf:openai/clip", "text_encoder": "trained"}
    excluded = tracker.excluded_prefixes(m, True)
    assert excluded == ["vision_encoder"]
    assert tracker.excluded_prefixes(m, False) == []
    ck = _ckpt(m, excluded, tracker.as_dict())
    assert not any(k.startswith("vision_encoder") for k in ck["model"])
    assert any(k.startswith("text_encoder") for k in ck["model"])

    fresh = Toy()
    report = load_model_state(
        fresh, ck, {"vision_encoder": "hf:openai/clip", "text_encoder": "trained"}
    )
    assert report.rematerialised == ["vision_encoder"]
    assert _sd_equal(m, fresh, "text_encoder") and _sd_equal(m, fresh, "encoder")
    with pytest.raises(CheckpointError, match="cannot be re-materialised"):
        load_model_state(Toy(), ck, {"vision_encoder": "hf:other", "text_encoder": "trained"})
    tampered = dict(ck)
    tampered["tower_provenance"] = {"vision_encoder": "trained", "text_encoder": "trained"}
    with pytest.raises(CheckpointError, match="not re-materialisable"):
        load_model_state(Toy(), tampered, {"vision_encoder": "trained", "text_encoder": "trained"})


def test_strict_diff_messages() -> None:
    m = Toy()
    prov = {"vision_encoder": "trained", "text_encoder": "trained"}
    ck = _ckpt(m, [], prov)
    extra = dict(ck)
    extra["model"] = {**ck["model"], "ghost.weight": torch.zeros(1)}
    with pytest.raises(CheckpointError, match="unexpected in checkpoint: ghost.weight"):
        load_model_state(Toy(), extra, prov)
    missing = dict(ck)
    missing["model"] = {k: v for k, v in ck["model"].items() if k != "encoder.bias"}
    with pytest.raises(CheckpointError, match="missing in checkpoint: encoder.bias"):
        load_model_state(Toy(), missing, prov)
    bad = dict(ck)
    bad["model"] = {**ck["model"], "encoder.bias": torch.zeros(7)}
    with pytest.raises(CheckpointError, match="shape mismatch"):
        load_model_state(Toy(), bad, prov)


def test_partial_prefix_load(tmp_path: Path) -> None:
    torch.manual_seed(1)
    src = Toy()
    for p in src.vision_encoder.parameters():
        p.requires_grad_(False)
    prov = {"vision_encoder": "hf:clip", "text_encoder": "trained"}
    ck = _ckpt(src, ["vision_encoder"], prov)
    path = atomic_torch_save(ck, tmp_path / "src.pt")

    dst = Toy()
    before_vision = {k: v.clone() for k, v in dst.vision_encoder.state_dict().items()}
    report = load_partial(dst, path, ["text_encoder", "logit_scale"], {"vision_encoder": "hf:clip"})
    assert _sd_equal(src, dst, "text_encoder") and torch.equal(dst.logit_scale, src.logit_scale)
    assert not _sd_equal(src, dst, "encoder.")
    assert all(torch.equal(before_vision[k], v) for k, v in dst.vision_encoder.state_dict().items())
    assert report.provenance == {"text_encoder": "trained"}
    # a prefix inside the excluded tower is tolerated when provenance matches, else an error
    r2 = load_partial(Toy(), ck, ["vision_encoder.lin"], {"vision_encoder": "hf:clip"})
    assert r2.rematerialised == ["vision_encoder"] and r2.loaded_keys == []
    with pytest.raises(CheckpointError):
        load_partial(Toy(), ck, ["vision_encoder"], {"vision_encoder": "hf:different"})
    with pytest.raises(ConfigError, match="matches nothing"):
        load_partial(Toy(), ck, ["decoder"], {})
    with pytest.raises(ConfigError, match="'all' cannot be combined"):
        load_partial(Toy(), ck, ["all", "encoder"], {})
    full = load_partial(
        Toy(), ck, ["all"], {"vision_encoder": "hf:clip", "text_encoder": "trained"}
    )
    assert full.rematerialised == ["vision_encoder"]


def test_manager_keep_last_n_last_symlink_and_best(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path / "ckpt", CheckpointCfg(keep_last_n=2), CTX)
    m = Toy()
    prov = {"vision_encoder": "trained", "text_encoder": "trained"}
    for step in (1, 2, 3):
        ck = _ckpt(m, [], prov)
        ck["step"] = step
        mgr.save_step(ck, step)
    names = sorted(p.name for p in mgr.step_files())
    assert names == ["step_0000002.pt", "step_0000003.pt"]
    assert not meta_path(mgr.step_path(1)).exists()
    assert mgr.last_path.is_symlink() and mgr.last_path.resolve().name == "step_0000003.pt"
    assert read_meta(mgr.last_path)["step"] == 3
    mgr.save_best(_ckpt(m, [], prov))
    assert mgr.best_path.exists() and load_checkpoint_file(mgr.best_path)["optimizer"] is None
    assert mgr.resolve_resume("auto") == mgr.last_path
    assert mgr.resolve_resume(None) is None
    with pytest.raises(CheckpointError):
        mgr.resolve_resume(str(tmp_path / "nope.pt"))
    assert (
        CheckpointManager(tmp_path / "empty", CheckpointCfg(), CTX).resolve_resume("auto") is None
    )


def test_resume_fingerprint_and_missing_optimizer(tmp_path: Path) -> None:
    m = Toy()
    prov = {"vision_encoder": "trained", "text_encoder": "trained"}
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    ck = _ckpt(m, [], prov, optimizer=opt, scheduler=sched)
    path = atomic_torch_save(ck, tmp_path / "r.pt")

    class Task:
        loaded = None

        def load_state_dict(self, d):
            self.loaded = d

    other = Toy(d=5)
    assert model_fingerprint(other) != model_fingerprint(m)
    kw = dict(
        optimizer=opt,
        scheduler=sched,
        scaler=None,
        train_state=TrainState(),
        task=Task(),
        ctx=CTX,
        current_provenance=prov,
    )
    with pytest.raises(CheckpointError, match="fingerprint"):
        load_for_resume(other, path, **kw)
    st = TrainState()
    task = Task()
    out = load_for_resume(Toy(), path, **{**kw, "train_state": st, "task": task})
    assert st.step == 3 and st.epoch == 1 and task.loaded == {"k": 1} and out["step"] == 3
    no_opt = _ckpt(m, [], prov)
    p2 = atomic_torch_save(no_opt, tmp_path / "noopt.pt")
    with pytest.raises(CheckpointError, match="no optimizer state"):
        load_for_resume(Toy(), p2, **kw)
    with pytest.raises(CheckpointError, match="not a ViVL-MoE checkpoint"):
        load_checkpoint_file(atomic_torch_save({"model": {}}, tmp_path / "bad.pt"))


def test_provenance_from_config_and_rng_roundtrip() -> None:
    m = Toy()
    cfg = ModelCfg(
        vision=VisionCfg(name="apple/dfn", pretrained=True),
        text=TextCfg(name="xlmr", pretrained=False),
    )
    t = ProvenanceTracker.from_config(cfg, m)
    assert t.as_dict() == {"vision_encoder": "hf:apple/dfn", "text_encoder": "trained"}
    t.inherit({"vision_encoder": "trained", "decoder": "hf:x"})
    assert t.as_dict()["vision_encoder"] == "trained"
    assert ProvenanceTracker.from_config(cfg, nn.Linear(2, 2)).as_dict() == {}

    state = collect_rng_state(torch.device("cpu"))
    a = torch.rand(3)
    restore_rng_state(state, torch.device("cpu"))
    assert torch.equal(torch.rand(3), a)


class Head(nn.Module):
    """Downstream-style model: the backbone is nested under ``backbone``."""

    def __init__(self, d: int = 4) -> None:
        super().__init__()
        self.backbone = Toy(d)
        self.head = nn.Linear(d, 2)


def test_fingerprint_ignores_dtype_and_towers_are_stored_fp32(tmp_path: Path) -> None:
    m, half = Toy(), Toy()
    half.vision_encoder.to(torch.bfloat16)
    assert model_fingerprint(m) == model_fingerprint(half)
    prov = {"vision_encoder": "trained", "text_encoder": "trained"}
    ck = _ckpt(half, [], prov)
    assert ck["model"]["vision_encoder.lin.weight"].dtype == torch.float32
    assert ck["model"]["vision_encoder.position_ids"].dtype == torch.int64  # non-float untouched
    assert next(half.vision_encoder.parameters()).dtype == torch.bfloat16  # model not touched
    # a frozen bf16 tower (dtype policy) resumes from a checkpoint written by the fp32 model
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    path = atomic_torch_save(_ckpt(m, [], prov, optimizer=opt), tmp_path / "r.pt")

    class Task:
        def load_state_dict(self, d):
            pass

    load_for_resume(
        half,
        path,
        optimizer=torch.optim.AdamW(half.parameters(), lr=1e-3),
        scheduler=None,
        scaler=None,
        train_state=TrainState(),
        task=Task(),
        ctx=CTX,
        current_provenance=prov,
    )
    assert torch.equal(
        half.vision_encoder.lin.weight.float(), m.vision_encoder.lin.weight.bfloat16().float()
    )


def test_provenance_tracker_uses_tower_paths_on_nested_models() -> None:
    cfg = ModelCfg(
        vision=VisionCfg(name="apple/dfn", pretrained=True),
        text=TextCfg(name="xlmr", pretrained=True),
    )
    head = Head()
    tracker = ProvenanceTracker.from_config(cfg, head)
    assert tracker.as_dict() == {
        "backbone.vision_encoder": "hf:apple/dfn",
        "backbone.text_encoder": "hf:xlmr",
    }
    for p in head.backbone.vision_encoder.parameters():
        p.requires_grad_(False)
    assert tracker.mark_trainable_towers(head) == ["backbone.text_encoder"]
    assert tracker.excluded_prefixes(head, True) == ["backbone.vision_encoder"]
    ck = _ckpt(head, ["backbone.vision_encoder"], tracker.as_dict())
    assert not any(k.startswith("backbone.vision_encoder") for k in ck["model"])
    assert "head.weight" in ck["model"] and "backbone.encoder.weight" in ck["model"]


def test_backbone_checkpoint_loads_into_head_with_remap(tmp_path: Path) -> None:
    torch.manual_seed(2)
    src = Toy()
    for p in src.vision_encoder.parameters():
        p.requires_grad_(False)
    prov = {"vision_encoder": "hf:clip", "text_encoder": "trained"}
    ck = _ckpt(src, ["vision_encoder"], prov)
    current = {"backbone.vision_encoder": "hf:clip", "backbone.text_encoder": "trained"}

    dst = Head()
    report = load_partial(dst, ck, ["all"], current)
    assert report.key_prefix == "backbone." and report.rematerialised == ["backbone.vision_encoder"]
    assert report.provenance == current
    assert _sd_equal(src, dst.backbone, "text_encoder") and _sd_equal(src, dst.backbone, "encoder")
    assert torch.equal(dst.backbone.logit_scale, src.logit_scale)

    dst2 = Head()
    before = {k: v.clone() for k, v in dst2.state_dict().items()}
    report = load_partial(dst2, ck, ["encoder", "text_encoder"], current)
    assert report.rematerialised == [] and report.provenance == {"backbone.text_encoder": "trained"}
    assert all(
        k.startswith(("backbone.encoder", "backbone.text_encoder")) for k in report.loaded_keys
    )
    assert _sd_equal(src, dst2.backbone, "text_encoder") and _sd_equal(
        src, dst2.backbone, "encoder"
    )
    assert torch.equal(dst2.head.weight, before["head.weight"])
    assert torch.equal(dst2.backbone.logit_scale, before["backbone.logit_scale"])
    # the model namespace is accepted too, and the excluded tower is re-materialised per prefix
    r3 = load_partial(Head(), ck, ["backbone.vision_encoder"], current)
    assert r3.rematerialised == ["backbone.vision_encoder"] and r3.loaded_keys == []
    with pytest.raises(CheckpointError, match="cannot be re-materialised"):
        load_partial(Head(), ck, ["all"], {**current, "backbone.vision_encoder": "hf:other"})
    with pytest.raises(ConfigError, match="matches nothing"):
        load_partial(Head(), ck, ["decoder"], current)
    # a head checkpoint into a head model: no remap
    head_ck = _ckpt(Head(), [], current)
    assert load_model_state(Head(), head_ck, current).key_prefix == ""
    # a head checkpoint into a bare backbone is a strict-diff error, never a silent remap
    with pytest.raises(CheckpointError, match="strict model load failed"):
        load_model_state(Toy(), head_ck, prov)


def test_resume_auto_dangling_symlink_is_error(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path / "ckpt", CheckpointCfg(keep_last_n=1), CTX)
    ck = _ckpt(Toy(), [], {"vision_encoder": "trained", "text_encoder": "trained"})
    mgr.save_step(ck, 1)
    mgr.step_path(1).unlink()
    with pytest.raises(CheckpointError, match="dangling symlink"):
        mgr.resolve_resume("auto")
