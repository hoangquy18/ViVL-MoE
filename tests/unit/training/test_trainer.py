from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch

from tests.unit.training.synthetic_task import SyntheticTask, make_cfg
from vivlmoe.config.schema import ConfigError
from vivlmoe.training.amp import AmpState, _make_scaler
from vivlmoe.training.checkpoint import load_checkpoint_file
from vivlmoe.training.distributed import DistContext, seed_everything
from vivlmoe.training.trainer import Trainer
from vivlmoe.utils.ckpt_meta import read_meta

CTX = DistContext.single_process("cpu")


def _run(
    tmp_path: Path,
    name: str,
    task_kwargs: dict | None = None,
    resume: str | None = None,
    dry_run: bool = False,
    **cfg_over,
):
    run_dir = tmp_path / name
    cfg = make_cfg(run_dir, **cfg_over)
    task = SyntheticTask(cfg, CTX, **(task_kwargs or {}))
    trainer = Trainer(cfg, task, CTX, run_dir, resume=resume, dry_run=dry_run)
    result = trainer.fit()
    return trainer, task, result


def _params(trainer: Trainer) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in trainer.raw_model.state_dict().items()}


def test_accumulation_equivalence(tmp_path: Path) -> None:
    a, ta, _ = _run(tmp_path, "a", {"dropout": 0.0}, data={"batch_size": 8}, train={"max_steps": 1})
    b, tb, _ = _run(
        tmp_path,
        "b",
        {"dropout": 0.0},
        data={"batch_size": 4},
        train={"max_steps": 1, "grad_accum_steps": 2},
    )
    assert [u for _, uids, _ in ta.records for u in uids] == [
        u for _, uids, _ in tb.records for u in uids
    ]
    pa, pb = _params(a), _params(b)
    for k in pa:
        assert torch.allclose(pa[k], pb[k], atol=1e-6), k
    assert ta.records[0][2] == pytest.approx((tb.records[0][2] + tb.records[1][2]) / 2, rel=1e-5)
    assert a.effective_batch == b.effective_batch == 8


def test_budget_by_steps_and_epochs(tmp_path: Path) -> None:
    t, task, res = _run(tmp_path, "s", {"n_train": 64}, train={"max_steps": 5})
    assert t.state.step == 5 and res["step"] == 5 and len(task.records) == 5
    assert t.scheduler.last_epoch == 5  # stepped exactly once per optimizer step
    t2, task2, _ = _run(tmp_path, "e", {"n_train": 16}, train={"max_steps": None, "epochs": 2})
    assert t2.state.step == 8 and t2.state.epoch == 2 and t2.total_steps == 8
    ep0 = [u for s, uids, _ in task2.records if s < 4 for u in uids]
    ep1 = [u for s, uids, _ in task2.records if s >= 4 for u in uids]
    assert sorted(ep0) == list(range(16)) == sorted(ep1) and ep0 != ep1
    t3, task3, _ = _run(
        tmp_path,
        "e3",
        {"n_train": 16},
        train={"max_steps": None, "epochs": 2, "grad_accum_steps": 3},
    )
    assert (
        t3.steps_per_epoch == 1 and t3.state.step == 2 and len(task3.records) == 6
    )  # tail micro-batch dropped


def test_eval_cadence_early_stopping_and_best(tmp_path: Path) -> None:
    scores = [1.0, 2.0, 1.5, 1.0, 0.5, 0.1]
    t, task, res = _run(
        tmp_path,
        "es",
        {"scripted_scores": scores},
        train={"max_steps": 20, "early_stopping": {"enabled": True, "patience": 2}},
        eval={"every_steps": 2, "final_splits": ["dev"]},
        checkpoint={"every_steps": 4},
    )
    assert t.state.step == 8 and t.state.best_step == 4 and t.state.best_metric == 2.0
    assert task.evaluators()["dev"].calls == 5  # 4 cadence evals + final
    ck = t.ckpt
    assert read_meta(ck.best_path)["step"] == 4
    assert load_checkpoint_file(ck.best_path)["optimizer"] is None  # save_optimizer: last_only
    assert load_checkpoint_file(ck.last_path)["optimizer"] is not None
    assert sorted(p.name for p in ck.step_files()) == ["step_0000004.pt", "step_0000008.pt"]
    assert (
        res["best_metric"] == 2.0 and res["final"]["final/dev/score"] == 0.5
    )  # 5th evaluator call
    lines = [
        json.loads(line)
        for line in (t.run_dir / "eval_history.jsonl").read_text().strip().splitlines()
    ]
    assert len(lines) == 5 and lines[-1]["final"] is True
    at_best = next(line for line in lines if line["step"] == 4)
    assert res["final"]["final/dev/mse"] == at_best["dev/mse"]  # final eval used best.pt weights
    assert lines[-2]["dev/mse"] != at_best["dev/mse"]
    assert read_meta(ck.last_path)["dev_split_sha1"] == "synthetic-dev"
    assert load_checkpoint_file(ck.last_path)["evaluator_state"] == {"dev": {"calls": 4}}


def test_mid_epoch_resume_reproduces_batches_and_losses(tmp_path: Path) -> None:
    common = dict(
        train={"max_steps": 11},
        checkpoint={"every_steps": 3, "keep_last_n": 10},
        eval={"every_epochs": 1},
    )
    a, ta, _ = _run(tmp_path, "A", {"n_train": 32, "dropout": 0.2}, **common)
    assert a.steps_per_epoch == 8 and a.state.epoch == 1
    ckpt3 = a.ckpt.step_path(3)
    b, tb, _ = _run(tmp_path, "B", {"n_train": 32, "dropout": 0.2}, resume=str(ckpt3), **common)
    assert b.state.step == 11 and b.state.epoch == 1
    rec_a = [r for r in ta.records if r[0] >= 3]
    assert [r[:2] for r in tb.records] == [r[:2] for r in rec_a]  # same (step, uids)
    assert [r[2] for r in tb.records] == [r[2] for r in rec_a]  # same losses (dropout RNG restored)
    pa, pb = _params(a), _params(b)
    assert all(torch.equal(pa[k], pb[k]) for k in pa)
    assert tb.evaluators()["dev"].calls == 1 and ta.evaluators()["dev"].calls == 1
    # resuming a finished run does nothing
    c, tc, _ = _run(tmp_path, "A", {"n_train": 32, "dropout": 0.2}, resume="auto", **common)
    assert c.state.step == 11 and tc.records == []


def test_nonfinite_raise_skip_and_abort(tmp_path: Path) -> None:
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        _run(tmp_path, "raise", {"nan_micro_batches": (1,), "dropout": 0.0}, train={"max_steps": 3})

    run_dir = tmp_path / "skip"
    cfg = make_cfg(run_dir, train={"max_steps": 2})
    task = SyntheticTask(cfg, CTX, nan_micro_batches=(1,), dropout=0.0)
    tr = Trainer(cfg, task, CTX, run_dir)
    tr.amp = AmpState(
        False, None, "cpu", _make_scaler(False), "skip"
    )  # emulate the fp16 policy on CPU
    tr.fit()
    assert (
        tr.state.step == 2 and tr.state.skipped_steps == 1 and tr.state.nonfinite_micro_batches == 1
    )
    assert tr.scheduler.last_epoch == 2
    ref, _, _ = _run(tmp_path, "ref", {"dropout": 0.0}, train={"max_steps": 1})
    p_skip, p_ref = _params(tr), _params(ref)
    assert all(torch.equal(p_skip[k], p_ref[k]) for k in p_ref)  # the skipped step changed nothing

    cfg2 = make_cfg(tmp_path / "abort", train={"max_steps": 10, "max_consecutive_nonfinite": 2})
    task2 = SyntheticTask(cfg2, CTX, nan_micro_batches=tuple(range(10)))
    tr2 = Trainer(cfg2, task2, CTX, tmp_path / "abort")
    tr2.amp = AmpState(False, None, "cpu", _make_scaler(False), "skip")
    with pytest.raises(RuntimeError, match="consecutive non-finite"):
        tr2.fit()


def test_frozen_module_stays_eval_and_untouched(tmp_path: Path) -> None:
    t, task, _ = _run(tmp_path, "frozen", train={"max_steps": 4}, eval={"every_steps": 2})
    raw = t.raw_model
    assert raw.training and raw.body.training and not raw.frozen.training
    assert all(not p.requires_grad and p.grad is None for p in raw.frozen.parameters())
    assert task.train_mode_calls >= 3  # startup + after each eval
    assert "frozen" in t.frozen_paths and "frozen.0.weight" not in {
        n for g in t.param_groups for n in g.param_names
    }


def test_dry_run(tmp_path: Path) -> None:
    t, task, report = _run(
        tmp_path, "dry", dry_run=True, train={"max_steps": 50}, checkpoint={"every_steps": 1}
    )
    assert report["dry_run"] and report["steps"] == 2 and len(task.records) == 2
    assert (
        report["effective_batch"] == 4
        and report["trainable_params"] > 0
        and len(report["losses"]) == 2
    )
    assert not (t.run_dir / "checkpoints").exists() or not list(
        (t.run_dir / "checkpoints").iterdir()
    )
    assert not (t.run_dir / "eval_history.jsonl").exists()


def test_unused_trainable_param_is_hard_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="received no gradient.*unused"):
        _run(tmp_path, "unused", {"unused_param": True}, train={"max_steps": 1})


def test_freeze_preset_and_init_from(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("vivlmoe.models.freezing")
    fake.freeze_paths_for_preset = lambda model, preset: ["head"]  # type: ignore[attr-defined]
    fake.apply_frozen_dtype_policy = lambda model, amp_dtype=None: {}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vivlmoe.models.freezing", fake)
    a, _, _ = _run(
        tmp_path,
        "src",
        train={"max_steps": 3},
        optim={"freeze_preset": "stage1", "freeze": ["scale"]},
    )
    assert a.frozen_paths == ["frozen", "head", "scale"]
    assert all(not p.requires_grad for p in a.raw_model.head.parameters())
    pa = _params(a)
    seed_everything(1)  # same seed the Trainer uses before build_model()
    fresh = SyntheticTask(make_cfg(tmp_path / "fresh"), CTX).build_model()
    assert not torch.equal(fresh.body[0].weight, pa["body.0.weight"])  # a trained the body

    run_dir = tmp_path / "dst"
    cfg = make_cfg(
        run_dir, train={"max_steps": 1}, init_from={"path": str(a.ckpt.last_path), "load": ["body"]}
    )
    b = Trainer(cfg, SyntheticTask(cfg, CTX), CTX, run_dir)
    pb = _params(b)
    assert all(torch.equal(pa[k], pb[k]) for k in pa if k.startswith("body."))
    assert torch.equal(pb["head.weight"], fresh.head.weight)  # head not loaded: fresh seeded init
    b.fit()
    with pytest.raises(ConfigError, match="matches nothing"):
        _run(
            tmp_path,
            "bad",
            train={"max_steps": 1},
            init_from={"path": str(a.ckpt.last_path), "load": ["decoder"]},
        )


def test_missing_evaluator_wiring_is_error(tmp_path: Path) -> None:
    run_dir = tmp_path / "wire"
    cfg = make_cfg(run_dir, train={"max_steps": 1})
    task = SyntheticTask(cfg, CTX)
    task.evaluators = lambda: {}  # type: ignore[method-assign]
    with pytest.raises(ConfigError, match="loaders without evaluator"):
        Trainer(cfg, task, CTX, run_dir)


def test_epoch_boundary_resume(tmp_path: Path) -> None:
    common = dict(
        train={"max_steps": 8},
        checkpoint={"every_steps": 4, "keep_last_n": 10},
        eval={"every_epochs": 1},
    )
    a, ta, _ = _run(tmp_path, "A", {"n_train": 16, "dropout": 0.2}, **common)
    assert a.steps_per_epoch == 4 and a.state.epoch == 2
    ck4 = load_checkpoint_file(a.ckpt.step_path(4))
    st = ck4["train_state"]
    assert st["epoch"] == 1 and st["micro_in_epoch"] == 0 and st["last_eval_step"] == 4
    # resume in the same run dir from the boundary checkpoint: fresh epoch, no repeated eval
    b, tb, _ = _run(
        tmp_path, "A", {"n_train": 16, "dropout": 0.2}, resume=str(a.ckpt.step_path(4)), **common
    )
    assert b.state.step == 8 and b.state.epoch == 2
    rec_a = [r for r in ta.records if r[0] >= 4]
    assert [r[:2] for r in tb.records] == [r[:2] for r in rec_a]
    assert [r[2] for r in tb.records] == [r[2] for r in rec_a]
    assert tb.evaluators()["dev"].calls == 2  # 1 restored from the checkpoint + the epoch-2 eval
    lines = [
        json.loads(line)
        for line in (b.run_dir / "eval_history.jsonl").read_text().strip().splitlines()
    ]
    assert [line["step"] for line in lines].count(4) == 1
    pa, pb = _params(a), _params(b)
    assert all(torch.equal(pa[k], pb[k]) for k in pa)


def test_consecutive_skipped_steps_abort(tmp_path: Path) -> None:
    run_dir = tmp_path / "one"
    cfg = make_cfg(run_dir, train={"max_steps": 3, "max_consecutive_nonfinite": 2})
    task = SyntheticTask(cfg, CTX, inf_grad_micro_batches=(1,), dropout=0.0)
    tr = Trainer(cfg, task, CTX, run_dir)
    tr.amp = AmpState(False, None, "cpu", _make_scaler(False), "skip")
    tr.fit()
    assert tr.state.step == 3 and tr.state.skipped_steps == 1
    assert tr.state.nonfinite_micro_batches == 0  # the loss itself was finite

    cfg2 = make_cfg(tmp_path / "abort", train={"max_steps": 10, "max_consecutive_nonfinite": 2})
    task2 = SyntheticTask(cfg2, CTX, inf_grad_micro_batches=tuple(range(10)), dropout=0.0)
    tr2 = Trainer(cfg2, task2, CTX, tmp_path / "abort")
    tr2.amp = AmpState(False, None, "cpu", _make_scaler(False), "skip")
    with pytest.raises(RuntimeError, match="consecutive non-finite gradient norms"):
        tr2.fit()
    assert tr2.state.skipped_steps == 3


def test_frozen_bf16_tower_fingerprint_and_resume(tmp_path: Path) -> None:
    from tests.unit.training.backbone_task import TinyBackboneTask, make_backbone_cfg
    from vivlmoe.training.checkpoint import model_fingerprint

    run_dir = tmp_path / "bb"
    cfg = make_backbone_cfg(run_dir, frozen_dtype="bf16", checkpoint={"every_steps": 1})
    task = TinyBackboneTask(cfg, CTX)
    tr = Trainer(cfg, task, CTX, run_dir)
    raw = tr.raw_model
    assert tr.tower_dtypes == {"vision_encoder": torch.bfloat16}
    assert next(raw.vision_encoder.model.parameters()).dtype == torch.bfloat16
    assert not raw.vision_encoder.model.training and raw.text_encoder.model.training
    assert tr.provenance.as_dict() == {"vision_encoder": "trained", "text_encoder": "trained"}
    assert tr.excluded == []  # random init cannot be re-materialised: stored (as fp32)
    fp_before = model_fingerprint(raw)
    tr.fit()
    assert model_fingerprint(raw) == fp_before
    ck = load_checkpoint_file(tr.ckpt.last_path)
    assert ck["model_fingerprint"] == fp_before
    assert (
        ck["model"]["vision_encoder.model.vision_model.embeddings.class_embedding"].dtype
        == torch.float32
    )
    assert (
        ck["model"]["text_encoder.model.embeddings.word_embeddings.weight"].dtype == torch.float32
    )
    # --resume auto: same architecture -> no fingerprint error, one more step
    cfg2 = make_backbone_cfg(
        run_dir, frozen_dtype="bf16", checkpoint={"every_steps": 1}, train={"max_steps": 3}
    )
    tr2 = Trainer(cfg2, TinyBackboneTask(cfg2, CTX), CTX, run_dir, resume="auto")
    assert tr2.state.step == 2
    tr2.fit()
    assert tr2.state.step == 3


def test_fp16_amp_with_bf16_frozen_tower_is_config_error(tmp_path: Path) -> None:
    run_dir = tmp_path / "amp"
    cfg = make_cfg(run_dir, train={"amp": {"enabled": True, "dtype": "fp16"}})
    cfg.model.vision.frozen_dtype = "bf16"
    with pytest.raises(ConfigError, match="frozen_dtype"):
        Trainer(cfg, SyntheticTask(cfg, CTX), CTX, run_dir)
    # "auto" follows the AMP dtype (fp16 tower under fp16 autocast) and "fp32" is always allowed.
    for dtype in ("auto", "fp32"):
        cfg.model.vision.frozen_dtype = dtype  # type: ignore[assignment]
        Trainer(cfg, SyntheticTask(cfg, CTX), CTX, run_dir)  # AMP is disabled on CPU with a warning
