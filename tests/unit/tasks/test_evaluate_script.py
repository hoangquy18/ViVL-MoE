"""scripts/evaluate.py and scripts/sweep_seeds.py in-process; downstream configs load with a sidecar."""

from __future__ import annotations

import importlib.util
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.unit.tasks.test_captioning_task import SMOKE, run_smoke
from vivlmoe.config import load_config, to_dict
from vivlmoe.config.schema import DEFAULT_ASPECTS, ConfigError
from vivlmoe.models.factory import load_model_cfg
from vivlmoe.tasks.evaluators.dump import read_predictions_jsonl

REPO = Path(__file__).resolve().parents[3]


def _load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        f"vivlmoe_script_{name}", REPO / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    saved = list(root.handlers), root.level
    try:
        yield
    finally:  # setup_logging replaces the root handlers; restore pytest's
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved[0]:
            root.addHandler(h)
        root.setLevel(saved[1])


@pytest.fixture(scope="module")
def caption_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A finished caption_tiny run: ``(tmp root, best.pt)``."""
    tmp = tmp_path_factory.mktemp("caption_run")
    cfg, _task, _result = run_smoke("caption_tiny", tmp)
    best = Path(cfg.run.dir) / cfg.run.name / "checkpoints" / "best.pt"
    assert best.exists()
    return tmp, best


def test_evaluate_script_on_caption_checkpoint(
    caption_run: tuple[Path, Path], restore_logging: None, capsys: pytest.CaptureFixture[str]
) -> None:
    tmp, best = caption_run
    evaluate = _load_script("evaluate")
    syn = str(tmp / "syn")
    out_dir = tmp / "eval_out"
    argv = [
        "--config",
        str(SMOKE / "caption_tiny.yaml"),
        "--checkpoint",
        str(best),
        "--splits",
        "dev",
        "--limit",
        "4",
        "--out-dir",
        str(out_dir),
        f"run.dir={tmp / 'runs'}",
        f"data.train.root={syn}",
        f"data.evals.dev.root={syn}",
    ]
    assert evaluate.main(argv) == 0
    printed = capsys.readouterr().out
    assert "bleu4" in printed and "dev" in printed
    summary = json.loads((out_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert summary["limit"] == 4 and "dev" in summary["splits"]
    # limit 4 records = 2 images (2 captions per image) -> 2 evaluated samples
    assert summary["splits"]["dev"]["num_samples"] == 2.0
    rows = read_predictions_jsonl(out_dir / "predictions" / "dev_step2.jsonl")
    assert len(rows) == 2 and [r["uid"] for r in rows] == [0, 1]
    assert (out_dir / "metrics" / "dev_step2.json").exists()
    with pytest.raises(KeyError, match="not in data.evals"):
        evaluate.main([a if a != "dev" else "test" for a in argv])


def test_sweep_seeds_dry_run_prints_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sweep = _load_script("sweep_seeds")
    argv = [
        "--config",
        str(SMOKE / "caption_tiny.yaml"),
        "--seeds",
        "0",
        "1",
        "--dry-run",
        f"run.dir={tmp_path}",
    ]
    assert sweep.main(argv) == 0
    out = capsys.readouterr().out.splitlines()
    cmd_lines = [ln for ln in out if ln.startswith("[seed ")]
    assert len(cmd_lines) == 2
    assert "run.seed=0" in cmd_lines[0] and "run.name=caption_tiny_seed0" in cmd_lines[0]
    assert "run.seed=1" in cmd_lines[1] and "scripts/train.py" in cmd_lines[1]
    assert "torchrun" not in cmd_lines[0]
    assert f"run.dir={tmp_path}" in cmd_lines[0]
    assert sweep.main([*argv, "--nproc", "4"]) == 0
    out2 = capsys.readouterr().out
    assert "torchrun" in out2 and "--nproc_per_node 4" in out2
    # aggregation: mean / std (ddof=1) over the metrics common to every seed
    summary = sweep.aggregate(
        {
            0: {
                "final": {"final/test/bleu4": 0.2},
                "last_eval": {"dev/bleu4": 0.3},
                "best_metric": 0.3,
            },
            1: {
                "final": {"final/test/bleu4": 0.4},
                "last_eval": {"dev/bleu4": 0.5},
                "best_metric": 0.5,
            },
        }
    )
    assert summary["final/test/bleu4"]["mean"] == pytest.approx(0.3)
    assert summary["final/test/bleu4"]["std"] == pytest.approx(0.1414213562)
    assert summary["final/test/bleu4"]["values"] == {"0": 0.2, "1": 0.4}
    assert "best_metric" in summary and "last/dev/bleu4" in summary
    table = sweep.markdown_table(summary, [0, 1])
    assert table.startswith("| metric | mean | std | seed 0 | seed 1 |")


def test_sweep_seeds_vary_upcycle_seed_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--vary-upcycle-seed: one upcycle.py command per seed, training points at its output."""
    sweep = _load_script("sweep_seeds")
    argv = [
        "--config",
        str(SMOKE / "caption_tiny.yaml"),
        "--seeds",
        "3",
        "4",
        "--dry-run",
        "--vary-upcycle-seed",
        "--upcycle-config",
        "configs/moe/upcycle.yaml",
        "--upcycle-args",
        "--noise-std-rel 0.02",
        f"run.dir={tmp_path}",
    ]
    assert sweep.main(argv) == 0
    out = capsys.readouterr().out.splitlines()
    up = [ln for ln in out if ln.startswith("[upcycle seed ")]
    tr = [ln for ln in out if ln.startswith("[seed ")]
    assert len(up) == 2 and len(tr) == 2
    ckpt = tmp_path / "caption_tiny_upseed3" / "upcycled.pt"
    assert "scripts/upcycle.py" in up[0] and "--seed 3" in up[0] and f"--out {ckpt}" in up[0]
    assert "--config configs/moe/upcycle.yaml" in up[0] and "--noise-std-rel 0.02" in up[0]
    assert f"model.from_checkpoint={ckpt}" in tr[0] and f"init_from.path={ckpt}" in tr[0]
    assert "run.seed=3" in tr[0] and "run.name=caption_tiny_seed3" in tr[0]
    assert "upseed4" in up[1] and "upseed4" in tr[1]
    # without the flag no upcycle command is planned
    assert sweep.main([a for a in argv if a not in ("--vary-upcycle-seed",)]) == 0
    assert not [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[upcycle")]


DOWNSTREAM = [
    "ktvic_caption",
    "uitviic_caption",
    "vimacsa",
    "openvivqa",
    "vitextvqa",
    "viocrvqa",
]


@pytest.mark.parametrize("name", DOWNSTREAM)
def test_downstream_configs_load_with_checkpoint_sidecar(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model section is inherited from a (fake) sparse checkpoint sidecar (tiny model)."""
    monkeypatch.chdir(tmp_path)
    ckpt = tmp_path / "runs" / "moe_sparse" / "checkpoints" / "best.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"not read")
    model = to_dict(load_model_cfg(REPO / "configs" / "model" / "tiny.yaml"))
    # a real runs/moe_sparse checkpoint carries the MoE topology; the stabiliser guard needs it
    model["moe"] = {"num_shared": 1, "num_routed": 8, "top_k": 2, "aux_free_bias": True}
    meta = {
        "format_version": 1,
        "config": {
            "model": model,
            "loss": {"router": {"use_z": True, "use_mi": True, "use_lb": False}},
        },
    }
    (ckpt.parent / "best.pt.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    cfg = load_config(REPO / "configs" / "downstream" / f"{name}.yaml")
    assert cfg.model.from_checkpoint == "runs/moe_sparse/checkpoints/best.pt"
    assert cfg.model.moe is not None and cfg.model.moe.num_routed == 8  # inherited topology
    assert cfg.init_from is not None and cfg.init_from.load == ["all"]
    assert cfg.optim.lr == pytest.approx(3.0e-5) and cfg.data.batch_size == 8
    assert cfg.train.early_stopping.enabled and cfg.train.early_stopping.patience == 3
    assert cfg.eval.final_splits == ["test"] and cfg.eval.selection_metric.startswith("dev/")
    assert cfg.analysis.routing_probe is not None and cfg.analysis.routing_probe.max_steps == 5000
    # epoch-based selection; the probe runs on its own step cadence (Trainer._run_routing_probe)
    assert cfg.eval.every_epochs == 1 and cfg.eval.every_steps is None
    assert cfg.analysis.routing_probe.every_steps == 500
    assert set(cfg.data.evals) == {"dev", "test"} and cfg.data.train is not None
    expected_metrics = {
        "ktvic_caption": ["bleu4", "rougeL"],
        "uitviic_caption": ["bleu4"],
        "vimacsa": ["macro_f1"],
        "openvivqa": ["rougeL", "cider"],
        "vitextvqa": ["f1", "em"],
        "viocrvqa": ["f1", "em"],
    }
    assert cfg.task.metrics == expected_metrics[name]
    if name == "vimacsa":
        assert cfg.train.epochs == 20 and cfg.model.classifier.num_aspects == 6
        assert cfg.task.aspects == list(DEFAULT_ASPECTS) and cfg.task.max_images == 6
    else:
        assert cfg.train.epochs == 10 and cfg.task.generation.num_beams == 5
        # HF-parity generation (the paper only states beam 5): no forced minimum length, no early stop
        assert cfg.task.generation.min_new_tokens == 0 and not cfg.task.generation.early_stopping
    if name == "vimacsa":
        pass  # no generation settings for a classifier
    elif name == "openvivqa":
        assert cfg.task.generation.max_new_tokens == 64
    elif name.endswith("caption"):
        assert cfg.task.generation.max_new_tokens == 40
        assert cfg.data.train.holdout is not None and cfg.data.train.holdout.ids_file.endswith(
            "_dev_image_ids.json"
        )
    else:
        assert cfg.task.generation.max_new_tokens == 24
    # the stabiliser guard: a mismatching use_z is rejected unless allowed
    with pytest.raises(ConfigError, match="stabilizer settings differ"):
        load_config(REPO / "configs" / "downstream" / f"{name}.yaml", ["loss.router.use_z=false"])
    load_config(
        REPO / "configs" / "downstream" / f"{name}.yaml",
        ["loss.router.use_z=false", "task.allow_stabilizer_override=true"],
    )


ABLATION_SWITCHES: dict[str, tuple[dict[str, bool], bool]] = {
    # overlay -> (loss.router switches of the ablation checkpoint, model.moe.aux_free_bias)
    "no_zloss": ({"use_z": False, "use_mi": True, "use_lb": False}, True),
    "no_miloss": ({"use_z": True, "use_mi": False, "use_lb": False}, True),
    "no_auxfree_bias": ({"use_z": True, "use_mi": True, "use_lb": False}, False),
    "switch_lb": ({"use_z": True, "use_mi": True, "use_lb": True}, False),
}


def _write_sidecar(ckpt: Path, router: dict[str, bool], aux_free_bias: bool) -> None:
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"not read")
    model = to_dict(load_model_cfg(REPO / "configs" / "model" / "tiny.yaml"))
    model["moe"] = {"num_shared": 1, "num_routed": 8, "top_k": 2, "aux_free_bias": aux_free_bias}
    meta = {"format_version": 1, "config": {"model": model, "loss": {"router": dict(router)}}}
    (ckpt.parent / f"{ckpt.name}.meta.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.mark.parametrize("abl", sorted(ABLATION_SWITCHES))
def test_downstream_ablation_overlays_match_their_checkpoint(
    abl: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A removed stabiliser stays off downstream: the overlay restates the switches of the
    ablation checkpoint and the loader guard verifies them (no allow_stabilizer_override)."""
    monkeypatch.chdir(tmp_path)
    router, bias = ABLATION_SWITCHES[abl]
    ckpt = tmp_path / "runs" / f"ablation_{abl}" / "checkpoints" / "best.pt"
    _write_sidecar(ckpt, router, bias)
    path = REPO / "configs" / "ablations" / f"downstream_{abl}.yaml"
    cfg = load_config(path)
    rel = f"runs/ablation_{abl}/checkpoints/best.pt"
    assert cfg.task.name == "captioning" and cfg.run.name == f"ktvic_caption_{abl}"
    assert cfg.model.from_checkpoint == rel and cfg.init_from.path == rel
    assert cfg.model.moe is not None and cfg.model.moe.aux_free_bias == bias
    assert (cfg.loss.router.use_z, cfg.loss.router.use_mi, cfg.loss.router.use_lb) == (
        router["use_z"],
        router["use_mi"],
        router["use_lb"],
    )
    if abl == "switch_lb":
        assert cfg.loss.router.lambda_lb == pytest.approx(0.01)
    assert not cfg.task.allow_stabilizer_override
    assert cfg.task.metrics == ["bleu4", "rougeL"] and cfg.eval.selection_metric == "dev/bleu4"
    # the same overlay on a checkpoint trained with every stabiliser on is refused
    _write_sidecar(ckpt, {"use_z": True, "use_mi": True, "use_lb": False}, True)
    with pytest.raises(ConfigError, match="stabilizer settings differ"):
        load_config(path)


def test_smoke_configs_load() -> None:
    for name in ("caption_tiny", "vqa_tiny", "vimacsa_tiny"):
        cfg = load_config(SMOKE / f"{name}.yaml")
        assert cfg.model.from_checkpoint is None and cfg.init_from is None
        assert (
            cfg.train.max_steps == 4
            and cfg.train.grad_accum_steps == 2
            and cfg.data.batch_size == 2
        )
        assert not cfg.train.amp.enabled and cfg.data.num_workers == 0 and cfg.eval.every_steps == 2
        if name != "vimacsa_tiny":
            assert cfg.task.generation.num_beams == 2 and cfg.task.generation.max_new_tokens == 4
