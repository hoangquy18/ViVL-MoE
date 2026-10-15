from __future__ import annotations

import json
from pathlib import Path

import pytest

from vivlmoe.config import (
    ConfigError,
    config_fingerprint,
    load_config,
    parse_overrides,
    save_resolved,
    to_dict,
)
from vivlmoe.config.loader import from_dict
from vivlmoe.config.schema import MoECfg, RunConfig

BASE = """
_include: [../_base/train_defaults.yaml, ../model/tiny.yaml]
run: {name: t}
task: {name: contrastive}
train: {max_steps: 3}
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture()
def cfg_dir(tmp_path: Path) -> Path:
    root = Path(__file__).resolve().parents[3] / "configs"
    (tmp_path / "_base").mkdir()
    (tmp_path / "model").mkdir()
    for rel in [
        "_base/train_defaults.yaml",
        "model/paper.yaml",
        "model/tiny.yaml",
        "model/small.yaml",
    ]:
        (tmp_path / rel).write_text((root / rel).read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "runs").mkdir()
    return tmp_path


def test_include_chain_and_precedence(cfg_dir: Path) -> None:
    a = _write(cfg_dir, "runs/a.yaml", "x: {p: 1, q: 1}\nlist: [1, 2]\n")
    b = _write(cfg_dir, "runs/b.yaml", "x: {q: 2, r: 2}\n")
    c = _write(cfg_dir, "runs/c.yaml", f"_include: [{a.name}, {b.name}]\nx: {{r: 3}}\nlist: [9]\n")
    from vivlmoe.config.loader import load_yaml_with_includes

    merged = load_yaml_with_includes(c)
    assert merged == {"x": {"p": 1, "q": 2, "r": 3}, "list": [9]}


def test_include_cycle_raises(cfg_dir: Path) -> None:
    _write(cfg_dir, "runs/x.yaml", "_include: [y.yaml]\n")
    _write(cfg_dir, "runs/y.yaml", "_include: [x.yaml]\n")
    from vivlmoe.config.loader import load_yaml_with_includes

    with pytest.raises(ConfigError, match="cycle"):
        load_yaml_with_includes(cfg_dir / "runs/x.yaml")


def test_load_tiny_config_and_roundtrip(cfg_dir: Path) -> None:
    p = _write(cfg_dir, "runs/stage.yaml", BASE)
    cfg = load_config(p)
    assert isinstance(cfg, RunConfig)
    assert cfg.model.vision.pretrained is False
    assert cfg.model.encoder.dim == 32
    assert cfg.model.moe is None
    out = save_resolved(cfg, cfg_dir / "runs/resolved.yaml", argv=["x"])
    cfg2 = load_config(out)
    assert to_dict(cfg2) == to_dict(cfg)
    assert config_fingerprint(cfg2) == config_fingerprint(cfg)


def test_cli_overrides_beat_file(cfg_dir: Path) -> None:
    p = _write(cfg_dir, "runs/stage.yaml", BASE)
    cfg = load_config(
        p,
        ["train.max_steps=7", "optim.lr=1e-5", "loss.router.use_z=false", "model.moe.num_routed=4"],
    )
    assert cfg.train.max_steps == 7
    assert cfg.optim.lr == pytest.approx(1e-5)
    assert cfg.loss.router.use_z is False
    assert cfg.model.moe is not None and cfg.model.moe.num_routed == 4 and cfg.model.moe.top_k == 2
    with pytest.raises(ConfigError, match="key.path=value"):
        parse_overrides(["train.max_steps"])


def test_unknown_key_reports_full_path(cfg_dir: Path) -> None:
    p = _write(cfg_dir, "runs/stage.yaml", BASE + "train: {grad_accum: 2}\n")
    with pytest.raises(ConfigError) as exc:
        load_config(p)
    assert "grad_accum" in str(exc.value) and "'train'" in str(exc.value)


def test_type_coercion_rules() -> None:
    from vivlmoe.config.schema import TrainCfg

    with pytest.raises(ConfigError, match="expected an int"):
        from_dict(TrainCfg, {"max_steps": 1.5})
    assert from_dict(TrainCfg, {"max_steps": 3}).max_steps == 3
    with pytest.raises(ConfigError, match="not one of"):
        from_dict(TrainCfg, {"amp": {"dtype": "fp8"}})
    from vivlmoe.config.schema import OptimCfg

    o = from_dict(OptimCfg, {"lr": "1e-3", "betas": [0.9, 0.98]})
    assert o.lr == pytest.approx(1e-3) and o.betas == (0.9, 0.98)
    with pytest.raises(ConfigError, match="expected a bool"):
        from_dict(OptimCfg, {"no_decay_norm_and_bias": "yes"})
    with pytest.raises(ConfigError, match="Missing required"):
        from_dict(
            MoECfg.__class__
            if False
            else __import__("vivlmoe.config.schema", fromlist=["TaskCfg"]).TaskCfg,
            {},
        )


def test_moe_validation() -> None:
    with pytest.raises(ConfigError, match="top_k"):
        MoECfg(num_routed=1, top_k=2)
    assert MoECfg(num_routed=1, top_k=1).top_k == 1
    with pytest.raises(ConfigError, match="num_shared"):
        MoECfg(num_shared=9)


def test_cross_field_validation(cfg_dir: Path) -> None:
    p = _write(cfg_dir, "runs/stage.yaml", BASE)
    with pytest.raises(ConfigError, match="max_steps / epochs"):
        load_config(p, ["train.epochs=2"])
    with pytest.raises(ConfigError, match="data.evals"):
        load_config(p, ["eval.selection_metric=dev/rsum"])
    with pytest.raises(ConfigError, match="warmup"):
        load_config(p, ["schedule.warmup_steps=10"])


def test_from_checkpoint_inherits_model_section(cfg_dir: Path) -> None:
    base = load_config(
        _write(cfg_dir, "runs/stage.yaml", BASE), ["model.moe.num_routed=4", "model.moe.top_k=1"]
    )
    ckpt = cfg_dir / "runs/best.pt"
    ckpt.write_bytes(b"")
    meta = {"format_version": 1, "config": to_dict(base), "step": 5}
    (cfg_dir / "runs/best.pt.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    child = _write(
        cfg_dir,
        "runs/child.yaml",
        "_include: [../_base/train_defaults.yaml]\nrun: {name: c}\ntask: {name: captioning}\ntrain: {max_steps: 1}\n"
        f"model: {{from_checkpoint: {ckpt}}}\nmodel_overrides: {{encoder: {{dropout: 0.0}}}}\n",
    )
    cfg = load_config(child)
    assert cfg.model.moe is not None and cfg.model.moe.num_routed == 4 and cfg.model.moe.top_k == 1
    assert cfg.model.encoder.dim == 32 and cfg.model.encoder.dropout == 0.0
    assert cfg.model.from_checkpoint == str(ckpt)
    # stabilizer mismatch is rejected unless explicitly allowed
    with pytest.raises(ConfigError, match="stabilizer"):
        load_config(child, ["loss.router.use_mi=false"])
    ok = load_config(child, ["loss.router.use_mi=false", "task.allow_stabilizer_override=true"])
    assert ok.loss.router.use_mi is False
    # other model keys next to from_checkpoint are an error
    bad = _write(cfg_dir, "runs/bad.yaml", child.read_text() + "model: {encoder: {dim: 16}}\n")
    with pytest.raises(ConfigError, match="model_overrides"):
        load_config(bad)


def test_fingerprint_ignores_from_checkpoint() -> None:
    a = {"model": {"encoder": {"dim": 32}, "from_checkpoint": None}}
    b = {"model": {"encoder": {"dim": 32}, "from_checkpoint": "x.pt"}}
    assert config_fingerprint(a) == config_fingerprint(b)
    assert config_fingerprint(a) != config_fingerprint({"model": {"encoder": {"dim": 64}}})
