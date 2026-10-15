"""configs/moe, configs/ablations and the MoE _smoke configs resolve against a checkpoint sidecar."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import CONFIGS
from vivlmoe.config import load_config
from vivlmoe.config.loader import load_yaml_with_includes
from vivlmoe.config.schema import ConfigError

TINY_MODEL = load_yaml_with_includes(CONFIGS / "model" / "tiny.yaml")["model"]
PAPER_MOE = {
    "num_shared": 1,
    "num_routed": 8,
    "top_k": 2,
    "aux_free_bias": True,
    "bias_update_rate": 1.0e-3,
}


def _sidecar(tmp_path: Path, name: str, moe: dict | None, router: dict | None = None) -> Path:
    ckpt = tmp_path / f"{name}.pt"
    config: dict = {"model": {**TINY_MODEL, "moe": moe}}
    if router is not None:
        config["loss"] = {"router": router}
    meta = {"format_version": 1, "config": config, "tower_provenance": {}, "step": 0}
    (tmp_path / f"{name}.pt.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return ckpt


def test_upcycle_config(tmp_path: Path) -> None:
    dense = _sidecar(tmp_path, "dense", None)
    cfg = load_config(
        CONFIGS / "moe" / "upcycle.yaml", [f"model.from_checkpoint={dense}", "train.max_steps=1"]
    )
    assert cfg.task.name == "contrastive" and cfg.model.from_checkpoint == str(dense)
    moe = cfg.model.moe
    assert moe is not None and (moe.num_shared, moe.num_routed, moe.top_k) == (1, 8, 2)
    assert moe.aux_free_bias and moe.bias_update_rate == 1.0e-3 and moe.bias_update_every == "step"
    assert cfg.model.encoder.dim == TINY_MODEL["encoder"]["dim"]  # inherited from the dense sidecar
    assert cfg.model_overrides["moe"]["num_routed"] == 8
    raw = load_yaml_with_includes(CONFIGS / "moe" / "upcycle.yaml")
    assert "upcycle" not in raw and set(raw["model"]) == {"from_checkpoint"}  # no schema section


def _sparse_overrides(
    tmp_path: Path, moe: dict | None = None, router: dict | None = None
) -> list[str]:
    up = _sidecar(
        tmp_path,
        "upcycled",
        moe or PAPER_MOE,
        router or {"use_z": True, "use_mi": True, "use_lb": False},
    )
    return [f"model.from_checkpoint={up}", f"init_from.path={up}"]


def test_sparse_train_config(tmp_path: Path) -> None:
    cfg = load_config(CONFIGS / "moe" / "sparse_train.yaml", _sparse_overrides(tmp_path))
    assert cfg.task.name == "contrastive" and cfg.run.name == "moe_sparse"
    assert cfg.init_from is not None and cfg.init_from.load == ["all"]
    assert cfg.optim.lr == 1.0e-5 and cfg.optim.freeze_preset == "sparse"
    assert (
        cfg.data.batch_size == 128
        and cfg.train.grad_accum_steps == 4
        and cfg.train.max_steps == 20000
    )
    assert cfg.data.batch_size * cfg.train.grad_accum_steps * 4 == 2048
    assert (
        cfg.eval.every_steps == 500
        and cfg.eval.every_epochs is None
        and cfg.eval.selection_metric == "ktvic_dev/rsum"
    )
    assert cfg.checkpoint.every_steps == 1000 and cfg.checkpoint.keep_last_n == 3
    assert (
        cfg.model.moe is not None and cfg.model.moe.num_routed == 8 and cfg.model.moe.aux_free_bias
    )
    rp = cfg.analysis.routing_probe
    assert rp is not None and rp.split == "ktvic_dev" and rp.every_steps == 500
    assert rp.split in cfg.data.evals and cfg.data.train is not None  # from configs/data/*.yaml
    r = cfg.loss.router
    assert (r.lambda_z, r.lambda_mi, r.use_z, r.use_mi, r.use_lb) == (
        1.0e-3,
        5.0e-3,
        True,
        True,
        False,
    )


@pytest.mark.parametrize("n", range(6))
def test_ns_ablations(tmp_path: Path, n: int) -> None:
    cfg = load_config(
        CONFIGS / "ablations" / f"ns{n}.yaml",
        _sparse_overrides(tmp_path, {**PAPER_MOE, "num_shared": n}),
    )
    assert (
        cfg.model.moe is not None
        and cfg.model.moe.num_shared == n
        and cfg.model.moe.num_routed == 8
    )
    assert cfg.run.name == f"ablation_ns{n}" and cfg.train.max_steps == 20000


@pytest.mark.parametrize("n", range(1, 9))
def test_nr_ablations(tmp_path: Path, n: int) -> None:
    name = "nr1_k1" if n == 1 else f"nr{n}"
    k = 1 if n == 1 else 2
    cfg = load_config(
        CONFIGS / "ablations" / f"{name}.yaml",
        _sparse_overrides(tmp_path, {**PAPER_MOE, "num_routed": n, "top_k": k}),
    )
    assert cfg.model.moe is not None and cfg.model.moe.num_routed == n and cfg.model.moe.top_k == k
    assert cfg.model.moe.aux_free_bias and cfg.loss.router.use_z and cfg.loss.router.use_mi


def test_nr1_requires_top_k_1(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="top_k"):
        load_config(
            CONFIGS / "ablations" / "nr1_k1.yaml",
            [*_sparse_overrides(tmp_path), "model_overrides.moe.top_k=2"],
        )


def test_stabilizer_ablations(tmp_path: Path) -> None:
    ov = _sparse_overrides(tmp_path)
    no_z = load_config(CONFIGS / "ablations" / "no_zloss.yaml", ov)
    assert (
        not no_z.loss.router.use_z
        and no_z.loss.router.use_mi
        and no_z.task.allow_stabilizer_override
    )
    no_mi = load_config(CONFIGS / "ablations" / "no_miloss.yaml", ov)
    assert not no_mi.loss.router.use_mi and no_mi.loss.router.use_z
    no_bias = load_config(CONFIGS / "ablations" / "no_auxfree_bias.yaml", ov)
    assert no_bias.model.moe is not None and not no_bias.model.moe.aux_free_bias
    switch = load_config(CONFIGS / "ablations" / "switch_lb.yaml", ov)
    assert switch.loss.router.use_lb and switch.loss.router.lambda_lb == 0.01
    assert switch.model.moe is not None and not switch.model.moe.aux_free_bias
    # without allow_stabilizer_override the loader refuses a stabilizer that differs from the checkpoint
    with pytest.raises(ConfigError, match="stabilizer"):
        load_config(
            CONFIGS / "ablations" / "no_zloss.yaml", [*ov, "task.allow_stabilizer_override=false"]
        )


def test_smoke_moe_configs(tmp_path: Path) -> None:
    dense = _sidecar(tmp_path, "dense", None)
    up = load_config(
        CONFIGS / "_smoke" / "upcycle_tiny.yaml",
        [f"model.from_checkpoint={dense}", "train.max_steps=1"],
    )
    assert (
        up.model.moe is not None and up.model.moe.num_routed == 4 and up.task.name == "contrastive"
    )
    tiny_moe = {
        "num_shared": 1,
        "num_routed": 4,
        "top_k": 2,
        "aux_free_bias": True,
        "bias_update_rate": 1.0e-2,
    }
    cfg = load_config(CONFIGS / "_smoke" / "moe_tiny.yaml", _sparse_overrides(tmp_path, tiny_moe))
    assert (
        cfg.task.name == "contrastive"
        and cfg.optim.freeze_preset == "sparse"
        and cfg.model.moe is not None
    )
    assert (
        cfg.data.train is not None
        and cfg.data.train.kind == "synthetic_pairs"
        and cfg.data.num_workers == 0
    )
    assert (
        not cfg.train.amp.enabled and cfg.train.max_steps == 4 and cfg.train.grad_accum_steps == 2
    )
    assert cfg.analysis.routing_probe is not None and cfg.analysis.routing_probe.split == "dev"
    assert cfg.eval.every_steps == 4 and cfg.eval.selection_metric == "dev/rsum"
    assert cfg.analysis.routing_probe is not None and cfg.analysis.routing_probe.every_steps == 2
