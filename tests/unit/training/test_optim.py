from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn

from vivlmoe.config.schema import ConfigError, OptimCfg, ParamGroupCfg
from vivlmoe.training.optim import (
    apply_freezing,
    build_optimizer,
    describe_param_groups,
    resolve_freeze_paths,
    resolve_param_groups,
    resolve_path,
)


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.text_encoder = nn.Linear(4, 4)
        self.encoder = nn.ModuleDict({"layers": nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])})
        self.logit_scale = nn.Parameter(torch.zeros(()))


def test_resolve_path_typo_lists_children() -> None:
    m = Toy()
    with pytest.raises(ConfigError, match=r"children of '<root>'.*vision_encoder"):
        resolve_path(m, "vision_encodr")
    with pytest.raises(ConfigError, match=r"children of 'encoder.layers'.*'0', '1'"):
        resolve_path(m, "encoder.layers.7")
    assert isinstance(resolve_path(m, "logit_scale"), nn.Parameter)
    assert resolve_path(m, "encoder.layers.1") is m.encoder["layers"][1]


def test_apply_freezing_modules_and_parameters() -> None:
    m = Toy()
    frozen = apply_freezing(m, ["vision_encoder", "logit_scale", "vision_encoder"])
    assert frozen == [
        "vision_encoder.0.weight",
        "vision_encoder.0.bias",
        "vision_encoder.1.weight",
        "vision_encoder.1.bias",
        "logit_scale",
    ]
    assert not m.logit_scale.requires_grad
    assert all(not p.requires_grad for p in m.vision_encoder.parameters())
    assert all(p.requires_grad for p in m.text_encoder.parameters())


def test_default_group_no_decay_split() -> None:
    m = Toy()
    apply_freezing(m, ["vision_encoder"])
    groups = resolve_param_groups(m, OptimCfg(lr=1e-3, weight_decay=0.01))
    by_name = {g.name: g for g in groups}
    assert set(by_name) == {"default/decay", "default/no_decay"}
    assert by_name["default/no_decay"].weight_decay == 0.0
    assert "logit_scale" in by_name["default/no_decay"].param_names
    assert "text_encoder.bias" in by_name["default/no_decay"].param_names
    assert "text_encoder.weight" in by_name["default/decay"].param_names
    assert not any(n.startswith("vision_encoder") for g in groups for n in g.param_names)
    table = describe_param_groups(groups)
    assert "default/decay" in table and "total" in table


def test_explicit_groups_overlap_and_coverage() -> None:
    m = Toy()
    ok = OptimCfg(
        lr=1e-4,
        groups=[
            ParamGroupCfg(modules=["vision_encoder", "text_encoder"], lr=1e-5, name="enc"),
            ParamGroupCfg(modules=["encoder", "logit_scale"], lr=5e-5, weight_decay=0.0),
        ],
    )
    groups = resolve_param_groups(m, ok)
    names = {g.name for g in groups}
    assert names == {"enc/decay", "enc/no_decay", "group1"}  # wd 0 -> no split
    opt = build_optimizer(groups, ok)
    assert sorted({g["lr"] for g in opt.param_groups}) == [1e-5, 5e-5]
    assert {g["name"] for g in opt.param_groups} == names

    overlap = OptimCfg(
        groups=[
            ParamGroupCfg(modules=["encoder"], lr=1.0),
            ParamGroupCfg(modules=["encoder.layers.0"], lr=2.0),
        ],
        strict_group_coverage=False,
    )
    with pytest.raises(ConfigError, match="claimed by both"):
        resolve_param_groups(m, overlap)

    uncovered = OptimCfg(groups=[ParamGroupCfg(modules=["encoder"], lr=1.0)])
    with pytest.raises(ConfigError, match="do not cover"):
        resolve_param_groups(m, uncovered)
    lax = OptimCfg(groups=[ParamGroupCfg(modules=["encoder"], lr=1.0)], strict_group_coverage=False)
    names = {g.name for g in resolve_param_groups(m, lax)}
    assert {"rest/decay", "rest/no_decay"} <= names

    with pytest.raises(ConfigError, match="not found"):
        resolve_param_groups(m, OptimCfg(groups=[ParamGroupCfg(modules=["nope"], lr=1.0)]))


def test_no_trainable_params_is_an_error() -> None:
    m = Toy()
    apply_freezing(m, ["vision_encoder", "text_encoder", "encoder", "logit_scale"])
    with pytest.raises(ConfigError, match="no trainable parameters"):
        resolve_param_groups(m, OptimCfg())


def test_freeze_preset_requires_models_freezing(monkeypatch: pytest.MonkeyPatch) -> None:
    m = Toy()
    monkeypatch.setitem(sys.modules, "vivlmoe.models.freezing", None)  # forces ImportError
    with pytest.raises(ConfigError, match="freeze_preset"):
        resolve_freeze_paths(m, OptimCfg(freeze_preset="stage2"), always_frozen=[])
    fake = types.ModuleType("vivlmoe.models.freezing")
    fake.freeze_paths_for_preset = lambda model, preset: (
        ["vision_encoder"] if preset == "stage2" else []
    )  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vivlmoe.models.freezing", fake)
    paths = resolve_freeze_paths(
        m,
        OptimCfg(freeze_preset="stage2", freeze=["logit_scale", "vision_encoder"]),
        always_frozen=["encoder"],
    )
    assert paths == ["encoder", "vision_encoder", "logit_scale"]
