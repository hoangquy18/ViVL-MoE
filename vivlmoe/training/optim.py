"""Freezing, parameter groups and the AdamW optimizer (paper Sec. 6.1; plan 6.3, 8.6).

Module paths are dotted attribute paths validated with ``get_submodule`` (a bare parameter such
as ``logit_scale`` is accepted too). Typos raise :class:`ConfigError` listing the children of the
deepest existing parent so the fix is obvious.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from vivlmoe.config.schema import ConfigError, OptimCfg, ParamGroupCfg
from vivlmoe.utils.misc import format_table

log = logging.getLogger(__name__)


@dataclass
class ParamGroup:
    name: str
    lr: float
    weight_decay: float
    params: list[nn.Parameter] = field(default_factory=list)
    param_names: list[str] = field(default_factory=list)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.params)

    def as_optimizer_group(self) -> dict:
        return {
            "params": self.params,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "name": self.name,
        }


# --------------------------------------------------------------------------------------------
# path resolution
# --------------------------------------------------------------------------------------------
def _children_of(module: nn.Module) -> list[str]:
    return sorted([*module._modules.keys(), *module._parameters.keys()])


def resolve_path(model: nn.Module, path: str) -> nn.Module | nn.Parameter:
    """Return the submodule or parameter at ``path`` or raise :class:`ConfigError` with hints."""
    if not path:
        raise ConfigError("empty module path")
    node: nn.Module = model
    parts = path.split(".")
    for i, part in enumerate(parts):
        if part in node._modules and node._modules[part] is not None:
            node = node._modules[part]
            continue
        if part in node._parameters and i == len(parts) - 1 and node._parameters[part] is not None:
            return node._parameters[part]
        parent = ".".join(parts[:i]) or "<root>"
        raise ConfigError(
            f"module path '{path}' not found: '{part}' is not a child of '{parent}'; "
            f"children of '{parent}': {_children_of(node)}"
        )
    return node


def params_under(model: nn.Module, path: str) -> list[tuple[str, nn.Parameter]]:
    target = resolve_path(model, path)
    if isinstance(target, nn.Parameter):
        return [(path, target)]
    return [(f"{path}.{n}" if n else path, p) for n, p in target.named_parameters()]


# --------------------------------------------------------------------------------------------
# freezing
# --------------------------------------------------------------------------------------------
def apply_freezing(model: nn.Module, paths: Sequence[str]) -> list[str]:
    """Set ``requires_grad=False`` under every path; returns the frozen parameter names.

    A module exposing ``set_trainable`` (``VisionTower``) is frozen through it so its own policy
    (eval mode, frozen dtype) is applied consistently with :mod:`vivlmoe.models.freezing`.
    """
    frozen: list[str] = []
    for path in dict.fromkeys(paths):  # de-duplicate, keep order
        target = resolve_path(model, path)
        if isinstance(target, nn.Module) and callable(getattr(target, "set_trainable", None)):
            target.set_trainable(False)
        for name, p in params_under(model, path):
            p.requires_grad_(False)
            frozen.append(name)
    if frozen:
        log.info("froze %d parameter tensors under %s", len(frozen), list(dict.fromkeys(paths)))
    return frozen


def resolve_freeze_paths(
    model: nn.Module, cfg: OptimCfg, always_frozen: Sequence[str]
) -> list[str]:
    """``task.always_frozen() + preset (models/freezing.py) + optim.freeze`` (validated later)."""
    paths = list(always_frozen)
    if cfg.freeze_preset is not None:
        try:
            from vivlmoe.models.freezing import freeze_paths_for_preset
        except ImportError as exc:  # models/freezing.py is milestone M3
            raise ConfigError(
                f"optim.freeze_preset={cfg.freeze_preset!r} requires vivlmoe.models.freezing ({exc})"
            ) from exc
        paths += list(freeze_paths_for_preset(model, cfg.freeze_preset))
    paths += list(cfg.freeze)
    return list(dict.fromkeys(paths))


# --------------------------------------------------------------------------------------------
# parameter groups
# --------------------------------------------------------------------------------------------
def _is_no_decay(name: str, p: nn.Parameter) -> bool:
    """Biases, LayerNorm weights and scalars (``logit_scale``) get no weight decay (legacy)."""
    return p.ndim <= 1 or name.endswith(".bias")


def _split_decay(name: str, group: ParamGroup, no_decay_split: bool) -> list[ParamGroup]:
    if not no_decay_split or group.weight_decay == 0.0:
        return [group] if group.params else []
    decay = ParamGroup(f"{name}/decay", group.lr, group.weight_decay)
    no_decay = ParamGroup(f"{name}/no_decay", group.lr, 0.0)
    for pname, p in zip(group.param_names, group.params, strict=True):
        dst = no_decay if _is_no_decay(pname, p) else decay
        dst.params.append(p)
        dst.param_names.append(pname)
    return [g for g in (decay, no_decay) if g.params]


def resolve_param_groups(model: nn.Module, cfg: OptimCfg) -> list[ParamGroup]:
    """Explicit module-list groups (``optim.groups``) or one group at ``optim.lr``.

    Errors: unknown module path, a trainable parameter claimed by two groups, and (with
    ``strict_group_coverage``) a trainable parameter not covered by any group.
    """
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise ConfigError("no trainable parameters: check optim.freeze / freeze_preset")
    if not cfg.groups:
        base = ParamGroup("default", cfg.lr, cfg.weight_decay)
        for n, p in trainable:
            base.params.append(p)
            base.param_names.append(n)
        return _split_decay("default", base, cfg.no_decay_norm_and_bias)

    owner: dict[str, str] = {}
    groups: list[ParamGroup] = []
    for i, spec in enumerate(cfg.groups):
        spec_name = _group_name(spec, i)
        wd = cfg.weight_decay if spec.weight_decay is None else spec.weight_decay
        group = ParamGroup(spec_name, spec.lr, wd)
        for path in spec.modules:
            members = [(n, p) for n, p in params_under(model, path) if p.requires_grad]
            if not members:
                log.warning("optim.groups[%s]: '%s' has no trainable parameters", spec_name, path)
            for n, p in members:
                if n in owner:
                    raise ConfigError(
                        f"optim.groups: parameter '{n}' is claimed by both '{owner[n]}' and "
                        f"'{spec_name}' (module paths overlap)"
                    )
                owner[n] = spec_name
                group.params.append(p)
                group.param_names.append(n)
        groups.append(group)
    uncovered = [n for n, _ in trainable if n not in owner]
    if uncovered:
        if cfg.strict_group_coverage:
            shown = ", ".join(uncovered[:20]) + (" ..." if len(uncovered) > 20 else "")
            raise ConfigError(
                f"optim.groups do not cover {len(uncovered)} trainable parameter(s): {shown}. "
                "Add them to a group, freeze them, or set optim.strict_group_coverage: false"
            )
        rest = ParamGroup("rest", cfg.lr, cfg.weight_decay)
        for n, p in trainable:
            if n not in owner:
                rest.params.append(p)
                rest.param_names.append(n)
        groups.append(rest)
        log.warning("optim.groups: %d uncovered parameters placed in group 'rest'", len(uncovered))
    out: list[ParamGroup] = []
    for g in groups:
        out += _split_decay(g.name, g, cfg.no_decay_norm_and_bias)
    return out


def _group_name(spec: ParamGroupCfg, index: int) -> str:
    return spec.name or f"group{index}"


def build_optimizer(groups: Sequence[ParamGroup], cfg: OptimCfg) -> torch.optim.Optimizer:
    if cfg.name != "adamw":
        raise ConfigError(f"unsupported optimizer '{cfg.name}'")
    return torch.optim.AdamW(
        [g.as_optimizer_group() for g in groups],
        lr=cfg.lr,
        betas=tuple(cfg.betas),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )


def describe_param_groups(groups: Sequence[ParamGroup]) -> str:
    rows = [
        (g.name, f"{g.lr:.3g}", f"{g.weight_decay:.3g}", len(g.params), f"{g.num_params:,}")
        for g in groups
    ]
    total = sum(g.num_params for g in groups)
    rows.append(("total", "", "", sum(len(g.params) for g in groups), f"{total:,}"))
    return format_table(("group", "lr", "wd", "tensors", "params"), rows)


__all__ = [
    "ParamGroup",
    "apply_freezing",
    "build_optimizer",
    "describe_param_groups",
    "params_under",
    "resolve_freeze_paths",
    "resolve_param_groups",
    "resolve_path",
]
