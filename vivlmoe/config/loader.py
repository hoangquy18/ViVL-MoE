"""YAML -> :class:`RunConfig` with ``_include`` chains, CLI overrides and strict typing.

Rules
-----
* ``_include: [a.yaml, b.yaml]`` (relative to the including file, or to the repo root when the
  path starts with ``configs/``) are deep-merged in order (later wins), then the file's own keys win.
* CLI overrides ``a.b.c=value`` are YAML-parsed and merged last.
* Unknown keys, wrong types and cross-field violations raise :class:`ConfigError` naming the full
  dotted path. ``1e-3``-style strings are accepted for float fields (PyYAML quirk).
* ``model.from_checkpoint: <ckpt>`` replaces the whole ``model`` section with the one stored in the
  checkpoint sidecar (``<ckpt>.meta.json``); ``model_overrides`` are deep-merged on top. Stabilizer
  switches (``loss.router.use_*``, ``model.moe.aux_free_bias``) must then match the checkpoint unless
  ``task.allow_stabilizer_override`` is true (paper ablations remove a stabilizer in both stages).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import sys
import types
import typing
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

import yaml

from vivlmoe.config.schema import ConfigError, RunConfig

_INCLUDE_KEY = "_include"
_MISSING = object()


# --------------------------------------------------------------------------------------------
# YAML + includes + overrides
# --------------------------------------------------------------------------------------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_include(spec: str, including_file: Path) -> Path:
    p = Path(spec)
    if p.is_absolute():
        return p
    if spec.startswith("configs/"):
        return repo_root() / spec
    return (including_file.parent / p).resolve()


def load_yaml_with_includes(path: str | Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    file = Path(path).resolve()
    if file in _stack:
        chain = " -> ".join(str(p) for p in (*_stack, file))
        raise ConfigError(f"_include cycle: {chain}")
    if not file.exists():
        raise ConfigError(f"config file not found: {file}")
    with file.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{file}: top level must be a mapping")
    includes = data.pop(_INCLUDE_KEY, [])
    if isinstance(includes, str):
        includes = [includes]
    merged: dict[str, Any] = {}
    for inc in includes:
        merged = deep_merge(
            merged, load_yaml_with_includes(_resolve_include(str(inc), file), (*_stack, file))
        )
    return deep_merge(merged, data)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; lists and scalars are replaced, ``None`` is a legitimate value."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_overrides(items: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"override '{item}' must look like key.path=value")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key or key.startswith(".") or key.endswith("."):
            raise ConfigError(f"override '{item}': malformed key")
        try:
            value = yaml.safe_load(raw) if raw.strip() != "" else ""
        except yaml.YAMLError as exc:
            raise ConfigError(f"override '{item}': cannot parse value ({exc})") from None
        node = result
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"override '{item}': '{part}' is not a mapping")
        node[parts[-1]] = value
    return result


# --------------------------------------------------------------------------------------------
# dict -> dataclass
# --------------------------------------------------------------------------------------------
def _tname(tp: Any) -> str:
    return getattr(tp, "__name__", str(tp))


def _is_union(tp: Any) -> bool:
    return get_origin(tp) in (Union, types.UnionType)


def coerce(tp: Any, value: Any, path: str) -> Any:
    """Convert ``value`` to annotation ``tp`` or raise :class:`ConfigError` naming ``path``."""
    if tp is Any:
        return value
    origin = get_origin(tp)
    args = get_args(tp)
    if _is_union(tp):
        options = [a for a in args if a is not type(None)]
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"'{path}' must not be null")
        errors: list[str] = []
        for option in options:
            try:
                return coerce(option, value, path)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(
            f"'{path}': {value!r} matches none of [{', '.join(_tname(o) for o in options)}]: {' | '.join(errors)}"
        )
    if origin is Literal:
        if value in args:
            return value
        raise ConfigError(f"'{path}': {value!r} is not one of {list(args)}")
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        return from_dict(tp, value, path)
    if origin in (list, typing.List):  # noqa: UP006
        if not isinstance(value, list):
            raise ConfigError(f"'{path}': expected a list, got {type(value).__name__}")
        (item_tp,) = args or (Any,)
        return [coerce(item_tp, v, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin in (tuple, typing.Tuple):  # noqa: UP006
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"'{path}': expected a list/tuple, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(coerce(args[0], v, f"{path}[{i}]") for i, v in enumerate(value))
        if args and len(args) != len(value):
            raise ConfigError(f"'{path}': expected {len(args)} items, got {len(value)}")
        return tuple(
            coerce(a, v, f"{path}[{i}]") for i, (a, v) in enumerate(zip(args, value, strict=False))
        )
    if origin in (dict, typing.Dict):  # noqa: UP006
        if not isinstance(value, dict):
            raise ConfigError(f"'{path}': expected a mapping, got {type(value).__name__}")
        key_tp, val_tp = args or (Any, Any)
        out = {}
        for k, v in value.items():
            if key_tp is str and not isinstance(k, str):
                raise ConfigError(f"'{path}': mapping keys must be strings, got {k!r}")
            out[k] = coerce(val_tp, v, f"{path}.{k}")
        return out
    if tp is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"'{path}': expected a bool, got {value!r}")
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"'{path}': expected an int, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool):
            raise ConfigError(f"'{path}': expected a float, got {value!r}")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        raise ConfigError(f"'{path}': expected a float, got {value!r}")
    if tp is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"'{path}': expected a string, got {value!r}")
    if tp is dict:
        if isinstance(value, dict):
            return dict(value)
        raise ConfigError(f"'{path}': expected a mapping, got {type(value).__name__}")
    return value


def from_dict(cls: type, data: Any, path: str = "") -> Any:
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"'{path or cls.__name__}': expected a mapping, got {type(data).__name__}"
        )
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        where = path or "<root>"
        raise ConfigError(
            f"Unknown config key(s) {unknown} at '{where}' ({cls.__name__}); allowed: {sorted(fields)}"
        )
    kwargs: dict[str, Any] = {}
    for name, f in fields.items():
        sub = f"{path}.{name}" if path else name
        if name in data:
            kwargs[name] = coerce(hints[name], data[name], sub)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"Missing required config key '{sub}' ({cls.__name__})")
    try:
        return cls(**kwargs)
    except ConfigError as exc:  # __post_init__ validation
        raise ConfigError(f"'{path or cls.__name__}': {exc}") from None


def to_dict(obj: Any) -> Any:
    """Dataclass -> plain JSON/YAML-able structure (tuples become lists)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def config_fingerprint(
    cfg: RunConfig | dict[str, Any], sections: Sequence[str] = ("model",)
) -> str:
    data = to_dict(cfg) if not isinstance(cfg, dict) else cfg
    subset = {s: data.get(s) for s in sections}
    if "model" in subset and isinstance(subset["model"], dict):
        subset["model"] = {k: v for k, v in subset["model"].items() if k != "from_checkpoint"}
    return hashlib.sha256(
        json.dumps(subset, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


# --------------------------------------------------------------------------------------------
# model.from_checkpoint inheritance
# --------------------------------------------------------------------------------------------
_STABILIZER_KEYS = (
    ("loss", "router", "use_z"),
    ("loss", "router", "use_mi"),
    ("loss", "router", "use_lb"),
)


def _get(d: dict[str, Any], keys: Sequence[str], default: Any = _MISSING) -> Any:
    node: Any = d
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def _apply_from_checkpoint(raw: dict[str, Any]) -> dict[str, Any]:
    model = raw.get("model") or {}
    ckpt_path = model.get("from_checkpoint")
    if not ckpt_path:
        return raw
    extra = sorted(k for k in model if k != "from_checkpoint")
    if extra:
        raise ConfigError(
            f"model.from_checkpoint is set; move the other model keys {extra} into 'model_overrides' "
            "(and do not _include a configs/model/*.yaml file)"
        )
    from vivlmoe.utils.ckpt_meta import read_meta

    meta = read_meta(ckpt_path)
    ckpt_cfg = meta.get("config") or {}
    ckpt_model = ckpt_cfg.get("model")
    if not isinstance(ckpt_model, dict):
        raise ConfigError(f"{ckpt_path}: checkpoint metadata has no 'config.model' section")
    inherited = copy.deepcopy(ckpt_model)
    inherited["from_checkpoint"] = ckpt_path
    inherited = deep_merge(inherited, raw.get("model_overrides") or {})
    raw = dict(raw)
    raw["model"] = inherited
    # stabilizer consistency (paper: ablations remove a stabilizer in sparse training AND downstream)
    allow = bool(_get(raw, ("task", "allow_stabilizer_override"), False))
    if not allow and inherited.get("moe") is not None:
        mismatches = []
        for keys in _STABILIZER_KEYS:
            here, there = _get(raw, keys), _get(ckpt_cfg, keys)
            if there is not _MISSING and here is not _MISSING and here != there:
                mismatches.append((".".join(keys), here, there))
        here_bias = _get(inherited, ("moe", "aux_free_bias"))
        there_bias = _get(ckpt_model, ("moe", "aux_free_bias"))
        if there_bias is not _MISSING and here_bias != there_bias:
            mismatches.append(("model.moe.aux_free_bias", here_bias, there_bias))
        if mismatches:
            desc = "; ".join(f"{k}: here={h!r} checkpoint={t!r}" for k, h, t in mismatches)
            raise ConfigError(
                f"stabilizer settings differ from checkpoint {ckpt_path} ({desc}); "
                "set task.allow_stabilizer_override: true to permit this"
            )
    return raw


# --------------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------------
def load_config(path: str | Path, overrides: Sequence[str] = ()) -> RunConfig:
    raw = load_yaml_with_includes(path)
    raw = deep_merge(raw, parse_overrides(overrides))
    raw = _apply_from_checkpoint(raw)
    cfg = from_dict(RunConfig, raw)
    cfg.validate()
    return cfg


def config_from_dict(raw: dict[str, Any], validate: bool = True) -> RunConfig:
    raw = _apply_from_checkpoint(raw)
    cfg = from_dict(RunConfig, raw)
    if validate:
        cfg.validate()
    return cfg


def save_resolved(cfg: RunConfig, path: str | Path, argv: Sequence[str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# resolved config\n# argv: {' '.join(argv if argv is not None else sys.argv)}\n"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.safe_dump(to_dict(cfg), fh, sort_keys=False, allow_unicode=True)
    return path


__all__ = [
    "ConfigError",
    "RunConfig",
    "coerce",
    "config_fingerprint",
    "config_from_dict",
    "deep_merge",
    "from_dict",
    "load_config",
    "load_yaml_with_includes",
    "parse_overrides",
    "save_resolved",
    "to_dict",
]
