"""Checkpoint format, tower provenance and strict loading (plan decision 13).

A checkpoint is a dict ``{format_version, model, excluded_modules, tower_provenance,
model_fingerprint, config, optimizer?, scheduler, scaler, train_state, task_state, rng, metrics,
created, world_size, dev_split_sha1, transform_spec}`` plus a JSON sidecar written by
:mod:`vivlmoe.utils.ckpt_meta`. A tower is dropped from ``model`` only when it is frozen in this run
*and* its provenance is still the HF checkpoint (``hf:<id>``); loading such a checkpoint into a model
whose tower has the same ``hf:`` provenance re-materialises it from the HF weights (logged), anything
else is a hard error. ``--resume`` additionally requires an identical :func:`model_fingerprint`.

Tower keys (provenance, exclusion) are full dotted paths: ``vision_encoder`` on a backbone,
``backbone.vision_encoder`` on a downstream head. Loading a backbone checkpoint into a head model
(``init_from`` handoff) prepends ``backbone.`` to every checkpoint key, see
:func:`checkpoint_key_prefix`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from vivlmoe.config.schema import CheckpointCfg, ConfigError, ModelCfg
from vivlmoe.training.distributed import DistContext, all_gather_object, barrier
from vivlmoe.utils.ckpt_meta import meta_path, write_meta
from vivlmoe.utils.io import atomic_torch_save, file_size, human_bytes

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
TOWER_ATTRS = ("vision_encoder", "text_encoder")


class CheckpointError(RuntimeError):
    """Incompatible or incomplete checkpoint for the requested operation."""


def _under(key: str, prefix: str) -> bool:
    return key == prefix or key.startswith(prefix + ".")


def model_fingerprint(model: nn.Module) -> str:
    """sha1 over ``name|shape`` of the full state dict (architecture, not values or dtypes).

    Dtype is deliberately excluded: a frozen tower may be held in bf16 on CUDA while the same
    architecture is fp32 on CPU / before the dtype policy is applied.
    """
    h = hashlib.sha1()
    for k, v in model.state_dict().items():
        h.update(f"{k}|{tuple(v.shape)}\n".encode())
    return h.hexdigest()[:16]


def find_tower_paths(model: nn.Module) -> dict[str, str]:
    """``{attr: dotted path}`` of the outermost ``vision_encoder`` / ``text_encoder`` in ``model``.

    Name-based (no dependency on :mod:`vivlmoe.models`); a backbone yields ``{'vision_encoder':
    'vision_encoder'}``, a downstream head ``{'vision_encoder': 'backbone.vision_encoder'}``.
    """
    found: dict[str, str] = {}
    for path, _module in model.named_modules():
        attr = path.rsplit(".", 1)[-1]
        if attr in TOWER_ATTRS and attr not in found:
            found[attr] = path
    return found


# --------------------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------------------
class ProvenanceTracker:
    """``{tower_path: 'hf:<id>' | 'trained'}`` for the towers present on the model.

    A tower becomes ``trained`` when it is trainable in this run or was ``trained`` in the
    ``init_from`` checkpoint; randomly initialised towers (``pretrained: false``) are ``trained``
    from the start because they cannot be re-materialised.
    """

    def __init__(self, provenance: dict[str, str] | None = None) -> None:
        self.provenance: dict[str, str] = dict(provenance or {})

    @classmethod
    def from_config(
        cls, model_cfg: ModelCfg, model: nn.Module, tower_paths: dict[str, str] | None = None
    ) -> ProvenanceTracker:
        """Initial provenance keyed by tower path (``tower_paths`` defaults to name-based lookup)."""
        paths = find_tower_paths(model) if tower_paths is None else tower_paths
        prov: dict[str, str] = {}
        for attr, tcfg in (("vision_encoder", model_cfg.vision), ("text_encoder", model_cfg.text)):
            if attr in paths:
                prov[paths[attr]] = f"hf:{tcfg.name}" if tcfg.pretrained else "trained"
        return cls(prov)

    def inherit(self, loaded: dict[str, str]) -> None:
        for tower, value in loaded.items():
            if tower not in self.provenance:
                continue
            if value != self.provenance[tower]:
                log.info(
                    "provenance of %s: %s -> %s (from init_from)",
                    tower,
                    self.provenance[tower],
                    value,
                )
            self.provenance[tower] = value

    def mark_trainable_towers(self, model: nn.Module) -> list[str]:
        marked = []
        for tower in self.provenance:
            module = model.get_submodule(tower)
            if any(p.requires_grad for p in module.parameters()):
                if self.provenance[tower] != "trained":
                    log.info(
                        "provenance of %s: %s -> trained (trainable in this run)",
                        tower,
                        self.provenance[tower],
                    )
                self.provenance[tower] = "trained"
                marked.append(tower)
        return marked

    def excluded_prefixes(self, model: nn.Module, exclude_pristine_frozen: bool) -> list[str]:
        if not exclude_pristine_frozen:
            return []
        out = []
        for tower, value in self.provenance.items():
            module = model.get_submodule(tower)
            frozen = not any(p.requires_grad for p in module.parameters())
            if frozen and value.startswith("hf:"):
                out.append(tower)
        return out

    def as_dict(self) -> dict[str, str]:
        return dict(self.provenance)


# --------------------------------------------------------------------------------------------
# rng
# --------------------------------------------------------------------------------------------
def collect_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


# --------------------------------------------------------------------------------------------
# build / save
# --------------------------------------------------------------------------------------------
def build_checkpoint(
    *,
    model: nn.Module,
    excluded: list[str],
    provenance: dict[str, str],
    config: dict[str, Any],
    train_state: dict[str, Any],
    task_state: dict[str, Any],
    ctx: DistContext,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    metrics: dict[str, float] | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the checkpoint dict on every rank (rng is gathered from all ranks: collective).

    Tower tensors (keys under a ``provenance`` path) are stored in fp32 so a ``trained`` tower held
    in bf16 by the frozen-dtype policy is never rounded on disk.
    """
    towers = list(provenance)
    model_sd: dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if any(_under(k, p) for p in excluded):
            continue
        if v.is_floating_point() and v.dtype != torch.float32 and any(_under(k, t) for t in towers):
            v = v.float()
        model_sd[k] = v
    rng_states = all_gather_object(collect_rng_state(ctx.device), ctx)
    ckpt: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model": model_sd,
        "excluded_modules": list(excluded),
        "tower_provenance": dict(provenance),
        "model_fingerprint": model_fingerprint(model),
        "config": config,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "train_state": dict(train_state),
        "task_state": dict(task_state),
        "rng": {rank: s for rank, s in enumerate(rng_states)},
        "metrics": dict(metrics or {}),
        "created": _now(),
        "world_size": ctx.world_size,
        "step": int(train_state.get("step", 0)),
        "epoch": int(train_state.get("epoch", 0)),
        "dev_split_sha1": None,
        "transform_spec": None,
    }
    ckpt.update(extras or {})
    return ckpt


def _now() -> str:
    from vivlmoe.utils.misc import now_iso

    return now_iso()


class CheckpointManager:
    """Writes ``step_<N>.pt`` (+ ``last.pt`` symlink), ``best.pt`` and prunes to ``keep_last_n``."""

    def __init__(self, ckpt_dir: Path, cfg: CheckpointCfg, ctx: DistContext) -> None:
        self.dir = Path(ckpt_dir)  # created lazily on the first save (a dry run writes nothing)
        self.cfg = cfg
        self.ctx = ctx

    @property
    def last_path(self) -> Path:
        return self.dir / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.dir / "best.pt"

    def step_path(self, step: int) -> Path:
        return self.dir / f"step_{step:07d}.pt"

    def step_files(self) -> list[Path]:
        return sorted(self.dir.glob("step_*.pt"))

    def save(self, path: Path, ckpt: dict[str, Any]) -> Path:
        """Rank 0 writes atomically (+ sidecar); every rank waits on the barrier."""
        if self.ctx.is_main:
            self.dir.mkdir(parents=True, exist_ok=True)
            atomic_torch_save(ckpt, path)
            write_meta(path, ckpt)
            log.info(
                "saved %s (%s, step %d)", path.name, human_bytes(file_size(path)), ckpt["step"]
            )
        barrier(self.ctx)
        return path

    def save_step(self, ckpt: dict[str, Any], step: int) -> Path:
        path = self.save(self.step_path(step), ckpt)
        if self.ctx.is_main:
            _relink(self.last_path, path)
            _relink(meta_path(self.last_path), meta_path(path))
            self.prune()
        barrier(self.ctx)
        return path

    def save_best(self, ckpt: dict[str, Any]) -> Path:
        return self.save(self.best_path, ckpt)

    def prune(self) -> list[Path]:
        removed = []
        files = self.step_files()
        for old in files[: max(len(files) - self.cfg.keep_last_n, 0)]:
            old.unlink()
            meta_path(old).unlink(missing_ok=True)
            removed.append(old)
        if removed:
            log.info("pruned %d old checkpoint(s): %s", len(removed), [p.name for p in removed])
        return removed

    def resolve_resume(self, resume: str | None) -> Path | None:
        """``'auto'`` -> ``last.pt`` when present (else None); a path -> that path (must exist).

        A dangling ``last.pt`` symlink (target pruned or deleted) is an error rather than a silent
        restart from scratch.
        """
        if resume is None:
            return None
        if resume == "auto":
            last = self.last_path
            if last.is_symlink() and not last.exists():
                raise CheckpointError(
                    f"--resume auto: {last} -> {os.readlink(last)} is a dangling symlink; "
                    "point --resume at an existing step file or remove the link"
                )
            return last if last.exists() else None
        path = Path(resume)
        if not path.exists():
            raise CheckpointError(f"--resume {path}: file not found")
        return path


def _relink(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target.name)


# --------------------------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------------------------
@dataclass
class LoadReport:
    loaded_keys: list[str]
    rematerialised: list[str] = field(default_factory=list)  # excluded prefixes accepted
    provenance: dict[str, str] = field(default_factory=dict)  # provenance stored in the checkpoint
    key_prefix: str = ""  # '' or 'backbone.' (checkpoint keys were remapped into the model)


def load_checkpoint_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or ckpt.get("format_version") != FORMAT_VERSION:
        raise CheckpointError(
            f"{path}: not a ViVL-MoE checkpoint (format_version={ckpt.get('format_version') if isinstance(ckpt, dict) else '?'})"
        )
    return ckpt


def checkpoint_key_prefix(model: nn.Module, ckpt: dict[str, Any]) -> str:
    """``'backbone.'`` when a backbone checkpoint is loaded into a head model, else ``''``.

    Auto-detected: the model has a ``backbone`` submodule, the checkpoint has no ``backbone.*``
    keys, and every top-level checkpoint key is a child of ``model.backbone`` (and of nothing else).
    """
    backbone = getattr(model, "backbone", None)
    if not isinstance(backbone, nn.Module):
        return ""
    top = {k.split(".")[0] for k in ckpt["model"]}
    top |= {p.split(".")[0] for p in ckpt.get("excluded_modules", [])}
    if not top or "backbone" in top:
        return ""
    model_top = {k.split(".")[0] for k in model.state_dict()}
    backbone_top = {k.split(".")[0] for k in backbone.state_dict()}
    if top <= backbone_top and not (top & model_top):
        return "backbone."
    return ""


def _remapped(ckpt: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Shallow copy of ``ckpt`` with model keys, excluded modules and provenance under ``prefix``."""
    if not prefix:
        return ckpt
    out = dict(ckpt)
    out["model"] = {prefix + k: v for k, v in ckpt["model"].items()}
    out["excluded_modules"] = [prefix + p for p in ckpt.get("excluded_modules", [])]
    out["tower_provenance"] = {prefix + t: v for t, v in ckpt.get("tower_provenance", {}).items()}
    return out


def _resolve_namespace(model: nn.Module, ckpt: dict[str, Any]) -> tuple[dict[str, Any], str]:
    prefix = checkpoint_key_prefix(model, ckpt)
    if prefix:
        log.info(
            "checkpoint keys are remapped under '%s' (backbone checkpoint -> %s)",
            prefix,
            type(model).__name__,
        )
    return _remapped(ckpt, prefix), prefix


def _tolerate_missing(
    missing: set[str], ckpt: dict[str, Any], current_provenance: dict[str, str]
) -> tuple[set[str], list[str]]:
    tolerated: set[str] = set()
    rematerialised: list[str] = []
    for prefix in ckpt.get("excluded_modules", []):
        keys = {k for k in missing if _under(k, prefix)}
        if not keys:
            continue
        there = ckpt.get("tower_provenance", {}).get(prefix)
        here = current_provenance.get(prefix)
        if there is None or not str(there).startswith("hf:"):
            raise CheckpointError(
                f"'{prefix}' was excluded from the checkpoint but its provenance is {there!r}: not re-materialisable"
            )
        if here != there:
            raise CheckpointError(
                f"'{prefix}' was excluded from the checkpoint as pristine {there!r}; the current model's "
                f"tower is {here!r}, so it cannot be re-materialised from the HF weights"
            )
        log.info(
            "re-materialising '%s' from %s (excluded from checkpoint as pristine)", prefix, there
        )
        tolerated |= keys
        rematerialised.append(prefix)
    return tolerated, rematerialised


def _check_diff(missing: set[str], unexpected: set[str], mismatched: list[str], what: str) -> None:
    if not (missing or unexpected or mismatched):
        return

    def fmt(keys: list[str]) -> str:
        return ", ".join(keys[:15]) + (" ..." if len(keys) > 15 else "")

    parts = []
    if missing:
        parts.append(f"{len(missing)} missing in checkpoint: {fmt(sorted(missing))}")
    if unexpected:
        parts.append(f"{len(unexpected)} unexpected in checkpoint: {fmt(sorted(unexpected))}")
    if mismatched:
        parts.append(f"{len(mismatched)} shape mismatch: {fmt(mismatched)}")
    raise CheckpointError(f"strict {what} failed: " + "; ".join(parts))


def _load_subset(
    model: nn.Module, sd: dict[str, torch.Tensor], keys: set[str], model_sd: dict[str, torch.Tensor]
) -> None:
    mismatched = [
        f"{k} {tuple(sd[k].shape)}!={tuple(model_sd[k].shape)}"
        for k in sorted(keys)
        if sd[k].shape != model_sd[k].shape
    ]
    _check_diff(set(), set(), mismatched, "load")
    result = model.load_state_dict({k: sd[k] for k in keys}, strict=False)
    if result.unexpected_keys:  # cannot happen after the diff above, but never load silently
        raise CheckpointError(f"unexpected keys: {result.unexpected_keys[:10]}")


def load_model_state(
    model: nn.Module, ckpt: dict[str, Any], current_provenance: dict[str, str]
) -> LoadReport:
    """Strict manual diff: every model key must be in the checkpoint unless re-materialisable.

    When a backbone checkpoint is loaded into a head model (``key_prefix == 'backbone.'``) the
    diff covers the backbone only; the head's own modules (decoder, bridge, cls_head) keep their
    initial weights (logged).
    """
    ckpt, key_prefix = _resolve_namespace(model, ckpt)
    sd, model_sd = ckpt["model"], model.state_dict()
    if key_prefix:
        outside = sorted({k.split(".")[0] for k in model_sd if not k.startswith(key_prefix)})
        model_sd = {k: v for k, v in model_sd.items() if k.startswith(key_prefix)}
        log.info(
            "load all: backbone checkpoint covers '%s*' only; %s keep their init",
            key_prefix,
            outside,
        )
    missing, unexpected = set(model_sd) - set(sd), set(sd) - set(model_sd)
    tolerated, remat = _tolerate_missing(missing, ckpt, current_provenance)
    _check_diff(missing - tolerated, unexpected, [], "model load")
    _load_subset(model, sd, set(sd), model_sd)
    return LoadReport(sorted(sd), remat, dict(ckpt.get("tower_provenance", {})), key_prefix)


def _remap_prefix(prefix: str, key_prefix: str, model_sd: dict[str, torch.Tensor]) -> str:
    """``init_from.load`` prefixes name checkpoint modules; map them into the model namespace."""
    if key_prefix and not any(_under(k, prefix) for k in model_sd):
        candidate = key_prefix + prefix
        if any(_under(k, candidate) for k in model_sd):
            return candidate
    return prefix


def load_partial(
    model: nn.Module,
    source: str | Path | dict[str, Any],
    prefixes: list[str],
    current_provenance: dict[str, str] | None = None,
) -> LoadReport:
    """Strictly load the given module prefixes (or ``['all']``) from a checkpoint (``init_from``).

    Prefixes may be given in the checkpoint's namespace (``encoder``) or the model's
    (``backbone.encoder``) when a backbone checkpoint is loaded into a head.
    """
    ckpt = load_checkpoint_file(source) if not isinstance(source, dict) else source
    current = current_provenance or {}
    if "all" in prefixes:
        if len(prefixes) != 1:
            raise ConfigError("init_from.load: 'all' cannot be combined with other prefixes")
        return load_model_state(model, ckpt, current)
    ckpt, key_prefix = _resolve_namespace(model, ckpt)
    sd, model_sd = ckpt["model"], model.state_dict()
    keys: set[str] = set()
    remat: list[str] = []
    resolved: list[str] = []
    for given in prefixes:
        prefix = _remap_prefix(given, key_prefix, model_sd)
        resolved.append(prefix)
        model_keys = {k for k in model_sd if _under(k, prefix)}
        if not model_keys:
            top = sorted({k.split(".")[0] for k in model_sd})
            raise ConfigError(
                f"init_from.load: prefix '{given}' matches nothing in the model; top-level: {top}"
            )
        ckpt_keys = {k for k in sd if _under(k, prefix)}
        tolerated, r = _tolerate_missing(model_keys - ckpt_keys, ckpt, current)
        remat += r
        _check_diff(
            model_keys - ckpt_keys - tolerated,
            ckpt_keys - model_keys,
            [],
            f"partial load of '{prefix}'",
        )
        keys |= ckpt_keys
    _load_subset(model, sd, keys, model_sd)
    prov = {
        t: v
        for t, v in ckpt.get("tower_provenance", {}).items()
        if any(_under(t, p) or _under(p, t) for p in resolved)
    }
    log.info("init_from: loaded %d tensors under %s", len(keys), resolved)
    return LoadReport(sorted(keys), remat, prov, key_prefix)


def load_for_resume(
    model: nn.Module,
    path: str | Path,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    train_state: Any,
    task: Any,
    ctx: DistContext,
    current_provenance: dict[str, str],
) -> dict[str, Any]:
    """Restore everything for ``--resume`` within the same run (fingerprint must match)."""
    ckpt = load_checkpoint_file(path)
    fp = model_fingerprint(model)
    if ckpt["model_fingerprint"] != fp:
        raise CheckpointError(
            f"{path}: model fingerprint {ckpt['model_fingerprint']} != current {fp}; --resume requires the "
            "identical architecture (use init_from for stage handoffs)"
        )
    load_model_state(model, ckpt, current_provenance)
    if ckpt.get("optimizer") is None:
        raise CheckpointError(
            f"{path} has no optimizer state (best.pt or save_optimizer=never): cannot resume"
        )
    optimizer.load_state_dict(ckpt["optimizer"])
    if ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if ckpt.get("scaler") is not None and scaler is not None:
        scaler.load_state_dict(ckpt["scaler"])
    train_state.load_state_dict(ckpt["train_state"])
    task.load_state_dict(ckpt.get("task_state", {}))
    if ckpt["world_size"] == ctx.world_size and ctx.rank in ckpt["rng"]:
        restore_rng_state(ckpt["rng"][ctx.rank], ctx.device)
    else:
        log.warning(
            "resume: checkpoint was written by %d rank(s), now %d: RNG streams are NOT restored",
            ckpt["world_size"],
            ctx.world_size,
        )
    log.info("resumed from %s at step %d (epoch %d)", path, ckpt["step"], ckpt["epoch"])
    return ckpt


__all__ = [
    "FORMAT_VERSION",
    "TOWER_ATTRS",
    "CheckpointError",
    "CheckpointManager",
    "LoadReport",
    "ProvenanceTracker",
    "build_checkpoint",
    "checkpoint_key_prefix",
    "collect_rng_state",
    "find_tower_paths",
    "load_checkpoint_file",
    "load_for_resume",
    "load_model_state",
    "load_partial",
    "model_fingerprint",
    "restore_rng_state",
]
