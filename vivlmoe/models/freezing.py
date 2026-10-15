"""Freezing presets per stage (paper Sec. 4.1-4.3, 5.4, 6.1) and the startup unused-parameter check."""

from __future__ import annotations

import logging
from dataclasses import fields, is_dataclass

import torch
from torch import nn

from vivlmoe.models.towers.vision import BaseVisionTower

log = logging.getLogger(__name__)

FREEZE_PRESETS: dict[str, list[str]] = {
    "stage1": [],  # student = text_encoder + fc1, both trained (Sec. 4.1)
    "stage2": ["vision_encoder", "vision_proj"],  # ViT and W_v frozen (Sec. 4.2)
    "stage3": [],  # end-to-end (Sec. 4.3)
    "sparse": ["vision_encoder"],  # MoE pretraining keeps the ViT frozen (Sec. 5.4)
    "downstream_seq2seq": ["backbone.vision_encoder", "backbone.logit_scale"],
    # captioning has no text tokens: text_encoder / text_proj would be unused trainable params
    "downstream_caption": [
        "backbone.vision_encoder",
        "backbone.logit_scale",
        "backbone.text_encoder",
        "backbone.text_proj",
    ],
    "downstream_cls": ["backbone.vision_encoder", "backbone.logit_scale"],
}


def _resolve_path(model: nn.Module, path: str) -> nn.Module | nn.Parameter:
    try:
        return model.get_submodule(path)
    except AttributeError:
        pass
    params = dict(model.named_parameters())
    if path in params:
        return params[path]
    raise ValueError(
        f"freeze path {path!r} is neither a submodule nor a parameter of {type(model).__name__}"
    )


def freeze_paths_for_preset(model: nn.Module, preset: str) -> list[str]:
    """Expand a preset into validated dotted paths (modules or scalar parameters) of ``model``."""
    if preset not in FREEZE_PRESETS:
        raise ValueError(f"unknown freeze preset {preset!r}; known: {sorted(FREEZE_PRESETS)}")
    paths = list(FREEZE_PRESETS[preset])
    for p in paths:
        _resolve_path(model, p)
    return paths


def apply_freeze(model: nn.Module, paths: list[str]) -> list[str]:
    """Set ``requires_grad=False`` under every path; vision towers go through ``set_trainable``.

    Returns the names of the frozen parameters.
    """
    frozen: list[str] = []
    for path in paths:
        target = _resolve_path(model, path)
        if isinstance(target, BaseVisionTower):
            target.set_trainable(False)
            frozen.extend(f"{path}.{n}" for n, _ in target.named_parameters())
        elif isinstance(target, nn.Parameter):
            target.requires_grad_(False)
            frozen.append(path)
        else:
            for n, p in target.named_parameters():
                p.requires_grad_(False)
                frozen.append(f"{path}.{n}")
    log.info("froze %d parameters under %s", len(frozen), paths)
    return frozen


def apply_frozen_dtype_policy(
    model: nn.Module, amp_dtype: torch.dtype | None = None
) -> dict[str, torch.dtype]:
    """Apply :meth:`BaseVisionTower.apply_dtype_policy` to every vision tower in ``model``.

    Call once after ``model.to(device)`` and before DDP wrapping / any checkpoint fingerprint: the tower
    never casts inside ``forward``, so this is the single point where a frozen ViT becomes bf16 (CUDA).
    ``amp_dtype`` is the Trainer's autocast dtype (None when AMP is disabled). Returns ``{path: dtype}``.
    """
    result: dict[str, torch.dtype] = {}
    for path, module in model.named_modules():
        if isinstance(module, BaseVisionTower):
            result[path] = module.apply_dtype_policy(amp_dtype)
    log.info("frozen-dtype policy applied: %s", {k: str(v) for k, v in result.items()})
    return result


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _surrogate_scalar(out: object) -> torch.Tensor:
    """Sum of every grad-carrying tensor in a model output (loss if present) so all used params get grads."""
    if isinstance(out, dict):
        values = list(out.values())
    elif is_dataclass(out) and not isinstance(out, type):
        values = [getattr(out, f.name) for f in fields(out)]
    else:
        raise TypeError(f"unsupported model output {type(out).__name__}")
    terms: list[torch.Tensor] = []
    for v in values:
        if isinstance(v, torch.Tensor) and v.requires_grad:
            terms.append(v.float().sum())
        elif isinstance(v, list):  # router outputs
            for ro in v:
                for t in (getattr(ro, "logits", None), getattr(ro, "topk_probs", None)):
                    if isinstance(t, torch.Tensor) and t.requires_grad:
                        terms.append(t.float().sum())
    if not terms:
        raise RuntimeError(
            "model output carries no differentiable tensor; nothing to backpropagate"
        )
    return torch.stack(terms).sum()


def assert_no_unused_trainable_params(model: nn.Module, batch: dict, mode: str) -> None:
    """Run one forward/backward on ``batch`` and raise if any trainable parameter received no gradient.

    Run once at startup on every rank before DDP wrapping (plan decision 19): a trainable parameter
    outside the graph would otherwise hang DDP with ``find_unused_parameters=False``. Gradients are
    cleared afterwards and the training/eval mode restored.
    """
    was_training = model.training
    device = next(model.parameters()).device
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        out = model(_move_batch(batch, device), mode)
        _surrogate_scalar(out).backward()
        unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    if unused:
        raise RuntimeError(
            f"{len(unused)} trainable parameter(s) received no gradient in mode {mode!r}; "
            f"freeze them or fix the forward: {unused[:20]}{' ...' if len(unused) > 20 else ''}"
        )
    log.info("unused-trainable-parameter check passed for mode %r", mode)
