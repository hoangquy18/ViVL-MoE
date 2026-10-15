"""Sparse upcycling of a dense ``ViVLBackbone`` (paper Sec. 5.3; plan decision 21).

Every dense FFN becomes ``N_s`` exact shared copies plus ``N_r`` routed copies with small noise
``N(0, (noise_std_rel * std(W))^2)`` on the weights (biases untouched unless ``noise_on_bias``), a
random router ``W_r ~ N(0, router_init_std^2)`` and ``expert_bias = 0``. Every other tensor is copied
verbatim. Noise seeds are integers derived with ``zlib.crc32(f"{seed}/{layer}/{expert}/{tensor}")``
(never Python ``hash()``), so the result is reproducible across processes and ``PYTHONHASHSEED``.
With ``noise_std_rel = 0`` and ``N_s >= 1`` the MoE encoder reproduces the dense one in eval mode
(Eq. 18: ``1/2 (FFN + sum p~ FFN) = FFN``); with ``N_s = 0`` the FFN contribution is halved.
"""

from __future__ import annotations

import dataclasses
import logging
import zlib
from typing import Any

import torch
from torch import nn

from vivlmoe.config.schema import MoECfg
from vivlmoe.models.backbone import ViVLBackbone
from vivlmoe.models.outputs import TEXT, VISUAL
from vivlmoe.upcycling.key_mapping import (
    dense_ffn_layers,
    expert_bias_key,
    moe_ffn_keys,
    parse_dense_ffn_key,
    routed_expert_key,
    router_proj_key,
    shared_expert_key,
)

log = logging.getLogger(__name__)


class UpcycleError(RuntimeError):
    """Malformed dense state dict or failed equivalence check."""


def noise_seed(seed: int, layer: int, expert: int | str, tensor: str) -> int:
    """Deterministic 32-bit seed for one noise draw (plan decision 21)."""
    return zlib.crc32(f"{seed}/{layer}/{expert}/{tensor}".encode())


def _seeded_randn(shape: tuple[int, ...], seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32).to(device)


def upcycle_state_dict(
    dense_sd: dict[str, torch.Tensor],
    moe_cfg: MoECfg,
    noise_std_rel: float = 0.01,
    noise_on_bias: bool = False,
    seed: int = 0,
    router_init_std: float | None = None,
) -> dict[str, torch.Tensor]:
    """Dense backbone state dict -> MoE backbone state dict (new tensors; input untouched)."""
    if noise_std_rel < 0:
        raise ValueError("noise_std_rel must be >= 0")
    router_std = moe_cfg.router_init_std if router_init_std is None else router_init_std
    layers = dense_ffn_layers(list(dense_sd))
    if not layers:
        raise UpcycleError(
            "no dense FFN keys (encoder.layers.{i}.ffn.{fc1,fc2}.*) in the state dict: "
            "is this already an MoE checkpoint or not a ViVLBackbone?"
        )
    clashes = [k for layer in layers for k in moe_ffn_keys(layer, moe_cfg) if k in dense_sd]
    if clashes:
        raise UpcycleError(f"state dict already contains MoE keys, e.g. {clashes[:3]}")
    out: dict[str, torch.Tensor] = {}
    for key, value in dense_sd.items():
        parsed = parse_dense_ffn_key(key)
        if parsed is None:
            out[key] = value.detach().clone()
            continue
        layer, tensor = parsed.layer, parsed.tensor
        base = value.detach()
        for s in range(moe_cfg.num_shared):
            out[shared_expert_key(layer, s, tensor)] = base.clone()
        add_noise = noise_std_rel > 0 and (parsed.is_weight or noise_on_bias)
        std = float(base.float().std()) if add_noise else 0.0
        for e in range(moe_cfg.num_routed):
            copy = base.clone()
            if add_noise:
                noise = _seeded_randn(
                    tuple(base.shape), noise_seed(seed, layer, e, tensor), base.device
                )
                copy = (base.float() + noise_std_rel * std * noise).to(base.dtype)
            out[routed_expert_key(layer, e, tensor)] = copy
    dim = int(dense_sd[f"encoder.layers.{layers[0]}.ffn.fc1.weight"].shape[1])
    device = dense_sd[f"encoder.layers.{layers[0]}.ffn.fc1.weight"].device
    for layer in layers:
        proj = _seeded_randn(
            (moe_cfg.num_routed, dim), noise_seed(seed, layer, "router", "proj.weight"), device
        )
        out[router_proj_key(layer)] = proj * router_std
        out[expert_bias_key(layer)] = torch.zeros(
            moe_cfg.num_routed, dtype=torch.float32, device=device
        )
    log.info(
        "upcycled %d layers: N_s=%d N_r=%d k=%d noise_std_rel=%g (bias=%s) seed=%d router_std=%g",
        len(layers),
        moe_cfg.num_shared,
        moe_cfg.num_routed,
        moe_cfg.top_k,
        noise_std_rel,
        noise_on_bias,
        seed,
        router_std,
    )
    return out


def upcycle_info(
    moe_cfg: MoECfg,
    noise_std_rel: float,
    noise_on_bias: bool,
    seed: int,
    source: str | None = None,
    router_init_std: float | None = None,
) -> dict[str, Any]:
    """JSON-able provenance record stored in the upcycled checkpoint (``upcycle_info``)."""
    return {
        "source_checkpoint": source,
        "num_shared": moe_cfg.num_shared,
        "num_routed": moe_cfg.num_routed,
        "top_k": moe_cfg.top_k,
        "aux_free_bias": moe_cfg.aux_free_bias,
        "noise_std_rel": noise_std_rel,
        "noise_on_bias": noise_on_bias,
        "seed": seed,
        "router_init_std": moe_cfg.router_init_std if router_init_std is None else router_init_std,
        "seed_scheme": "zlib.crc32('{seed}/{layer}/{expert}/{tensor}')",
    }


def upcycle_model(
    dense: ViVLBackbone,
    moe_cfg: MoECfg,
    noise_std_rel: float = 0.01,
    noise_on_bias: bool = False,
    seed: int = 0,
    router_init_std: float | None = None,
) -> ViVLBackbone:
    """Build the MoE backbone from ``dense.cfg`` with ``moe=moe_cfg`` and load the upcycled weights strictly."""
    if dense.cfg.moe is not None:
        raise UpcycleError("upcycle_model expects a dense backbone (cfg.moe is set)")
    moe_model_cfg = dataclasses.replace(dense.cfg, moe=moe_cfg)
    moe = ViVLBackbone(moe_model_cfg)
    sd = upcycle_state_dict(
        dense.state_dict(), moe_cfg, noise_std_rel, noise_on_bias, seed, router_init_std
    )
    moe.load_state_dict(sd, strict=True)
    moe.to(next(dense.parameters()).device)
    moe.train(dense.training)  # same mode as the source (eval for the equivalence check)
    for name, p in moe.named_parameters():  # keep the freezing state (e.g. frozen ViT)
        p.requires_grad_(_dense_requires_grad(dense, name))
    return moe


def _dense_requires_grad(dense: nn.Module, moe_name: str) -> bool:
    params = dict(dense.named_parameters())
    if moe_name in params:
        return params[moe_name].requires_grad
    if ".ffn.shared_experts." in moe_name or ".ffn.routed_experts." in moe_name:
        prefix, tail = moe_name.split(".ffn.", 1)
        dense_name = f"{prefix}.ffn.{tail.split('.', 2)[2]}"  # drop 'shared_experts.{s}.'
        return params[dense_name].requires_grad
    return True  # router: new, trainable


@torch.no_grad()
def verify_equivalence(
    dense: ViVLBackbone, moe: ViVLBackbone, batch: dict[str, Any], atol: float = 1e-5
) -> dict[str, float]:
    """Compare eval-mode ``contrastive`` embeddings and encoder outputs; raise :class:`UpcycleError` above ``atol``.

    Only meaningful for ``noise_std_rel = 0`` and ``N_s >= 1`` (exact Eq. 18 identity).
    """
    dense_training, moe_training = dense.training, moe.training
    dense.eval()
    moe.eval()
    try:
        device = next(dense.parameters()).device
        px = batch["pixel_values"].to(device)
        ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        img_tokens = dense.encode_image_tokens(px)
        txt_tokens = dense.encode_text_tokens(ids, mask)
        report = {
            "encoder_visual": _max_diff(
                dense.forward_single(img_tokens, None, VISUAL).last_hidden_state,
                moe.forward_single(img_tokens, None, VISUAL).last_hidden_state,
            ),
            "encoder_text": _max_diff(  # padded positions are not routed (zeros): compare valid ones
                dense.forward_single(txt_tokens, mask.bool(), TEXT).last_hidden_state[mask.bool()],
                moe.forward_single(txt_tokens, mask.bool(), TEXT).last_hidden_state[mask.bool()],
            ),
        }
        a = dense({"pixel_values": px, "input_ids": ids, "attention_mask": mask}, "contrastive")
        b = moe({"pixel_values": px, "input_ids": ids, "attention_mask": mask}, "contrastive")
        report["image_embeds"] = _max_diff(a.image_embeds, b.image_embeds)
        report["text_embeds"] = _max_diff(a.text_embeds, b.text_embeds)
    finally:
        dense.train(dense_training)
        moe.train(moe_training)
    report["max_abs_diff"] = max(report.values())
    report["atol"] = atol
    if report["max_abs_diff"] > atol:
        raise UpcycleError(
            f"upcycled model is not equivalent to the dense one (max |diff| {report['max_abs_diff']:.3e} > atol {atol:g}): {report}"
        )
    log.info("equivalence check passed: %s", {k: f"{v:.2e}" for k, v in report.items()})
    return report


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        raise UpcycleError(
            f"shape mismatch in equivalence check: {tuple(a.shape)} vs {tuple(b.shape)}"
        )
    return float((a.float() - b.float()).abs().max()) if a.numel() else 0.0


__all__ = [
    "UpcycleError",
    "noise_seed",
    "upcycle_info",
    "upcycle_model",
    "upcycle_state_dict",
    "verify_equivalence",
]
