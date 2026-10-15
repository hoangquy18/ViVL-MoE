#!/usr/bin/env python
"""Sparse-upcycle a dense checkpoint into an MoE checkpoint (paper Sec. 5.3).

python scripts/upcycle.py --config configs/moe/upcycle.yaml --out runs/moe_upcycled/upcycled.pt
python scripts/upcycle.py --config configs/moe/upcycle.yaml --dense runs/stage3/checkpoints/best.pt \\
    --out runs/moe_upcycled_ns0/upcycled.pt model_overrides.moe.num_shared=0

The config's ``model: {from_checkpoint: <dense>}`` + ``model_overrides: {moe: {...}}`` define the MoE
topology (the ``model`` section is inherited from the dense checkpoint's sidecar). The dense
checkpoint defaults to ``model.from_checkpoint``; the output path is a CLI flag because the config
schema has no ``upcycle`` section. The noise-0 equivalence check must pass before anything is saved.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from vivlmoe.config.loader import (  # noqa: E402
    config_from_dict,
    deep_merge,  # noqa: E402
    from_dict,
    load_yaml_with_includes,
    parse_overrides,
    to_dict,
)
from vivlmoe.config.schema import CheckpointCfg, ConfigError, ModelCfg  # noqa: E402
from vivlmoe.models.factory import build_task_model  # noqa: E402
from vivlmoe.training.checkpoint import (  # noqa: E402
    CheckpointManager,
    ProvenanceTracker,
    build_checkpoint,
    load_checkpoint_file,
    load_model_state,
)
from vivlmoe.training.distributed import DistContext  # noqa: E402
from vivlmoe.upcycling.upcycle import (  # noqa: E402
    upcycle_info,
    upcycle_state_dict,
    verify_equivalence,
)
from vivlmoe.utils.misc import count_parameters  # noqa: E402

log = logging.getLogger("scripts.upcycle")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--config", required=True, help="YAML with model.from_checkpoint + model_overrides.moe"
    )
    p.add_argument(
        "--dense", default=None, help="dense checkpoint (default: model.from_checkpoint)"
    )
    p.add_argument("--out", required=True, help="output checkpoint path (.pt)")
    p.add_argument(
        "--noise-std-rel", type=float, default=0.01, help="routed-expert noise, relative to std(W)"
    )
    p.add_argument("--noise-on-bias", action="store_true", help="also perturb expert biases")
    p.add_argument("--seed", type=int, default=0, help="upcycling seed (crc32-derived per tensor)")
    p.add_argument("--verify-atol", type=float, default=1e-5, help="noise-0 equivalence tolerance")
    p.add_argument("--verify-batch-size", type=int, default=2)
    p.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="pixel size of the check batch (default: tower)",
    )
    p.add_argument(
        "overrides",
        nargs="*",
        help="key.path=value overrides (e.g. model_overrides.moe.num_routed=4)",
    )
    return p.parse_args(argv)


def _synthetic_batch(model, batch_size: int, image_size: int | None, seed: int = 0) -> dict:
    """Random pixels / token ids: the equivalence identity holds for any input."""
    g = torch.Generator().manual_seed(seed)
    size = image_size or int(model.vision_encoder.image_size)
    vocab = int(model.text_encoder.model.config.vocab_size)
    length = 12
    ids = torch.randint(3, vocab, (batch_size, length), generator=g)
    mask = torch.ones(batch_size, length, dtype=torch.long)
    if batch_size > 1:
        mask[-1, length // 2 :] = 0
        ids[-1, length // 2 :] = int(
            getattr(model.text_encoder.model.config, "pad_token_id", 1) or 1
        )
    return {
        "pixel_values": torch.randn(batch_size, 3, size, size, generator=g),
        "input_ids": ids,
        "attention_mask": mask,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    raw = deep_merge(load_yaml_with_includes(args.config), parse_overrides(args.overrides))
    if args.dense is not None:
        raw = deep_merge(raw, {"model": {"from_checkpoint": args.dense}})
    cfg = config_from_dict(raw, validate=False)  # model section inherited from the dense sidecar
    dense_path = cfg.model.from_checkpoint
    if dense_path is None:
        raise ConfigError("set model.from_checkpoint in the config or pass --dense")
    if cfg.model.moe is None:
        raise ConfigError(
            "model_overrides.moe must define the MoE topology (num_shared/num_routed/top_k)"
        )
    moe_cfg = cfg.model.moe
    ckpt = load_checkpoint_file(dense_path)
    dense_model_cfg = from_dict(ModelCfg, ckpt["config"]["model"], "model")
    if dense_model_cfg.moe is not None:
        raise ConfigError(f"{dense_path} is already an MoE checkpoint")
    ctx = DistContext.single_process("cpu")
    torch.manual_seed(args.seed)
    dense = build_task_model("backbone", dense_model_cfg)
    tracker = ProvenanceTracker.from_config(dense_model_cfg, dense)
    report = load_model_state(dense, ckpt, tracker.as_dict())
    tracker.inherit(report.provenance)
    dense.eval()
    log.info(
        "loaded dense checkpoint %s (step %s, provenance %s)",
        dense_path,
        ckpt.get("step"),
        tracker.as_dict(),
    )

    moe = build_task_model("backbone", cfg.model)
    dense_sd = dense.state_dict()
    moe.load_state_dict(
        upcycle_state_dict(dense_sd, moe_cfg, noise_std_rel=0.0, seed=args.seed), strict=True
    )
    moe.eval()
    if moe_cfg.num_shared >= 1:
        batch = _synthetic_batch(dense, args.verify_batch_size, args.image_size)
        verify_equivalence(dense, moe, batch, atol=args.verify_atol)  # raises -> nothing is saved
    else:
        log.warning(
            "num_shared=0: the MoE output is half the dense FFN path, equivalence check skipped"
        )
    sd = upcycle_state_dict(
        dense_sd,
        moe_cfg,
        noise_std_rel=args.noise_std_rel,
        noise_on_bias=args.noise_on_bias,
        seed=args.seed,
    )
    moe.load_state_dict(sd, strict=True)
    n_dense, n_moe = count_parameters(dense), count_parameters(moe)
    log.info(
        "parameters: dense %s -> MoE %s (+%.1f%%)",
        f"{n_dense:,}",
        f"{n_moe:,}",
        100.0 * (n_moe / n_dense - 1),
    )

    info = upcycle_info(moe_cfg, args.noise_std_rel, args.noise_on_bias, args.seed, str(dense_path))
    info["dense_step"] = ckpt.get("step")
    info["param_count_dense"], info["param_count_moe"] = n_dense, n_moe
    out = build_checkpoint(
        model=moe,
        excluded=[],
        provenance=tracker.as_dict(),
        config=to_dict(cfg),
        train_state={"step": 0, "epoch": 0},
        task_state={},
        ctx=ctx,
        metrics=dict(ckpt.get("metrics", {})),
        extras={
            "upcycle_info": info,
            "dev_split_sha1": ckpt.get("dev_split_sha1"),
            "transform_spec": ckpt.get("transform_spec"),
        },
    )
    out_path = Path(args.out)
    CheckpointManager(out_path.parent, CheckpointCfg(), ctx).save(out_path, out)
    log.info("wrote %s (+ sidecar)", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
