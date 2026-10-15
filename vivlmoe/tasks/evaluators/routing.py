"""Routing probe (paper Sec. 6.5; plan decision 20): a task-agnostic evaluator over a fixed dev stream.

Every ``every_steps`` optimizer steps (up to ``max_steps``) the model runs in ``forward_mode`` over the
split's loader; the per-token top-k sets are gathered in dataset order (:func:`gather_ordered`, on
every rank) into a :class:`RoutingSnapshot`, and the change rate against the previous snapshot (kept
in ``state_dict`` so it survives a resume) plus the modality heatmap / sym-KL, router margins and
expert load are written to ``run_dir/routing/step_<N>.json`` (+ PNG heatmap when matplotlib is
importable). Counters are untouched: the router accumulates only in training mode.

``max_batches`` bounds the probe by a *global* number of samples, ``max_batches * batch_size``
dataset uids taken before sharding, so the "fixed dev tokens" are the same for every world size
(``data.evals.<split>.limit`` bounds the split itself the same way). If a restored snapshot still
covers a different token stream (the split or its limit changed) the change rate restarts from the
current snapshot with a warning instead of aborting the run.

The Trainer runs the probe on its own cadence (``analysis.routing_probe.every_steps``) in addition
to the split's regular evaluation, where :meth:`RoutingProbeEvaluator.run` returns ``{}`` when not due.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from vivlmoe.analysis.routing_stats import (
    ExpertLoadCollector,
    ModalityRoutingCollector,
    RouterMarginCollector,
    RoutingSnapshot,
    expert_change_rate,
)
from vivlmoe.config.schema import ConfigError, RoutingProbeCfg, RunConfig
from vivlmoe.data.loaders import move_to_device
from vivlmoe.models.outputs import TEXT, VISUAL, RouterOutput
from vivlmoe.tasks.evaluators.base import Evaluator, gather_ordered
from vivlmoe.training.distributed import DistContext, all_reduce_sum_
from vivlmoe.training.state import EvalContext
from vivlmoe.utils.io import write_json

log = logging.getLogger(__name__)

FORWARD_MODES = ("contrastive", "joint_caption", "joint_vqa", "joint_cls")
Side = tuple[str, list[RouterOutput], torch.Tensor]  # (side name, per-layer outputs, tokens/sample)


def encoder_of(model: nn.Module) -> nn.Module:
    """``model.encoder`` for a backbone, ``model.backbone.encoder`` for downstream heads."""
    return getattr(model, "backbone", model).encoder


def backbone_of(model: nn.Module) -> nn.Module:
    return getattr(model, "backbone", model)


def _sub_loaders(loader: Any) -> list[tuple[str, Any]]:
    """Accept a plain DataLoader, ``(image_loader, text_loader)``, a dict, or an object with
    ``images`` / ``texts`` (or ``image_loader`` / ``text_loader``) attributes."""
    if isinstance(loader, DataLoader):
        return [("batches", loader)]
    if isinstance(loader, dict):
        return [(str(k), v) for k, v in loader.items()]
    if isinstance(loader, (tuple, list)):
        return [(f"loader{i}", ld) for i, ld in enumerate(loader)]
    for img, txt in (("images", "texts"), ("image_loader", "text_loader"), ("image", "text")):
        if hasattr(loader, img) and hasattr(loader, txt):
            return [("images", getattr(loader, img)), ("texts", getattr(loader, txt))]
    if hasattr(loader, "__iter__"):
        return [("batches", loader)]
    raise TypeError(f"routing probe cannot iterate a loader of type {type(loader).__name__}")


class RoutingProbeEvaluator(Evaluator):
    def __init__(self, name: str, cfg: RoutingProbeCfg, forward_mode: str, num_layers: int) -> None:
        if forward_mode not in FORWARD_MODES:
            raise ValueError(f"forward_mode must be one of {FORWARD_MODES}, got {forward_mode!r}")
        self.name = name
        self.cfg = cfg
        self.forward_mode = forward_mode
        self.num_layers = num_layers
        self.previous: RoutingSnapshot | None = None
        self.history: list[int] = []

    # ------------------------------------------------------------------ scheduling
    def is_due(self, step: int) -> bool:
        if self.cfg.max_steps is not None and step > self.cfg.max_steps:
            return False
        return step % self.cfg.every_steps == 0

    # ------------------------------------------------------------------ forward passes
    def _sides(self, model: nn.Module, batch: dict[str, Any]) -> list[Side]:
        """Run ``forward_mode`` on one batch -> sides of ``(name, router outputs, tokens per sample)``."""
        bb = backbone_of(model)
        n_lat = int(bb.num_visual_tokens)
        b = len(batch["uids"])
        if self.forward_mode == "contrastive":
            sides: list[Side] = []
            if "pixel_values" in batch:
                enc = bb.forward_single(bb.encode_image_tokens(batch["pixel_values"]), None, VISUAL)
                sides.append(("visual", enc.router_outputs, torch.full((b,), n_lat)))
            if "input_ids" in batch:
                mask = batch["attention_mask"]
                enc = bb.forward_single(
                    bb.encode_text_tokens(batch["input_ids"], mask), mask.bool(), TEXT
                )
                sides.append(("text", enc.router_outputs, mask.bool().sum(dim=1).cpu()))
            if not sides:
                raise KeyError("contrastive probe batch has neither pixel_values nor input_ids")
            return sides
        out = model(batch, self.forward_mode)
        ros = out.router_outputs
        if self.forward_mode == "joint_caption":
            counts = torch.full((b,), n_lat)
        elif self.forward_mode == "joint_vqa":
            counts = n_lat + batch["attention_mask"].bool().sum(dim=1).cpu()
        else:  # joint_cls: A sequences per sample, each [valid images x n_lat ; aspect text]
            img = batch["image_mask"].bool().sum(dim=1).cpu() * n_lat  # [B]
            a = batch["input_ids"].shape[1]
            txt = batch["attention_mask"].bool().sum(dim=(1, 2)).cpu()  # [B]
            counts = img * a + txt
        return [("joint", ros, counts)]

    @staticmethod
    def _per_sample(
        ros: list[RouterOutput], counts: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Split the flat token stream of one batch into ``(topk [L, n_i, k], modality [n_i])`` per sample."""
        if not ros:
            raise RuntimeError("routing probe: the model returned no RouterOutput (dense model?)")
        by_layer = sorted(ros, key=lambda r: r.layer_idx)
        topk = torch.stack([r.topk_idx.detach().cpu() for r in by_layer])  # [L, T, k]
        mods = by_layer[0].modality_ids.detach().cpu()
        total = int(counts.sum())
        if topk.shape[1] != total:
            raise RuntimeError(
                f"routing probe: {topk.shape[1]} routed tokens but {total} expected from the batch"
            )
        items = []
        start = 0
        for n in counts.tolist():
            n = int(n)
            items.append(
                (topk[:, start : start + n].to(torch.int16), mods[start : start + n].to(torch.int8))
            )
            start += n
        return items

    # ------------------------------------------------------------------ sample budget
    def _uid_budget(self, sub: Any) -> int | None:
        """Global uid bound ``max_batches * batch_size`` (None = unbounded); world-size independent."""
        if self.cfg.max_batches is None:
            return None
        bs = getattr(sub, "batch_size", None)
        if not bs:
            raise ConfigError(
                "analysis.routing_probe.max_batches needs a loader with a fixed batch_size "
                "(use data.evals.<split>.limit for loaders without one)"
            )
        return int(self.cfg.max_batches) * int(bs)

    @staticmethod
    def _select(batch: dict[str, Any], budget: int | None) -> tuple[dict[str, Any] | None, bool]:
        """Keep the samples with ``uid < budget``; ``(batch or None, exhausted)``.

        Eval shards are ordered (``OrderedShardSampler``), so once every uid of a batch is beyond
        the budget the remaining batches of this rank are too.
        """
        uids = [int(u) for u in batch["uids"]]
        if budget is None:
            return batch, False
        keep = [i for i, u in enumerate(uids) if u < budget]
        if not keep:
            return None, True
        if len(keep) == len(uids):
            return batch, False
        idx = torch.tensor(keep)
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == len(uids):
                out[k] = v.index_select(0, idx.to(v.device))
            elif isinstance(v, (list, tuple)) and len(v) == len(uids):
                out[k] = type(v)(v[i] for i in keep)
            else:
                out[k] = v
        return out, True

    # ------------------------------------------------------------------ run
    def run(self, model: nn.Module, loader: DataLoader, ec: EvalContext) -> dict[str, float] | None:
        if not self.is_due(ec.step):
            return {}
        ctx = ec.ctx
        encoder = encoder_of(model)
        if not getattr(encoder, "is_moe", False):
            raise ConfigError("analysis.routing_probe requires an MoE model (model.moe is null)")
        n_r = int(encoder.moe_cfg.num_routed)
        heat = ModalityRoutingCollector(self.num_layers, n_r)
        margin = RouterMarginCollector(self.num_layers)
        load = ExpertLoadCollector(self.num_layers, n_r)
        per_side: dict[str, tuple[list[int], list[Any]]] = {}
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for _, sub in _sub_loaders(loader):
                budget = self._uid_budget(sub)
                for batch in sub:
                    batch, exhausted = self._select(batch, budget)
                    if batch is not None:
                        batch = move_to_device(batch, ctx.device)
                        for side, ros, counts in self._sides(model, batch):
                            for ro in ros:
                                heat.add(ro)
                                margin.add(ro)
                                load.add(ro)
                            ids, payload = per_side.setdefault(side, ([], []))
                            ids.extend(int(u) for u in batch["uids"])
                            payload.extend(self._per_sample(ros, counts))
                    if exhausted:
                        break
        model.train(was_training)
        # collectives: identical sequence on every rank
        for c in (heat, margin, load):
            c.sums = all_reduce_sum_(c.sums.to(ctx.device), ctx).cpu()
        side_names = self._side_names()
        gathered = {side: gather_ordered(*per_side.get(side, ([], [])), ctx) for side in side_names}
        snapshot = self._snapshot(gathered, side_names, ec.step)
        change = self._change_rate(snapshot)
        report = self._report(snapshot, change, heat, margin, load, encoder, ec)
        self.previous = snapshot
        self.history.append(ec.step)
        if ctx.is_main:
            self._write(report, snapshot, ec)
        return self._flatten(report) if ctx.is_main else None

    def _change_rate(self, snapshot: RoutingSnapshot) -> dict[str, Any] | None:
        """Change rate vs the previous snapshot; a different token stream restarts the series."""
        if self.previous is None:
            return None
        try:
            return expert_change_rate(self.previous, snapshot)
        except ValueError as exc:
            log.warning(
                "routing probe: previous snapshot (step %d) does not match the current token stream "
                "(%s); the change-rate series restarts at step %d",
                self.previous.step,
                exc,
                snapshot.step,
            )
            self.previous = None
            return None

    def _side_names(self) -> list[str]:
        return ["visual", "text"] if self.forward_mode == "contrastive" else ["joint"]

    def _snapshot(
        self, gathered: dict[str, list[Any]], sides: list[str], step: int
    ) -> RoutingSnapshot:
        topks, mods = [], []
        for side in sides:
            for t, m in gathered[side]:
                topks.append(t)
                mods.append(m)
        if not topks:
            raise RuntimeError("routing probe saw no tokens (empty loader?)")
        return RoutingSnapshot(step, torch.cat(topks, dim=1), torch.cat(mods))

    def _report(
        self,
        snapshot: RoutingSnapshot,
        change: dict[str, Any] | None,
        heat: ModalityRoutingCollector,
        margin: RouterMarginCollector,
        load: ExpertLoadCollector,
        encoder: nn.Module,
        ec: EvalContext,
    ) -> dict[str, Any]:
        return {
            "step": ec.step,
            "split": ec.split_name,
            "forward_mode": self.forward_mode,
            "num_tokens": snapshot.num_tokens,
            "num_layers": snapshot.num_layers,
            "top_k": int(snapshot.topk.shape[-1]),
            "previous_step": self.previous.step if self.previous is not None else None,
            "change_rate": change,
            "modality": heat.result(),
            "margin": margin.result(),
            "load": load.result(encoder.router_bias_state()),
        }

    def _flatten(self, r: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        if r["change_rate"] is not None:
            cr = r["change_rate"]
            out["routing/change_rate"] = cr["mean"]
            for name, v in cr["per_modality"].items():
                if v is not None:
                    out[f"routing/change_rate_{name}"] = v
            for layer, v in enumerate(cr["per_layer"]):
                out[f"routing/layer{layer}/change_rate"] = v
        mod, mar, load = r["modality"], r["margin"], r["load"]
        for key in ("sym_kl", "sym_kl_layer_mean"):
            if mod[key] is not None:
                out[f"routing/{key}"] = mod[key]
        for key in (
            "margin_full",
            "margin_topk",
            "margin_full_visual",
            "margin_full_text",
            "margin_topk_visual",
            "margin_topk_text",
        ):
            if mar[key] is not None:
                out[f"routing/{key}"] = mar[key]
        for key in ("expert_frac_max", "expert_frac_min", "bias_range"):
            if load.get(key) is not None:
                out[f"routing/{key}"] = load[key]
        for layer in range(r["num_layers"]):
            if mod["sym_kl_per_layer"][layer] is not None:
                out[f"routing/layer{layer}/sym_kl"] = mod["sym_kl_per_layer"][layer]
            out[f"routing/layer{layer}/margin_full"] = mar["margin_full_per_layer"][layer]
            out[f"routing/layer{layer}/margin_topk"] = mar["margin_topk_per_layer"][layer]
        return out

    def _write(self, report: dict[str, Any], snapshot: RoutingSnapshot, ec: EvalContext) -> None:
        out_dir = Path(ec.run_dir) / "routing"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"step_{ec.step:07d}"
        write_json(out_dir / f"{stem}.json", report)
        torch.save(snapshot.state_dict(), out_dir / f"snapshot_{stem}.pt")
        png = save_heatmap_png(
            report["modality"]["heatmap"], out_dir / f"{stem}_heatmap.png", ec.step
        )
        log.info(
            "routing probe @ step %d: %d tokens, change_rate=%s sym_kl=%s margin_full=%s -> %s%s",
            ec.step,
            snapshot.num_tokens,
            None if report["change_rate"] is None else round(report["change_rate"]["mean"], 4),
            None
            if report["modality"]["sym_kl"] is None
            else round(report["modality"]["sym_kl"], 4),
            None
            if report["margin"]["margin_full"] is None
            else round(report["margin"]["margin_full"], 4),
            out_dir / f"{stem}.json",
            f" (+ {png.name})" if png else "",
        )

    # ------------------------------------------------------------------ state
    def state_dict(self) -> dict[str, Any]:
        return {
            "previous": self.previous.state_dict() if self.previous is not None else None,
            "history": list(self.history),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        prev = state.get("previous")
        self.previous = RoutingSnapshot.from_state_dict(prev) if prev is not None else None
        self.history = list(state.get("history", []))


def save_heatmap_png(heatmap: list[list[float]], path: Path, step: int) -> Path | None:
    """Expert x modality routing-share heatmap; returns None when matplotlib is not importable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("matplotlib not importable: skipping %s", path)
        return None
    data = torch.tensor(heatmap, dtype=torch.float32).t()  # [2, N_r]
    fig, ax = plt.subplots(figsize=(max(4, 0.6 * data.shape[1] + 2), 2.6))
    im = ax.imshow(data.numpy(), aspect="auto", cmap="viridis", vmin=0.0)
    ax.set_yticks([0, 1], labels=["visual", "text"])
    ax.set_xticks(range(data.shape[1]), labels=[f"E{i}" for i in range(data.shape[1])])
    ax.set_title(f"routing share per modality (layer-averaged), step {step}")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def maybe_add_routing_probe(
    evaluators: dict[str, Evaluator], cfg: RunConfig, ctx: DistContext, forward_mode: str
) -> None:
    """Wrap ``evaluators[cfg.analysis.routing_probe.split]`` into a composite with the routing probe.

    No-op when no probe is configured or the model is dense. ``ctx`` is accepted for interface
    symmetry (the probe reads the context from the ``EvalContext`` at run time).
    """
    rp = cfg.analysis.routing_probe
    if rp is None or not cfg.model.is_moe:
        return
    if rp.split not in evaluators:
        raise ConfigError(
            f"analysis.routing_probe.split '{rp.split}' has no evaluator (have {sorted(evaluators)})"
        )
    from vivlmoe.tasks.evaluators.composite import CompositeEvaluator

    probe = RoutingProbeEvaluator("routing_probe", rp, forward_mode, cfg.model.encoder.num_layers)
    base = evaluators[rp.split]
    parts = base.evaluators if isinstance(base, CompositeEvaluator) else [base]
    evaluators[rp.split] = CompositeEvaluator([*parts, probe])
    log.info(
        "routing probe on '%s' every %d steps (max_steps=%s, max_batches=%s, mode=%s)",
        rp.split,
        rp.every_steps,
        rp.max_steps,
        rp.max_batches,
        forward_mode,
    )
    warn_probe_cadence(cfg)


def warn_probe_cadence(cfg: RunConfig) -> bool:
    """Warn when the probe cadence is not a multiple of ``eval.every_steps`` (returns True then).

    The Trainer runs the probe on its own step cadence, so a mismatch is not an error; the warning
    only flags that probe steps and full-eval steps (selection, ``dev/rsum``...) will not coincide.
    """
    rp, ev = cfg.analysis.routing_probe, cfg.eval
    if rp is None:
        return False
    if ev.every_steps is None or rp.every_steps % ev.every_steps != 0:
        log.warning(
            "analysis.routing_probe.every_steps=%d is not a multiple of eval.every_steps=%s: the probe "
            "runs on its own step cadence (Trainer._run_routing_probe), separately from the epoch/step "
            "evaluations",
            rp.every_steps,
            ev.every_steps,
        )
        return True
    return False


__all__ = [
    "FORWARD_MODES",
    "RoutingProbeEvaluator",
    "backbone_of",
    "encoder_of",
    "maybe_add_routing_probe",
    "save_heatmap_png",
    "warn_probe_cadence",
]
