"""The single training loop shared by every task (plan Sec. 9; decisions 2, 15, 19, 22).

Build order: seed -> ``task.build_model()`` on CPU -> freezing (task + preset + explicit) ->
``init_from`` partial load -> ``.to(device)`` -> AMP state -> frozen-dtype policy of the vision
towers (``vivlmoe.models.freezing.apply_frozen_dtype_policy``, so DDP and the checkpoint
fingerprint see the final dtype) -> ``wrap_ddp`` -> per-rank RNG streams -> loaders -> step budget
-> param groups / optimizer / scheduler / writer / checkpoints -> resume.
The unused-trainable-parameter check runs after the first backward (all ranks, hard error).

Per optimizer step: ``accum`` micro-batches (``no_sync`` for all but the last), one scaled
backward each, unscale -> clip -> step -> ``scheduler.step()`` exactly once -> ``state.step += 1``
-> ``task.post_optimizer_step``. Non-finite losses / gradient norms raise under bf16 / fp32; under
fp16 the optimizer step is skipped, the scaler backs off and the run aborts after
``train.max_consecutive_nonfinite`` consecutive non-finite micro-batches *or* consecutive skipped
optimizer steps (the backward still runs so DDP's gradient all-reduce stays consistent across
ranks). ``scheduler.step()`` and ``state.step`` advance on skipped steps too, so the step budget
and lr schedule stay aligned with the checkpoint cadence.

Epoch bookkeeping happens inside ``_after_step``: the step that completes an epoch rolls
``state.epoch`` and runs the epoch-end evaluation *before* any checkpoint of that step is written,
so a checkpoint at an epoch boundary resumes into a fresh epoch without repeating the eval.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vivlmoe.config.loader import to_dict
from vivlmoe.config.schema import ConfigError, RunConfig
from vivlmoe.data.loaders import move_to_device
from vivlmoe.tasks.base import LoaderBundle, Task
from vivlmoe.training.amp import AmpState, build_amp
from vivlmoe.training.checkpoint import (
    CheckpointManager,
    ProvenanceTracker,
    build_checkpoint,
    load_checkpoint_file,
    load_for_resume,
    load_model_state,
    load_partial,
)
from vivlmoe.training.ddp import (
    check_all_trainable_params_received_grad,
    sync_context,
    unwrap,
    wrap_ddp,
)
from vivlmoe.training.distributed import (
    DistContext,
    all_reduce_mean,
    broadcast_object,
    seed_everything,
    seed_rank_streams,
)
from vivlmoe.training.early_stopping import EarlyStopping
from vivlmoe.training.logging import MetricsWriter, NullWriter, build_metrics_writer
from vivlmoe.training.optim import (
    ParamGroup,
    apply_freezing,
    build_optimizer,
    describe_param_groups,
    resolve_freeze_paths,
    resolve_param_groups,
)
from vivlmoe.training.schedulers import build_scheduler, resolve_warmup_steps
from vivlmoe.training.state import EvalContext, StepContext, TrainState
from vivlmoe.training.throughput import ThroughputMeter, peak_memory_gb, reset_peak_memory
from vivlmoe.utils.io import append_jsonl
from vivlmoe.utils.misc import count_parameters

log = logging.getLogger(__name__)


@dataclass
class StepResult:
    loss: torch.Tensor  # mean over micro-batches (local rank, detached)
    samples: int
    tokens: int
    grad_norm: float
    skipped: bool
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)


class Trainer:
    def __init__(
        self,
        cfg: RunConfig,
        task: Task,
        ctx: DistContext,
        run_dir: Path,
        resume: str | None = None,
        dry_run: bool = False,
    ) -> None:
        self.cfg, self.task, self.ctx = cfg, task, ctx
        self.run_dir = Path(run_dir)
        self.dry_run = dry_run
        self.device = ctx.device
        self.state = TrainState()
        self._configure_backend()
        self._check_amp_vs_frozen_dtype()

        # ---- model ----------------------------------------------------------------------
        seed_everything(cfg.run.seed)
        raw = task.build_model()
        self.provenance = ProvenanceTracker.from_config(cfg.model, raw, _tower_paths(raw))
        self.frozen_paths = resolve_freeze_paths(raw, cfg.optim, task.always_frozen())
        apply_freezing(raw, self.frozen_paths)
        if cfg.init_from is not None:
            report = load_partial(
                raw, cfg.init_from.path, list(cfg.init_from.load), self.provenance.as_dict()
            )
            self.provenance.inherit(report.provenance)
        self.provenance.mark_trainable_towers(raw)
        raw.to(self.device)
        self.amp: AmpState = build_amp(cfg.train.amp, self.device)
        self.tower_dtypes = _apply_frozen_dtype_policy(raw, self.amp.dtype)
        self._set_train_mode(raw)
        self.model = wrap_ddp(raw, ctx, cfg.dist)
        seed_rank_streams(cfg.run.seed, ctx)

        # ---- data + budget --------------------------------------------------------------
        self.loaders: LoaderBundle = task.build_loaders()
        if self.loaders.train is None:
            raise ConfigError("task.build_loaders() returned no train loader")
        self.train_loader = self.loaders.train
        self.sampler = self._sampler_of(self.train_loader)
        self.accum = cfg.train.grad_accum_steps
        self.steps_per_epoch = len(self.train_loader) // self.accum
        if self.steps_per_epoch < 1:
            raise ConfigError(
                f"train loader yields {len(self.train_loader)} micro-batches per epoch, fewer than "
                f"grad_accum_steps={self.accum}"
            )
        if cfg.train.max_steps is not None:
            self.total_steps = cfg.train.max_steps
        else:
            self.total_steps = int(cfg.train.epochs or 0) * self.steps_per_epoch
        if dry_run:
            self.total_steps = min(2, self.total_steps)
        self.evaluators = task.evaluators()
        self._check_eval_wiring()

        # ---- optimisation ---------------------------------------------------------------
        self.param_groups: list[ParamGroup] = resolve_param_groups(raw, cfg.optim)
        self.trainable_params = [p for g in self.param_groups for p in g.params]
        self.optimizer = build_optimizer(self.param_groups, cfg.optim)
        self.scheduler = build_scheduler(self.optimizer, cfg.schedule, self.total_steps)
        self.writer: MetricsWriter = (
            NullWriter()
            if dry_run
            else build_metrics_writer(cfg.logging, ctx, self.run_dir, cfg.run.name, to_dict(cfg))
        )
        self.ckpt = CheckpointManager(self.run_dir / "checkpoints", cfg.checkpoint, ctx)
        self.selection_metric, self.selection_mode = self._resolve_selection()
        self.early = EarlyStopping(cfg.train.early_stopping, self.selection_mode)
        self.excluded = self.provenance.excluded_prefixes(
            raw, cfg.checkpoint.exclude_pristine_frozen_towers
        )
        self.meter = ThroughputMeter()
        self.last_eval: dict[str, float] = {}
        self._first_backward_done = False
        self._consecutive_nonfinite = 0  # micro-batches with a non-finite loss (fp16)
        self._consecutive_skipped = 0  # optimizer steps skipped on a non-finite grad norm (fp16)
        self._stop_requested = False
        self._log_sums: dict[str, torch.Tensor] = {}
        self._log_count = 0
        self._resumed = False

        # ---- resume ---------------------------------------------------------------------
        path = None if dry_run else self.ckpt.resolve_resume(resume)
        if path is not None:
            ckpt = load_for_resume(
                raw,
                path,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.amp.scaler,
                train_state=self.state,
                task=task,
                ctx=ctx,
                current_provenance=self.provenance.as_dict(),
            )
            self.provenance = ProvenanceTracker(ckpt["tower_provenance"])
            self.last_eval = dict(ckpt.get("metrics", {}))
            for split, st in (ckpt.get("evaluator_state") or {}).items():
                if split in self.evaluators:
                    self.evaluators[split].load_state_dict(st)
            self._first_backward_done = True  # the unused-param check already passed in this run
            self.state.last_ckpt_step = self.state.step  # this file is that step's checkpoint
            self._resumed = True
        elif resume is not None:
            log.info("--resume %s: no checkpoint found, starting from scratch", resume)

    # ------------------------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------------------------
    @property
    def raw_model(self) -> nn.Module:
        return unwrap(self.model)

    def _check_amp_vs_frozen_dtype(self) -> None:
        """Explicit bf16-frozen towers under fp16 autocast are forbidden (``auto`` follows the AMP dtype)."""
        amp, vision = self.cfg.train.amp, self.cfg.model.vision
        if amp.enabled and amp.dtype == "fp16" and vision.frozen_dtype == "bf16":
            raise ConfigError(
                "model.vision.frozen_dtype=bf16 cannot be combined with train.amp.dtype=fp16; "
                "set model.vision.frozen_dtype: auto (follows AMP) or fp32"
            )

    def _configure_backend(self) -> None:
        t = self.cfg.train
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = t.allow_tf32
            torch.backends.cudnn.allow_tf32 = t.allow_tf32
            torch.backends.cudnn.benchmark = t.cudnn_benchmark and not t.deterministic
        if t.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)

    @staticmethod
    def _sampler_of(loader: Any) -> Any:
        for cand in (getattr(loader, "sampler", None), getattr(loader, "batch_sampler", None)):
            if cand is not None and hasattr(cand, "set_epoch") and hasattr(cand, "skip_first"):
                return cand
        raise ConfigError(
            "the train loader's sampler must implement set_epoch(epoch) and skip_first (resume protocol)"
        )

    def _check_eval_wiring(self) -> None:
        missing = sorted(set(self.loaders.evals) - set(self.evaluators))
        extra = sorted(set(self.evaluators) - set(self.loaders.evals))
        if missing or extra:
            raise ConfigError(
                f"eval splits and evaluators disagree: loaders without evaluator {missing}, "
                f"evaluators without loader {extra}"
            )

    def _resolve_selection(self) -> tuple[str | None, str]:
        if self.cfg.eval.selection_metric is not None:
            return self.cfg.eval.selection_metric, self.cfg.eval.selection_mode
        metric, mode = self.task.selection_metric()
        if metric is not None and metric.split("/", 1)[0] not in self.loaders.evals:
            raise ConfigError(
                f"task selection metric '{metric}' names a split without an eval loader"
            )
        return metric, mode

    def _set_train_mode(self, raw: nn.Module) -> None:
        raw.train()
        self.task.on_train_mode(raw)

    @property
    def effective_batch(self) -> int:
        return int(self.cfg.data.batch_size) * self.accum * self.ctx.world_size

    def describe(self) -> str:
        raw = self.raw_model
        warmup = resolve_warmup_steps(self.cfg.schedule, self.total_steps)
        lines = [
            f"task={self.task.name} model={type(raw).__name__} params={count_parameters(raw):,} "
            f"trainable={count_parameters(raw, trainable_only=True):,}",
            f"frozen paths={self.frozen_paths} provenance={self.provenance.as_dict()} excluded_from_ckpt={self.excluded}",
            f"tower dtypes={ {k: str(v) for k, v in self.tower_dtypes.items()} }",
            f"effective batch = {self.cfg.data.batch_size} x accum {self.accum} x world {self.ctx.world_size} = {self.effective_batch}",
            f"steps/epoch={self.steps_per_epoch} total_steps={self.total_steps} warmup={warmup} "
            f"schedule={self.cfg.schedule.kind} {self.amp.describe()} device={self.device}",
        ]
        if self.cfg.logging.log_param_groups:
            lines.append(describe_param_groups(self.param_groups))
        return "\n".join(lines)

    # ------------------------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------------------------
    def fit(self) -> dict[str, Any]:
        log.info("startup\n%s", self.describe())
        if self.dry_run:
            return self._dry_run()
        if self.cfg.eval.at_start and self.state.step == 0 and not self._resumed:
            self._evaluate_and_select()
        while not self._stop_requested and self.state.step < self.total_steps:
            self._run_epoch()
        self._save_step_checkpoint()  # last.pt always reflects the final state
        final = self._final_evaluation()
        self.close()
        return {
            "step": self.state.step,
            "epoch": self.state.epoch,
            "best_metric": self.state.best_metric,
            "best_step": self.state.best_step,
            "last_eval": dict(self.last_eval),
            "final": final,
            "skipped_steps": self.state.skipped_steps,
        }

    def _new_iterator(self) -> Any:
        # Creating a DataLoader iterator draws a base seed from the global RNG; fork it so a
        # mid-epoch resume reproduces the exact dropout stream of the interrupted run.
        with torch.random.fork_rng(devices=[]):
            return iter(self.train_loader)

    def _run_epoch(self) -> None:
        """Run the remaining optimizer steps of ``state.epoch`` (the epoch rolls in ``_after_step``)."""
        st = self.state
        self.sampler.set_epoch(st.epoch)
        done = st.micro_in_epoch // self.accum
        if done >= self.steps_per_epoch:
            raise RuntimeError(
                f"inconsistent train state: micro_in_epoch={st.micro_in_epoch} covers "
                f"{done} >= steps_per_epoch={self.steps_per_epoch} (the epoch should have rolled)"
            )
        if st.micro_in_epoch > 0:
            self.sampler.skip_first = st.micro_in_epoch * self._batch_size()
            log.info(
                "resuming epoch %d after %d optimizer steps (skipping %d samples)",
                st.epoch,
                done,
                self.sampler.skip_first,
            )
        it = self._new_iterator()
        for _ in range(self.steps_per_epoch - done):
            result = self.train_optimizer_step(it)
            self._after_step(result)
            if self._stop_requested or st.step >= self.total_steps:
                break

    def _batch_size(self) -> int:
        bs = getattr(self.train_loader, "batch_size", None)
        if not bs:
            raise ConfigError(
                "train loader must be built with batch_size= (needed for skip_first on resume)"
            )
        return int(bs)

    def train_optimizer_step(self, it: Any) -> StepResult:
        """Consume ``accum`` micro-batches from ``it`` and apply one optimizer step."""
        st, cfg = self.state, self.cfg.train
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=self.device)
        metric_sums: dict[str, torch.Tensor] = {}
        samples = tokens = 0
        for micro in range(self.accum):
            try:
                batch = next(it)
            except StopIteration:
                raise RuntimeError(
                    f"train loader exhausted at step {st.step} micro {micro}: the sampler yielded fewer "
                    f"batches than len(loader)={len(self.train_loader)}"
                ) from None
            batch = move_to_device(batch, self.device, non_blocking=self.device.type == "cuda")
            sc = StepContext(step=st.step, micro=micro, accum=self.accum, device=self.device)
            with sync_context(self.model, sync=micro == self.accum - 1):
                with self.amp.autocast():
                    out = self.task.train_step(self.model, batch, sc)
                loss = out.loss
                if not bool(torch.isfinite(loss.detach())):
                    self._on_nonfinite_loss(loss, micro)
                else:
                    self._consecutive_nonfinite = 0
                self.amp.scaler.scale(loss / self.accum).backward()
            if not self._first_backward_done:
                self._first_backward_done = True
                check_all_trainable_params_received_grad(self.model)
                self.task.after_first_backward(self.raw_model)
            loss_sum += loss.detach().float()
            samples += out.batch_size
            tokens += out.num_tokens
            for k, v in out.metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + torch.as_tensor(
                    v, dtype=torch.float32, device=self.device
                )
        # -- optimizer step ------------------------------------------------------------
        self.amp.scaler.unscale_(self.optimizer)
        max_norm = cfg.max_grad_norm if cfg.max_grad_norm is not None else math.inf
        grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm)
        skipped = False
        if not bool(torch.isfinite(grad_norm)):
            if self.amp.nonfinite_policy != "skip":
                raise FloatingPointError(
                    f"non-finite gradient norm {float(grad_norm)} at step {st.step} (bf16/fp32 run): aborting"
                )
            st.skipped_steps += 1
            self._consecutive_skipped += 1
            skipped = True
            self.optimizer.zero_grad(set_to_none=True)
            self.amp.scaler.update()  # backs the scale off using the inf recorded by unscale_
            log.warning(
                "step %d skipped: non-finite gradient norm (total skipped %d, consecutive %d)",
                st.step,
                st.skipped_steps,
                self._consecutive_skipped,
            )
            if self._consecutive_skipped > cfg.max_consecutive_nonfinite:
                raise RuntimeError(
                    f"{self._consecutive_skipped} consecutive non-finite gradient norms / skipped "
                    f"optimizer steps (> train.max_consecutive_nonfinite={cfg.max_consecutive_nonfinite}): "
                    "the run has diverged, aborting"
                )
        else:
            self._consecutive_skipped = 0
            self.amp.scaler.step(self.optimizer)
            self.amp.scaler.update()
        self.scheduler.step()  # also on skipped steps: the schedule follows state.step
        return StepResult(
            loss=loss_sum / self.accum,
            samples=samples,
            tokens=tokens,
            grad_norm=float(grad_norm),
            skipped=skipped,
            metrics={k: v / self.accum for k, v in metric_sums.items()},
        )

    def _on_nonfinite_loss(self, loss: torch.Tensor, micro: int) -> None:
        st = self.state
        st.nonfinite_micro_batches += 1
        self._consecutive_nonfinite += 1
        value = float(loss.detach())
        if self.amp.nonfinite_policy != "skip":
            raise FloatingPointError(
                f"non-finite loss {value} at step {st.step} micro {micro} under {self.amp.describe()}: "
                "this is a bug, not an fp16 overflow"
            )
        if self._consecutive_nonfinite > self.cfg.train.max_consecutive_nonfinite:
            raise RuntimeError(
                f"{self._consecutive_nonfinite} consecutive non-finite micro-batches "
                f"(> train.max_consecutive_nonfinite={self.cfg.train.max_consecutive_nonfinite}): aborting"
            )
        log.warning(
            "non-finite loss %s at step %d micro %d (fp16): step will be skipped",
            value,
            st.step,
            micro,
        )

    def _after_step(self, result: StepResult) -> None:
        st = self.state
        st.step += 1
        st.micro_in_epoch += self.accum
        st.samples_seen += result.samples * self.ctx.world_size
        self.task.post_optimizer_step(self, st)
        self.meter.update(result.samples * self.ctx.world_size, result.tokens * self.ctx.world_size)
        self._accumulate_log(result)
        if st.step % self.cfg.train.log_every == 0 or st.step == self.total_steps:
            self._log_train(result)
        epoch_done = st.micro_in_epoch >= self.steps_per_epoch * self.accum
        if epoch_done:  # roll before the cadence so a boundary checkpoint stores a fresh epoch
            st.epoch += 1
            st.micro_in_epoch = 0
        ev, ck = self.cfg.eval, self.cfg.checkpoint
        if (ev.every_steps and st.step % ev.every_steps == 0) or (
            epoch_done and ev.every_epochs and st.epoch % ev.every_epochs == 0
        ):
            self._evaluate_and_select()
        rp = self.cfg.analysis.routing_probe
        if (
            rp is not None
            and st.step % rp.every_steps == 0
            and (rp.max_steps is None or st.step <= rp.max_steps)
            and st.last_eval_step != st.step  # the full eval of this step already ran the probe
        ):
            self._run_routing_probe(rp.split)
        if (ck.every_steps and st.step % ck.every_steps == 0) or (
            epoch_done and ck.every_epochs and st.epoch % ck.every_epochs == 0
        ):
            self._save_step_checkpoint()

    def _run_routing_probe(self, split: str) -> dict[str, float]:
        """Routing probe on its own cadence (paper Sec. 6.5: every 500 steps, independent of eval).

        Runs only the ``routing_probe`` part of the split's :class:`CompositeEvaluator` (all ranks,
        collectives inside), logs ``{split}/routing/...`` and appends to ``eval_history.jsonl``;
        never touches selection / early stopping.
        """
        from vivlmoe.tasks.evaluators.composite import CompositeEvaluator

        composite = self.evaluators[split]
        if not isinstance(composite, CompositeEvaluator):
            return {}  # dense model: maybe_add_routing_probe did not wrap the evaluator
        probes = [e for e in composite.evaluators if e.name == "routing_probe"]
        if not probes:
            return {}
        raw = self.raw_model
        raw.eval()
        ec = EvalContext(
            ctx=self.ctx,
            step=self.state.step,
            run_dir=self.run_dir,
            split_name=split,
            writer=self.writer,
            save_predictions=self.cfg.eval.save_predictions,
        )
        with torch.no_grad():
            local = probes[0].run(raw, self.loaders.evals[split], ec)
        self._set_train_mode(raw)
        if self.ctx.is_main and local is None:
            raise RuntimeError("routing probe returned None on rank 0")
        results = {f"{split}/{k}": float(v) for k, v in broadcast_object(local, self.ctx).items()}
        if results:
            self.writer.log_scalars(results, self.state.step)
            if self.ctx.is_main:
                append_jsonl(
                    self.run_dir / "eval_history.jsonl",
                    {"step": self.state.step, "epoch": self.state.epoch, **results},
                )
            log.info(
                "routing probe %s @ step %d: %s",
                split,
                self.state.step,
                {k: round(v, 5) for k, v in results.items() if k.count("/") == 2},
            )
        return results

    # ------------------------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------------------------
    def _accumulate_log(self, result: StepResult) -> None:
        self._log_count += 1
        self._log_sums["loss"] = self._log_sums.get("loss", 0.0) + result.loss
        for k, v in result.metrics.items():
            self._log_sums[k] = self._log_sums.get(k, 0.0) + v

    def _log_train(self, result: StepResult) -> None:
        keys = sorted(self._log_sums)
        packed = torch.stack([self._log_sums[k] / max(self._log_count, 1) for k in keys]).float()
        packed = all_reduce_mean(packed, self.ctx)
        scalars = {f"train/{k}": float(v) for k, v in zip(keys, packed.tolist(), strict=True)}
        for g in self.optimizer.param_groups:
            scalars[f"lr/{g['name']}"] = float(g["lr"])
        if self.cfg.logging.log_grad_norm:
            scalars["train/grad_norm"] = result.grad_norm
        scalars.update({f"throughput/{k}": v for k, v in self.meter.snapshot().items()})
        if self.cfg.logging.log_memory:
            peak = peak_memory_gb(self.device)
            if peak is not None:
                scalars["memory/peak_gb"] = peak
        scalars["train/epoch"] = float(self.state.epoch)
        self.writer.log_scalars(scalars, self.state.step)
        if self.ctx.is_main:
            brief = " ".join(
                f"{k.split('/', 1)[1]}={v:.4g}"
                for k, v in scalars.items()
                if k.startswith("train/")
            )
            lrs = " ".join(f"{k[3:]}={v:.2e}" for k, v in scalars.items() if k.startswith("lr/"))
            log.info(
                "step %d/%d ep %d | %s | %s | %.1f samples/s",
                self.state.step,
                self.total_steps,
                self.state.epoch,
                brief,
                lrs,
                scalars["throughput/samples_per_s"],
            )
        self._log_sums.clear()
        self._log_count = 0
        self.meter.reset()

    # ------------------------------------------------------------------------------------
    # evaluation / selection / checkpoints
    # ------------------------------------------------------------------------------------
    def evaluate(self, splits: list[str]) -> dict[str, float]:
        """Run the evaluators on the unwrapped model; results are broadcast from rank 0."""
        raw = self.raw_model
        raw.eval()
        results: dict[str, float] = {}
        with torch.no_grad():
            for split in splits:
                ec = EvalContext(
                    ctx=self.ctx,
                    step=self.state.step,
                    run_dir=self.run_dir,
                    split_name=split,
                    writer=self.writer,
                    save_predictions=self.cfg.eval.save_predictions,
                )
                local = self.evaluators[split].run(raw, self.loaders.evals[split], ec)
                if self.ctx.is_main and local is None:
                    raise RuntimeError(f"evaluator '{split}' returned None on rank 0")
                res = broadcast_object(local, self.ctx)
                for k, v in res.items():
                    results[f"{split}/{k}"] = float(v)
                log.info(
                    "eval %s @ step %d: %s",
                    split,
                    self.state.step,
                    {k: round(float(v), 5) for k, v in res.items()},
                )
        self._set_train_mode(raw)
        return results

    def _evaluate_and_select(self) -> dict[str, float]:
        results = self.evaluate(list(self.loaders.evals))
        self.last_eval = results
        self.state.last_eval_step = self.state.step
        self.writer.log_scalars(results, self.state.step)
        if self.ctx.is_main:
            append_jsonl(
                self.run_dir / "eval_history.jsonl",
                {"step": self.state.step, "epoch": self.state.epoch, **results},
            )
        if self.selection_metric is not None:
            if self.selection_metric not in results:
                raise KeyError(
                    f"selection metric '{self.selection_metric}' not in eval results {sorted(results)}"
                )
            value = results[self.selection_metric]
            if self.early.record(value, self.state):
                log.info(
                    "new best %s=%.6g at step %d", self.selection_metric, value, self.state.step
                )
                if self.cfg.checkpoint.save_best:
                    self._save_best()
            if self.early.should_stop(self.state):
                self._stop_requested = True
        return results

    def _build_ckpt(self, include_optimizer: bool) -> dict[str, Any]:
        extras = dict(self.task.checkpoint_metadata())
        extras["evaluator_state"] = {k: e.state_dict() for k, e in self.evaluators.items()}
        return build_checkpoint(
            model=self.raw_model,
            excluded=self.excluded,
            provenance=self.provenance.as_dict(),
            config=to_dict(self.cfg),
            train_state=self.state.state_dict(),
            task_state=self.task.state_dict(),
            ctx=self.ctx,
            optimizer=self.optimizer if include_optimizer else None,
            scheduler=self.scheduler,
            scaler=self.amp.scaler if self.amp.scaler.is_enabled() else None,
            metrics=self.last_eval,
            extras=extras,
        )

    def _save_step_checkpoint(self) -> None:
        if self.state.last_ckpt_step == self.state.step:
            return
        include = self.cfg.checkpoint.save_optimizer in ("always", "last_only")
        self.ckpt.save_step(self._build_ckpt(include), self.state.step)
        self.state.last_ckpt_step = self.state.step

    def _save_best(self) -> None:
        self.ckpt.save_best(self._build_ckpt(self.cfg.checkpoint.save_optimizer == "always"))

    def _final_evaluation(self) -> dict[str, float]:
        splits = list(self.cfg.eval.final_splits)
        if not splits:
            return {}
        if self.ckpt.best_path.exists():
            ckpt = load_checkpoint_file(self.ckpt.best_path)
            load_model_state(self.raw_model, ckpt, self.provenance.as_dict())
            log.info("final evaluation on %s with best.pt (step %d)", splits, ckpt["step"])
        else:
            log.warning("final evaluation on %s: best.pt not found, using the last weights", splits)
        results = {f"final/{k}": v for k, v in self.evaluate(splits).items()}
        self.writer.log_scalars(results, self.state.step)
        if self.ctx.is_main:
            append_jsonl(
                self.run_dir / "eval_history.jsonl",
                {"step": self.state.step, "final": True, **results},
            )
        return results

    # ------------------------------------------------------------------------------------
    # dry run
    # ------------------------------------------------------------------------------------
    def _dry_run(self) -> dict[str, Any]:
        reset_peak_memory(self.device)
        self.meter.reset()
        self.sampler.set_epoch(0)
        it = self._new_iterator()
        losses = []
        for _ in range(self.total_steps):
            result = self.train_optimizer_step(it)
            self.state.step += 1
            self.task.post_optimizer_step(self, self.state)
            self.meter.update(
                result.samples * self.ctx.world_size, result.tokens * self.ctx.world_size
            )
            losses.append(float(all_reduce_mean(result.loss, self.ctx)))
        report = {
            "dry_run": True,
            "steps": self.state.step,
            "effective_batch": self.effective_batch,
            "trainable_params": count_parameters(self.raw_model, trainable_only=True),
            "total_params": count_parameters(self.raw_model),
            "losses": losses,
            "peak_memory_gb": peak_memory_gb(self.device),
            "throughput": self.meter.snapshot(),
            "param_groups": [
                {"name": g.name, "lr": g.lr, "weight_decay": g.weight_decay, "params": g.num_params}
                for g in self.param_groups
            ],
        }
        log.info("dry-run report: %s", {k: v for k, v in report.items() if k != "param_groups"})
        self.close()
        return report

    def close(self) -> None:
        self.writer.close()


def _tower_paths(model: nn.Module) -> dict[str, str]:
    """Typed tower lookup from :mod:`vivlmoe.models` (imported lazily: the models package is M3)."""
    from vivlmoe.models.towers import find_tower_paths

    return find_tower_paths(model)


def _apply_frozen_dtype_policy(model: nn.Module, amp_dtype: torch.dtype | None) -> dict[str, Any]:
    """Cast frozen vision towers once, on the target device, before DDP / any fingerprint."""
    from vivlmoe.models.freezing import apply_frozen_dtype_policy

    return apply_frozen_dtype_policy(model, amp_dtype)


__all__ = ["StepResult", "Trainer"]
