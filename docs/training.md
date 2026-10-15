# Training

`scripts/train.py --config <yaml> [--resume auto|<path>] [--dry-run] [key.path=value ...]`
runs one Task (`task.name`) with the single `Trainer` (`vivlmoe/training/trainer.py`).
Launch with `torchrun --nproc_per_node N` for DDP; a plain `python` launch is single-process.

## Startup order

seed → `task.build_model()` on CPU → freezing (`task.always_frozen()` + `optim.freeze_preset` via
`vivlmoe.models.freezing` + `optim.freeze`; a `VisionTower` is frozen through `set_trainable`) →
`init_from` partial load (strict per prefix) → `.to(device)` → AMP state → frozen-dtype policy
(`vivlmoe.models.freezing.apply_frozen_dtype_policy`: a frozen ViT becomes bf16 on CUDA here, once,
before DDP and before any checkpoint fingerprint) → `wrap_ddp` → per-rank RNG streams →
`task.build_loaders()` → step budget → parameter groups → AdamW → scheduler → metrics writer →
checkpoint manager → resume. `train.amp.dtype: fp16` with `model.vision.frozen_dtype: bf16|auto`
is rejected at startup. The unused-trainable-parameter check runs after the first backward on
every rank (hard error), followed by `task.after_first_backward(model)`.

## Budget and batches

Exactly one of `train.max_steps` / `train.epochs`. `steps_per_epoch = len(train_loader) //
grad_accum_steps`; leftover micro-batches at the end of an epoch are dropped so budgets and
resumes are exact. Effective batch = `data.batch_size × grad_accum_steps × world_size` (logged at
startup; contrastive negatives come from the per-step global batch, accumulation adds none).

## Optimizer step

`zero_grad(set_to_none)` → for each micro-batch: `no_sync` unless last, `task.train_step`,
`scaler.scale(loss / accum).backward()` → `unscale_` → `clip_grad_norm_(train.max_grad_norm)` →
`scaler.step / update` → `scheduler.step()` once → `state.step += 1` → `task.post_optimizer_step`
(MoE bias update) → log every `train.log_every` steps (all-reduced means, per-group lrs, grad norm,
throughput, peak memory) → eval / checkpoint cadence (`every_steps` or `every_epochs`).

Non-finite handling: under bf16 / fp32 a non-finite loss or gradient norm raises. Under fp16 the
backward still runs (so DDP's all-reduce stays consistent), the optimizer step is skipped, the
`GradScaler` backs off, and the run aborts after `train.max_consecutive_nonfinite` consecutive
non-finite micro-batches **or** consecutive skipped optimizer steps (non-finite unscaled gradient
norm with a finite loss). Skipped steps still advance `state.step` and the scheduler so the budget,
lr schedule and cadences stay aligned. Counts live in `TrainState.skipped_steps /
nonfinite_micro_batches`.

Epoch bookkeeping: the optimizer step that completes an epoch rolls `state.epoch` and runs the
`eval.every_epochs` evaluation *before* any checkpoint of that step, so a checkpoint written at an
epoch boundary stores `(epoch+1, micro_in_epoch=0)` and resumes into a fresh epoch without
repeating the evaluation (`TrainState.last_eval_step / last_ckpt_step` are persisted).

## Evaluation and selection

Evaluators run on the unwrapped model in `eval()` under `no_grad`; the rank-0 dict is broadcast so
early stopping (`train.early_stopping`) is identical on all ranks. Scalars are logged as
`<split>/<metric>`, which is the `eval.selection_metric` syntax. `best.pt` is written on improvement
(no optimizer unless `checkpoint.save_optimizer: always`), `last.pt` after every step checkpoint and
at the end. `eval.final_splits` are evaluated once with `best.pt` (falls back to the last weights with
a warning) and reported under `final/<split>/<metric>` in `final_metrics.json`.

## Dry run

`--dry-run` builds everything, runs 2 optimizer steps without checkpoints or TensorBoard, and writes
`dry_run_report.json` (param-group table, effective batch, trainable params, peak memory,
throughput). Run it for every real config on the training box before launching.

## Run directory

`config.resolved.yaml`, `command.txt`, `train.log` (rank 0), `tb/`, `checkpoints/`,
`eval_history.jsonl`, `final_metrics.json`.

## Batch and memory plan for 4x A40 48 GB (paper-scale config)

| stage | per-GPU batch x accum x GPUs | trainable | notes |
|---|---|---|---|
| Stage 1 teacher | 128 x 4 x 4 = 2048 | XLM-R large + FC1 | text only; CLIP text teacher frozen |
| Stage 2 contrastive | 64 x 8 x 4 = 2048 | XLM-R, `W_t`, resampler, `f_enc` | ViT-H frozen in bf16 (`frozen_dtype: auto`); negatives per step = 256 |
| Stage 3 end-to-end | 32 x 16 x 4 = 2048 | everything | ViT-H @378 trainable: `gradient_checkpointing: true` for vision and text (non-reentrant); ~32 GB/GPU |
| MoE sparse training | 128 x 4 x 4 = 2048, 20K steps | all but ViT | negatives per step = 512 |
| downstream | 8 x 1 x 4 = 32 | all but ViT (+ T5 encoder removed) | seq2seq ~1.85B trainable params: enable `model.generation.gradient_checkpointing` if memory is tight |

Run `scripts/train.py --config <stage>.yaml --dry-run` first: it prints the parameter-group table, the
effective batch and the peak memory of two real optimizer steps. `scripts/bench_loader.py` sizes
`data.num_workers` so the frozen ViT-H is never starved. `torch >= 2.6` is recommended on the box
(`.bin` checkpoints load directly); with older torch use the safetensors dirs written by
`scripts/fetch_models.py`. Set `HF_HUB_OFFLINE=1` on nodes without internet once the models are cached.
