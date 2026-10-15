# Tasks and datasets

## The Task protocol

One `Trainer` (`vivlmoe/training/trainer.py`), one `Task` per *objective* (`vivlmoe/tasks/base.py`,
plan decision 1). Tasks are registered in `TASKS` by `task.name` (`vivlmoe/tasks/registry.py`;
`import_builtin_tasks()` loads `stage1_teacher`, `contrastive`, `captioning`, `vqa`, `vimacsa`).
Datasets are registry *kinds*, not tasks (see "Adding a dataset" below).

| Method | Called by the Trainer | Contract |
|---|---|---|
| `build_model()` | once, on CPU, after `seed_everything` | returns the task model; every task model exposes `forward(batch: dict, mode: str)` and is called **exactly once per micro-batch** through the DDP wrapper (decision 2) |
| `always_frozen()` | before freezing | module paths frozen regardless of `optim.freeze` |
| `build_loaders()` | after DDP wrapping | `LoaderBundle(train, evals)`; the train sampler implements `set_epoch` / `skip_first`; eval loaders shard `rank::world` without padding. Every built-in task splits it into `build_train_loader()` + public `build_eval_loaders()` so `scripts/evaluate.py` / `scripts/analyze_routing.py` can rebuild the eval side of a checkpoint without indexing `data.train` |
| `train_step(model, batch, sc)` | every micro-batch, inside autocast | returns `StepOutput(loss, batch_size, num_tokens, metrics)` with the *un-divided* loss; metrics are detached scalars averaged across ranks at log time |
| `post_optimizer_step(trainer, state)` | after `optimizer.step()`, all ranks | the MoE aux-free bias update lives here (decision 7); may log through `trainer.writer` |
| `after_first_backward(model)` | once, unwrapped model, after the first *real* micro-batch's backward | a no-op for every built-in task: the Trainer has no startup forward, so resetting MoE dispatch counters here would drop the first micro-batch's routing counts from the first Eq. 22 update |
| `on_train_mode(raw_model)` | after every `model.train()` | keep frozen towers in eval mode |
| `evaluators()` | after `build_loaders()` | `{split: Evaluator}`; keys must equal `LoaderBundle.evals` |
| `selection_metric()` | when `eval.selection_metric` is unset | `("<split>/<metric>", "max"|"min")` |
| `state_dict()` / `load_state_dict()` | checkpoints / resume | task-private state |
| `checkpoint_metadata()` | every checkpoint | `dev_split_sha1`, `transform_spec` |

Evaluators (`vivlmoe/tasks/evaluators/base.py`) run on the unwrapped model in `eval()` under
`no_grad`, must call `gather_ordered` / `gather_ordered_tensors` **unconditionally** (they are
collectives that reassemble per-rank shards by `uid` and assert coverage of `0..N-1`, even on an
empty shard), and return the metric dict on rank 0 (`None` elsewhere); the Trainer broadcasts it.
No collective is ever called inside an `nn.Module`; everything goes through
`vivlmoe.training.distributed`. Every evaluator also writes its metric dict (plus split sizes /
settings) to `<run_dir>/metrics/<split>_step<N>.json` on rank 0 (`vivlmoe/tasks/evaluators/dump.py`);
the Trainer appends the merged results to `<run_dir>/eval_history.jsonl`.

Batches are plain dicts (keys per `vivlmoe/data/collators.py`); the tasks tokenise inside the
collators and never look at file formats. The encoder `f_enc` of any task model is reached with
`getattr(model, "backbone", model).encoder` (`vivlmoe.tasks.contrastive.encoder_of`).

## Pretraining tasks

### `stage1_teacher` - `TeacherStudentTask` (`vivlmoe/tasks/stage1_teacher.py`)

Paper Sec. 4.1, Eq. 12: `L_1 = (1/N) sum_i ||z_vi - z_en||^2`.

- Model: `build_task_model("stage1")` = `Stage1Student` (`text_encoder` + `fc1`). The frozen CLIP
  text teacher (`models.factory.build_teacher`) is **owned by the task** (decision 11): built lazily
  on the first micro-batch, moved to the batch device, permanently `eval()` + `no_grad`; it is not a
  submodule, so DDP and checkpoints never see it.
- Data: `data.train` is a parallel-text source (`phomt`, `synthetic_parallel`) through
  `build_parallel_text` + `TeacherCollator` (`en_input_ids [B,77]` for CLIP, `vi_input_ids` for
  XLM-R; `build_tokenizers(need_clip=True)`, fake tiny tokenizers for `pretrained: false` towers).
- `train_step`: `z_en = teacher(en)`, `out = model(batch, "stage1")`,
  `TeacherMSELoss(model.stage1.loss_preset)(out.student, z_en)`; logs `student_norm`, `teacher_norm`.
- Evaluators: every parallel-text eval split gets a `TeacherMSEEvaluator` (`{split}/mse`, per-sample
  distances gathered by uid, mean on rank 0). With `task.eval_clip_retrieval: true`, image-text eval
  splits (KTVIC / UIT-ViIC dev) get a `ClipSpaceRetrievalEvaluator`: CLIP image embeddings from
  `build_image_teacher` are computed **once per split** on all ranks and cached on the CPU, the text
  side is the L2-normalised student output, and `recall_at_k` gives `i2t_r1..t2i_r10`, `rsum`
  (Appendix A). Without the flag an image-text eval split is a `ConfigError`.
- Selection default: first parallel split `/mse`, `min`. Freezing: `optim.freeze_preset: stage1`.

### `contrastive` - `ContrastiveTask` (`vivlmoe/tasks/contrastive.py`)

Paper Sec. 4.2-4.3 (Eq. 13) and Sec. 5.4 - the same task drives Stage 2, Stage 3 and MoE sparse
training; only the config differs (freeze preset, lr groups, `model.moe`).

- Model: `build_task_model("backbone")` = `ViVLBackbone`.
- Data: `build_image_text_train` + `ContrastiveCollator`; the train sampler draws one caption per
  image per epoch (`data.one_caption_per_image`). Each `data.evals` entry must be an image-text
  split and becomes a `RetrievalLoaders` pair (image side + text side of a `RetrievalEvalDataset`).
- `train_step`: `out = model(batch, "contrastive")` (two separate single-modality passes inside one
  forward), `ContrastiveLoss(loss.contrastive.gather, symmetric_reduction)`; for a MoE encoder
  `RouterLossComputer(loss.router)(out.router_outputs)` adds `aux` (z-loss / modality-importance /
  optional Switch-LB, M6) and its logs. Metrics: `loss_contrastive`, `acc_i2t`, `acc_t2i`,
  `logit_scale` (+ `loss_aux`, `loss_z`, `loss_mi`, ...).
- MoE bookkeeping: `post_optimizer_step` runs `aux_free_bias_step(encoder, ctx, eta)` (one
  all-reduce, Eq. 22 sign rule, counters reset) when `model.moe.aux_free_bias` and
  `bias_update_every: step` (`micro` runs it at the end of every `train_step`); its logs
  (`moe/expert_frac_max`, `moe/expert_frac_min`, `moe/bias_range`) go to `trainer.writer` every
  `train.log_every` steps. The update sees the counts of every micro-batch of the step (nothing is
  reset in `after_first_backward`; `aux_free_bias_step` resets the counters itself).
- Evaluators: `RetrievalEvaluator` per split (`encode_image` / `encode_text` modes, gathered by
  uid, `sim = img @ txt.T` on the CPU in fp32, `recall_at_k` with the dataset's `img2caps` /
  `cap2img`; keys `i2t_r1, i2t_r5, i2t_r10, t2i_r1, t2i_r5, t2i_r10, rsum`). With
  `analysis.routing_probe` set, `maybe_add_routing_probe(evaluators, cfg, ctx, "contrastive")` (M6)
  wraps the probed split.
- Selection default: first eval split `/rsum`, `max`. Checkpoint metadata: `dev_split_sha1`
  (`{split: sha1 | null}` from the adapters' holdout info) and `transform_spec`.

### Stage configs

`configs/pretrain/{stage1_teacher,stage2_contrastive,stage3_end2end}.yaml` (effective batch
2048 = per-rank batch x `grad_accum_steps` x 4 GPUs, documented in each file). **Recipe deviation:**
the InfoNCE negatives are the per-*step* global batch `data.batch_size x world` (plan decision 15) -
256 pairs in Stage 2/3 and 512 in `configs/moe/sparse_train.yaml` against the paper's 2048; gradient
accumulation recovers the optimizer batch, not the negatives. `ContrastiveTask.build_train_loader`
logs `contrastive negatives per step = B x world` at startup and `docs/paper_mapping.md` records it;
on the A40s prefer `grad_accum_steps: 1` with the largest per-GPU batch that fits (frozen-ViT feature
cache, M9). Stage 2 starts from
`init_from: {path: <stage1 best>, load: [text_encoder]}`, Stage 3 from the Stage-2 best with
`load: [all]`, `freeze_preset: stage3` and two lr groups (encoders 1e-5, projections 5e-5); the
ViT is trainable there, so its provenance becomes `trained` and it is stored in every later
checkpoint. Shared pieces: `configs/data/evals_retrieval.yaml` (KTVIC / UIT-ViIC dev holdouts; roots
from `configs/_base/data_paths.yaml`) and `configs/data/pretrain_corpus.yaml` (the decontaminated
`jsonl` corpus written by `scripts/build_pretrain_corpus.py`). `configs/_smoke/{stage1,stage2,stage3}_tiny.yaml` run the same chain on CPU with random tiny
towers and synthetic data (`tests/integration/test_cli_smoke.py`, `pytest -m slow`).

## Adding a dataset

Datasets are registry *kinds*, not tasks: an adapter turns files into immutable records and an
image store, and any task that consumes that record type can use it.

1. Create `vivlmoe/data/adapters/<name>.py`:

   ```python
   from dataclasses import dataclass
   from vivlmoe.config.schema import SplitSpec
   from vivlmoe.data.adapters.base import BaseAdapter
   from vivlmoe.data.records import ImageRef, ImageTextRecord, ListRecordSet
   from vivlmoe.data.registry import register_dataset

   @register_dataset("mydata")
   class MyAdapter(BaseAdapter):
       @dataclass
       class Options:            # validated strictly from YAML `options:` (unknown keys fail)
           caption_key: str = "caption"

       default_splits = {"train": SplitSpec(file="train.json", image_root="images")}

       def _raw_records(self, split):
           data = ...  # read self.split_file(split)
           return ListRecordSet([ImageTextRecord(uid=i, image_id=..., image=ImageRef(0, filename),
                                                 text=..., refs=(...)) for i, ... in enumerate(data)])
   ```

   Record types: `ImageTextRecord` (contrastive / captioning / retrieval), `VQARecord`,
   `MultiImageClsRecord` (ViMACSA-style), `ParallelTextRecord` (Stage 1). Text-only adapters
   override `_image_check()` to return `"none"`. Non-local images: override `image_store(split)`
   (see `HFImageStore`). Large corpora: return a `ColumnarImageText` instead of a list.
2. Import the module in `vivlmoe/data/adapters/__init__.py`.
3. Reference it from YAML — everything else (holdout, image validation, normalisation, `limit`,
   uid renumbering, samplers, collators) is inherited from `BaseAdapter.load_split`:

   ```yaml
   data:
     train: {kind: mydata, root: /data/mydata, split: train, image_check: exists,
             missing_image_policy: error, text_normalize: pretrain, options: {caption_key: caption}}
     evals:
       dev: {kind: mydata, root: /data/mydata, split: dev,
             holdout: {from_split: train, to_split: dev, ids_file: configs/data/splits/mydata_dev_image_ids.json}}
   ```

   Generate and commit the holdout id list with `scripts/make_dev_split.py` (or
   `vivlmoe.data.splits.holdout_image_ids`); its sha1 is stored in every checkpoint.
4. Add a fixture-based test under `tests/unit/data/` (tiny JSON + `write_images`, no network).

Useful one-offs: `scripts/build_manifest.py` (validate/decode every image once and cache the
result), `scripts/bench_loader.py` (samples/s of the training loader without a model).

## See also

- `docs/moe.md` - the MoE tasks (`upcycle`, sparse training with `contrastive`), router losses, the
  aux-free bias update and the routing probe (`analysis.routing_probe`, `scripts/analyze_routing.py`).
- `docs/downstream.md` - captioning / VQA / ViMACSA tasks, `GenerationEvaluator` /
  `ClassificationEvaluator`, `scripts/evaluate.py` and the downstream configs.
- `docs/training.md`, `docs/checkpoints.md`, `docs/configs.md` - the Trainer loop, checkpoint
  sidecars / provenance and the config schema the tasks consume.
