# Downstream tasks (paper Sec. 6.1, milestone M7)

Three objectives share one base class (`vivlmoe/tasks/downstream_common.py`,
`DownstreamTask`): image captioning (`captioning`), visual question answering (`vqa`) and
ViMACSA aspect-category sentiment (`vimacsa`). Each one is a `Task` (see `docs/tasks.md`) whose
model wraps the pretrained `ViVLBackbone` (dense or MoE - the topology is inherited from the
checkpoint through `model.from_checkpoint`).

## What the base class does

| Concern | Implementation |
|---|---|
| model | `build_task_model(kind, cfg.model)` with `kind = seq2seq` (captioning, VQA: backbone + `bridge` + ViT5 decoder without its encoder stack) or `classifier` (ViMACSA: backbone + `cls_head`) |
| freezing | `always_frozen()` returns the freeze preset of the objective (`vivlmoe/models/freezing.py`): `downstream_seq2seq` = `backbone.vision_encoder` + `backbone.logit_scale`; `downstream_caption` additionally freezes `backbone.text_encoder` + `backbone.text_proj` (no text tokens enter the joint pass, so they would be unused trainable parameters); `downstream_cls` = ViT + `logit_scale` |
| forward | `model(batch, forward_mode)` is called exactly once per micro-batch inside `train_step` (`joint_caption`, `joint_vqa`, `joint_cls`) |
| MoE stabilisers | `RouterLossComputer(loss.router)` (M6, imported lazily) turns the forward's `router_outputs` into `aux` = z-loss (Eq. 20) + modality-importance loss (Eq. 24) + optional Switch-LB; `loss = task_loss + aux`. Captioning has no text tokens, so `L_mi = 0` (decision 19). Logged as `train/aux`, `train/loss_z`, `train/loss_mi` |
| aux-free bias | `post_optimizer_step` calls `aux_free_bias_step(encoder, ctx, eta)` (one all-reduce, Eq. 22) when `model.moe.aux_free_bias` and `bias_update_every: step`; `micro` runs it at the end of every `train_step`. The update uses the counts of every micro-batch of the step: `after_first_backward` is a no-op (the Trainer has no startup forward) |
| routing probe | `maybe_add_routing_probe(evaluators, cfg, ctx, forward_mode)` (M6) wraps `evaluators[analysis.routing_probe.split]` when the model is MoE; with a dense backbone the configured probe is skipped with a warning (the dense baseline shares `_downstream_defaults.yaml`). The Trainer runs the probe on its own cadence (`Trainer._run_routing_probe` every `routing_probe.every_steps` optimizer steps up to `max_steps`, independent of `eval.every_steps` / `every_epochs`; it also fires inside a regular evaluation that lands on a probe step), so epoch-based downstream evaluation still yields the Sec. 6.5 analysis every 500 steps. `warn_probe_cadence` (M6) only logs when the two cadences do not coincide |
| checkpoints | `checkpoint_metadata()` stores `dev_split_sha1` (sha1 of every committed holdout id list referenced by `data.train` / `data.evals`; `null` when no split uses one) and `transform_spec` |
| eval loaders | `build_eval_loaders()` is public so `scripts/evaluate.py` can reuse it; one ordered-shard loader per `data.evals` entry |
| selection | default `<first eval split>/<first task metric>`, `max`; downstream configs set `eval.selection_metric` explicitly (`dev/bleu4`, `dev/rougeL`, `dev/f1`, `dev/macro_f1`) |

The encoder of any task model is `getattr(model, "backbone", model).encoder`
(`vivlmoe.tasks.downstream_common.encoder_of`).

## `captioning` - `CaptioningTask` (`vivlmoe/tasks/captioning.py`)

Fig. 4a: image tokens only -> joint pass -> `bridge` -> ViT5 decoder; CE with `-100` pads.

- Train data: `build_image_text_train` over `data.train` (`ktvic`, `uitviic`, `synthetic_caption`).
  Every (image, caption) annotation is one example (singleton groups; no one-caption-per-image
  sampler). Collator `Seq2SeqCollator(with_text=False, train=True)` with the decoder tokenizer
  (`build_tokenizers(need_decoder=True)`), targets truncated at `task.max_target_len`.
- Eval data: `build_caption_eval` (one item per unique image, all references) +
  `Seq2SeqCollator(train=False)`; batch keys `pixel_values`, `refs`, `prompts` (empty), `uids`,
  `image_ids`.
- Metrics (`task.metrics`): any of `bleu4`, `rougeL`, `cider`, `f1`, `em` (KTVIC: `[bleu4, rougeL]`,
  UIT-ViIC: `[bleu4]`).

## `vqa` - `VQATask` (`vivlmoe/tasks/vqa.py`)

Fig. 4b: `[image tokens ; question tokens]` -> joint pass -> decoder generates the answer.

- Train data: `build_vqa(expand_answers=task.train_on_all_answers)`; with the flag every gold
  answer becomes its own example, otherwise the first answer is the target. Collator
  `Seq2SeqCollator(with_text=True, prompt_template, max_source_len, max_target_len)`.
- Prompt template: the `vqa_generic` adapter option `prompt_template` is used unless
  `task.prompt_template` is set explicitly (a differing task template overrides the adapter's
  with a warning). It must contain `{question}`.
- Eval: one item per question with all gold answers as references. Metrics: any subset of
  `rougeL`, `cider`, `f1`, `em` (OpenViVQA `[rougeL, cider]`, ViTextVQA / ViOCR-VQA `[f1, em]`).
  `bleu4` is rejected for VQA.

## `vimacsa` - `ViMACSATask` (`vivlmoe/tasks/vimacsa.py`)

Fig. 4c and decision 18 (legacy `train_macsa.py` protocol).

- Data: `build_vimacsa(image_hw=(crop, crop), max_images=task.max_images)` -> zero-padded
  `pixel_values [B, M, 3, H, W]` + `image_mask`; collator `MultiImageClsCollator(tokenizer,
  task.aspects, use_image_tags, cls_max_len, legacy_parity)` builds one `"<aspect> </s></s>
  <review>"` sequence per aspect (+ ResNet-152 image tags as the second segment). Reviews should be
  pre-normalised with `text_normalize: vimacsa_legacy` for parity with the legacy code.
- Model: `MultiImageClassifier` encodes the images once per sample, replicates them per aspect,
  pools (`model.classifier.pool`: `mean_joint` default, `concat_mean_v_t` for legacy parity) and
  returns `logits [B, A, 4]` and `loss = sum over aspects of the batch-mean CE`.
- Validation: `task.metrics` must be `[macro_f1]`; `model.classifier.num_aspects` must equal
  `len(task.aspects)` (the downstream config sets it through `model_overrides`).
- Metrics: `macro_f1`, `macro_p`, `macro_r`, `accuracy`, `f1/<aspect>`.
  `task.exclude_none_in_f1: false` keeps the `None` label in the per-aspect macro average
  (legacy protocol); `true` averages over the three polarities only.

## Evaluators

### `GenerationEvaluator` (`vivlmoe/tasks/evaluators/generation.py`)

Per shard batch (unwrapped model, `eval()`, `no_grad`, autocast on CUDA):
`ids = model.generate(batch, mode=..., **generation_kwargs(task.generation))` with
`num_beams`, `max_new_tokens`, `min_new_tokens`, `length_penalty`, `early_stopping`,
`no_repeat_ngram_size`, `repetition_penalty`, `do_sample=False`; ids are decoded with the decoder
tokenizer (`skip_special_tokens=True`). The paper only states beam 5, so the downstream configs
use HF-parity values for everything else (`min_new_tokens: 0`, `early_stopping: false`,
`length_penalty: 1.0`); the schema defaults (`min_new_tokens: 1`, `early_stopping: true`) differ
and are written verbatim into `metrics/*.json` (`generation`), so a run's settings are traceable
either way. The smoke configs keep `min_new_tokens: 1` so a random tiny decoder always emits a
token. `(prediction, references, prompt, image_id)` are gathered
by `uid` with `gather_ordered` - a collective called on every rank, even for an empty shard - and
rank 0 computes `vivlmoe.metrics.compute_generation_metrics(task.metrics, ...)` with
`MetricTokenizer(task.metric_tokenizer)` (BLEU uses `task.bleu_tokenizer`, paper-compatible
`13a` by default) and, when `task.report_paper_compat_rouge`, adds `rougeL_paper_compat`.
Extra keys: `num_samples`, `pred_len_mean`, `bleu4_bp`, `bleu4_len_ratio`.

### `ClassificationEvaluator` (`vivlmoe/tasks/evaluators/classification.py`)

`logits.argmax(-1)` per (sample, aspect) and the gold labels are gathered in `uid` order with
`gather_ordered_tensors` (two collectives on every rank); rank 0 runs
`vivlmoe.metrics.aspect_macro_f1` (macro-P/R/F1 over the labels per aspect, mean over aspects).

### Prediction and metrics dumps (`vivlmoe/tasks/evaluators/dump.py`)

When `EvalContext.save_predictions` (`eval.save_predictions`, default true):

- `<run_dir>/predictions/<split>_step<N>.jsonl` - one object per sample in `uid` order, written
  with `ensure_ascii=False` (Vietnamese stays readable). Generation rows:
  `{uid, image_id, prompt, prediction, references}`; classification rows:
  `{uid, pred: {aspect: label}, gold: {...}, pred_ids, gold_ids}`.
- `<run_dir>/metrics/<split>_step<N>.json` - the metric dict plus provenance: `metric_names`,
  `tokenizer_signatures` (metric tokenizer, BLEU tokenizer, ROUGE aggregator), the full
  `generation` config, the decoder tokenizer class, `scales`; classification adds `aspects`,
  `labels`, `exclude_none_in_f1`.

### Metric scales

Metrics are reported **raw**, exactly as produced by `vivlmoe.metrics`; nothing is multiplied by
100:

| metric | scale |
|---|---|
| `bleu4`, `rougeL`, `rougeL_paper_compat`, `f1`, `em`, `macro_f1`, `macro_p`, `macro_r`, `accuracy` | fraction in `[0, 1]` (x100 for paper tables) |
| `cider` | CIDEr-D, `[0, 10]` |
| `bleu4_bp`, `bleu4_len_ratio` | brevity penalty / hypothesis-to-reference length ratio |
| `pred_len_mean` | mean number of metric-tokenizer tokens per prediction |

`rougeL` uses the configured tokenizer (`vi_safe`: NFC, lower-case, punctuation stripped, all
Unicode letters kept) and a plain per-sample mean; `rougeL_paper_compat` mirrors HF `evaluate`
(`rouge_score`'s default tokenizer, which drops Vietnamese letters, plus the bootstrap aggregator)
and exists only to compare with the paper's tables.

## Configs

`configs/downstream/_downstream_defaults.yaml` (included by every dataset file): AdamW lr 3e-5,
wd 0.01, cosine with 10 % warmup, `data.batch_size: 8` (x 4 GPUs = 32), 10 epochs, early
stopping patience 3 on `eval.selection_metric`, eval + checkpoint every epoch, `final_splits:
[test]`, `analysis.routing_probe: {split: dev, every_steps: 500, max_steps: 5000}` (run by the
Trainer on its own step cadence, see the table above, so the epoch-end evaluation steps need not
be multiples of 500).

```yaml
model: {from_checkpoint: runs/moe_sparse/checkpoints/best.pt}   # inherits topology + stabilisers
init_from: {path: runs/moe_sparse/checkpoints/best.pt, load: [all]}
```

`model.from_checkpoint` replaces the whole model section with the checkpoint's (so the file must
**not** include a `configs/model/*.yaml`; extra model keys go into `model_overrides`, as
`vimacsa.yaml` does for the classifier head). Stabiliser switches (`loss.router.use_*`,
`model.moe.aux_free_bias`) must match the checkpoint unless `task.allow_stabilizer_override:
true` (paper ablations disable a stabiliser in sparse training *and* downstream). For the dense
baseline point both paths at `runs/stage3_end2end/checkpoints/best.pt`.
`configs/_base/data_paths.yaml` is not included (its partial retrieval entries only validate with
`configs/data/evals_retrieval.yaml`); dataset roots live in each file.

### Downstream runs from ablation checkpoints

`configs/ablations/downstream_{no_zloss,no_miloss,no_auxfree_bias,switch_lb}.yaml` include
`../downstream/ktvic_caption.yaml`, point `model.from_checkpoint` / `init_from.path` at
`runs/ablation_<x>/checkpoints/best.pt` and **restate** the switches of that ablation
(`loss.router.use_z: false`, `use_mi: false`, `model_overrides.moe.aux_free_bias: false`,
`use_lb: true` + `lambda_lb: 0.01`). They deliberately do not set
`task.allow_stabilizer_override`, so the loader's guard compares the restated switches with the
checkpoint sidecar and a mismatch (e.g. pointing the overlay at a fully-stabilised checkpoint) is
refused instead of silently re-enabling a stabiliser. `_include` cannot be changed from the
command line: for another dataset copy the overlay and swap the include
(`../downstream/openvivqa.yaml`, ...).

| config | task | metrics | generation | notes |
|---|---|---|---|---|
| `ktvic_caption.yaml` | captioning | `[bleu4, rougeL]` | beam 5, 40 tokens | train = KTVIC train minus the committed dev holdout (`configs/data/splits/ktvic_dev_image_ids.json`), dev = that holdout, test = official public test |
| `uitviic_caption.yaml` | captioning | `[bleu4]` | beam 5, 40 tokens | same scheme with `uitviic_dev_image_ids.json` |
| `vimacsa.yaml` | vimacsa | `[macro_f1]` | - | 6 aspects, `use_image_tags`, `max_images 6`, `text_normalize: vimacsa_legacy`, `cls_max_len 170`, 20 epochs, `image_aspect_json` option |
| `vqa_generic.yaml` | vqa | `[rougeL]` | beam 5, 24 tokens | base for the overlays (`max_source_len 64`, `max_target_len 32`) |
| `openvivqa.yaml` | vqa | `[rougeL, cider]` | 64 tokens | dict-of-dicts annotations + image table |
| `vitextvqa.yaml`, `viocrvqa.yaml` | vqa | `[f1, em]` | 24 tokens | COCO-like annotations/images tables |

The VQA layouts (`data.*.options` of the `vqa_generic` adapter: `format`, `records_path`,
`id_key`, `image_key`, `question_key`, `answer_key` / `answers_key`, `image_lookup`) are the
layouts assumed in plan 8.2 - **verify them against the real files** before a run; the roots under
`/data/...` are placeholders. KTVIC / UIT-ViIC roots point at this machine's checkouts (the same
paths as `configs/_base/data_paths.yaml`).

Smoke configs `configs/_smoke/{caption,vqa,vimacsa}_tiny.yaml` train the tiny random model
(`configs/model/tiny.yaml`) on the synthetic kinds (`synthetic_caption`, `synthetic_vqa`,
`synthetic_vimacsa`) on CPU: 4 optimizer steps, accumulation 2, batch 2, eval every 2 steps,
beam 2 with 4 new tokens; they do not use `from_checkpoint`, so they run standalone:

```bash
python scripts/train.py --config configs/_smoke/caption_tiny.yaml
```

## `scripts/evaluate.py`

```bash
python scripts/evaluate.py --config configs/downstream/ktvic_caption.yaml \
    --checkpoint runs/ktvic_caption/checkpoints/best.pt --splits dev,test [--limit 8] \
    [--out-dir DIR] [key.path=value ...]
torchrun --nproc_per_node 4 scripts/evaluate.py ...     # sharded evaluation, same outputs
```

Builds the task and its model, applies the run's freezing (dtype policy), loads the checkpoint
strictly with `load_model_state` (provenance-aware: a frozen pristine tower excluded from the
checkpoint is re-materialised from the HF weights), builds the eval loaders (`--limit N` becomes
`data.evals.<split>.limit=N`), runs the task's evaluators and prints a `split | metric | value`
table on rank 0. Outputs under `--out-dir` (default `<run.dir>/<run.name>/eval/<checkpoint stem>`):
`predictions/<split>_step<N>.jsonl`, `metrics/<split>_step<N>.json`, `metrics/summary.json`
(`N` = the checkpoint's step). Unknown splits are a `KeyError`.

## `scripts/sweep_seeds.py`

```bash
python scripts/sweep_seeds.py --config configs/downstream/ktvic_caption.yaml --seeds 0 1 2 3 4 \
    [--nproc 4] [--dry-run] [key.path=value ...]
```

Runs the config once per seed as a sequential subprocess (`torchrun --standalone
--nproc_per_node N scripts/train.py` when `--nproc > 1`, else `python scripts/train.py`) with
`run.seed=<s> run.name=<name>_seed<s>`; the first failure aborts the sweep.
`--vary-upcycle-seed` (plan decision 21, upcycling-seed ablation) first runs
`python scripts/upcycle.py --config <--upcycle-config, default configs/moe/upcycle.yaml> --seed <s>
--out <run.dir>/<name>_upseed<s>/upcycled.pt [--upcycle-args "..."]` and trains with
`model.from_checkpoint=<that file> init_from.path=<that file>` appended to the overrides; the
config must inherit its model from an upcycled checkpoint (`configs/moe/sparse_train.yaml`). The
summary records `vary_upcycle_seed` / `upcycle_config`. The runs'
`final_metrics.json` (`final/*`, `last_eval/*` as `last/*`, `best_metric`) are aggregated into
`<run.dir>/<name>_sweep/sweep_summary.json` (`{metric: {mean, std, values}}`, std with `ddof=1`)
and `sweep_summary.md` (Markdown table). `--dry-run` prints the commands and exits.

## Tests

- `tests/unit/tasks/test_{captioning,vqa,vimacsa}_task.py`: two optimizer steps + one eval through
  the Trainer with the tiny smoke configs (finite loss, metric keys, JSONL with Vietnamese
  preserved, checkpoint + sidecar), wiring checks (freeze preset, selection metric, batch keys,
  metric validation).
- `tests/unit/tasks/test_evaluators_generation.py`: `GenerationEvaluator` with an echo model
  (ordering by uid, dump contents, generation kwargs), `ClassificationEvaluator` against
  `aspect_macro_f1`, and a 2-process gloo gather with an empty shard (`-m gloo`).
- `tests/unit/tasks/test_evaluate_script.py`: `evaluate.py` in-process on a tiny caption
  checkpoint with `--limit 4`, `sweep_seeds.py --dry-run` + aggregation (incl.
  `--vary-upcycle-seed`), every downstream config loading through a fake MoE checkpoint sidecar
  (incl. the stabiliser guard and the HF-parity generation settings), and the four
  `downstream_*` ablation overlays against matching / mismatching sidecars.
  `test_captioning_task.py::test_routing_probe_attached_for_moe_regardless_of_eval_cadence`
  covers the probe wiring with epoch-based evaluation.
- `tests/integration/test_cli_smoke_downstream.py` (`-m slow`): subprocess chain `train.py` ->
  `evaluate.py` -> `sweep_seeds.py --dry-run` for the three smoke configs.
- `test_captioning_step_on_real_ktvic_with_small_towers` (`-m hf_cache`): one optimizer step on
  8 real KTVIC captions with `configs/model/small.yaml` towers and the local `models/vit5-base`.
