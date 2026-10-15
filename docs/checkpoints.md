# Checkpoints

`vivlmoe/training/checkpoint.py` (format_version 1). Files under `<run_dir>/checkpoints/`:
`step_<N>.pt` (kept: `checkpoint.keep_last_n`), `last.pt` (symlink to the newest step file),
`best.pt` (on selection-metric improvement). Each `.pt` has a `<name>.pt.meta.json` sidecar
(config, provenance, excluded modules, fingerprint, step, metrics) read by `model.from_checkpoint`
without touching tensors. Writes are atomic (temp file + `os.replace`).

## Contents

```
format_version, model (unwrapped state_dict minus excluded prefixes), excluded_modules,
tower_provenance {<tower path>: 'hf:<id>' | 'trained'}, model_fingerprint,
config, optimizer (policy below), scheduler, scaler, train_state, task_state, evaluator_state,
rng (per rank), metrics (last eval), created, world_size, step, epoch, dev_split_sha1, transform_spec
```

Tower keys are full dotted paths: `vision_encoder` / `text_encoder` on a backbone,
`backbone.vision_encoder` / `backbone.text_encoder` on a downstream head (found by walking
`named_modules()`). Tower tensors are always stored in fp32, even when the frozen-dtype policy
holds the ViT in bf16, so a `trained` ViT is never rounded on disk. `model_fingerprint` hashes
`name|shape` only (dtype-agnostic), so a bf16 frozen tower and its fp32 counterpart share it.

`checkpoint.save_optimizer`: `always` (also in best.pt), `last_only` (step files only; default),
`never`.

## Provenance and exclusion

A tower starts as `hf:<name>` (pretrained) or `trained` (random init, `pretrained: false`). It
becomes `trained` when it is trainable in the run or was `trained` in the `init_from` checkpoint.
With `checkpoint.exclude_pristine_frozen_towers: true` a tower is dropped from `model` only if it is
frozen in this run **and** still `hf:`. Stage 3 trains the ViT, so from then on every checkpoint
carries it.

## Loading

- `--resume auto|<path>` (same run): model fingerprint must match; restores model, optimizer,
  scheduler, scaler, train/task/evaluator state and per-rank RNG (skipped with a warning if the
  world size changed). `best.pt` cannot be resumed from (no optimizer).
- `init_from: {path, load: [prefix, ...] | [all]}` (stage handoff): strict per prefix — every model
  key under the prefix must exist in the checkpoint with the same shape, and no extra keys may be
  present. An excluded tower is accepted only when the checkpoint's provenance is `hf:<id>` and the
  current model uses the same id (logged re-materialisation); anything else is a `CheckpointError`.
- `load_model_state` performs the same strict diff for `all`.
- Backbone → head handoff (`checkpoint_key_prefix`): when the model has a `backbone` submodule and
  every top-level checkpoint key is a child of it, checkpoint keys, `excluded_modules` and
  `tower_provenance` are remapped under `backbone.` (logged). Prefixes in `init_from.load` may use
  the checkpoint namespace (`encoder`) or the model's (`backbone.encoder`); `all` then covers the
  backbone only and the head's own modules (decoder, bridge, cls_head) keep their initial weights.
- `--resume auto` with a dangling `last.pt` symlink (target pruned) is a `CheckpointError`, never a
  silent restart.
