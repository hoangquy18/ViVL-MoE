# Mixture-of-Experts (paper Sec. 5, milestone M6)

## Layer (`vivlmoe/models/moe/`)

`encoder.py` swaps every dense FFN for `MoEFFN(dim, ffn_dim, dropout, moe_cfg, layer_idx)` when
`model.moe` is set. State-dict keys:
`encoder.layers.{i}.ffn.{shared_experts.{s},routed_experts.{e}}.{fc1,fc2}.{weight,bias}` and
`encoder.layers.{i}.ffn.router.{proj.weight,expert_bias}`.

* **Expert** = `DenseFFN` (fc1 → GELU → Dropout(`encoder.ffn_inner_dropout`) → fc2), the same class
  as the dense FFN, so upcycling is an exact copy; the inner dropout is what separates identical
  shared experts during training.
* **Router** (`router.py`, plan decision 5): `proj = Linear(dim, N_r, bias=False)` initialised
  `N(0, moe.router_init_std²)` (`ViVLBackbone.init_new_weights` skips modules named `router`).
  Forward on the `T` valid tokens with autocast disabled: `z = x.float() @ Wᵀ`,
  `S(x) = Top-k(z + b)` (Eq. 21, `b = expert_bias`, only when `aux_free_bias`),
  `p~ = softmax(z[S(x)])` (Eq. 15, from `z` not `z + b`). Returns a fp32 `RouterOutput(layer_idx,
  logits, topk_idx, topk_probs, modality_ids)`. `dispatch_counts [N_r]` / `token_count` (int64,
  non-persistent) accumulate only in training mode via `one_hot().sum()`; the bias is never touched
  in `forward`.
* **Dispatch** (`dispatch.py`, decision 6): one stable `argsort` of the `[T·k]` assignments, one
  contiguous slice per expert, `index_add_` of the gate-weighted outputs in fp32 (double inputs stay
  double so `torch.autograd.gradcheck` can run on it). **Idle experts are still called on an empty
  `[0, d]` slice**, which keeps every expert parameter in the autograd graph
  (DDP with `find_unused_parameters=False`, no `broadcast_buffers`).
* **Eq. 18** (`moe_ffn.py`): `h = ½[(1/N_s)Σ_s E_s^sh(x) + Σ_{i∈S(x)} p~_i E_i^rt(x)]`, or `½·routed`
  when `N_s = 0`. Padded tokens are never routed (zeros are scattered back), output cast to the
  block dtype.

## Aux-free balancing (`balancing.py`, Eq. 22)

`aux_free_bias_step(encoder, ctx, eta)` — called by the Task on every rank in `post_optimizer_step`
(`moe.bias_update_every: step`, default) or at the end of every micro-batch (`micro`): packs
`encoder.dispatch_stats()` (`[L, N_r+1]`), **one** `all_reduce_sum_` through
`vivlmoe.training.distributed`, `f_i = counts_i / max(tokens, 1)`,
`b_i += η·sign(k/N_r − f_i)` (`encoder.apply_router_bias_update`), reset. When no token was routed
since the last reset (e.g. a step whose counters were cleared) the biases are left untouched with a
warning — `f = 0` is not a load signal and would shift every bias by `+η`. Returns
`moe/expert_frac_max|min`, `moe/bias_range`, `moe/tokens` (+ per-layer with `per_layer_logs=True`).
The counters of the very first micro-batch feed the first update like any other (no reset in
`after_first_backward`: the Trainer has no separate startup forward).
Because the update is deterministic on all-reduced counts, `expert_bias` stays bit-identical across
ranks and is saved as a persistent buffer; counts are not saved.

## Router losses (`vivlmoe/losses/router_losses.py`, Eq. 19/20/23/24)

`RouterLossComputer(cfg.loss.router)(router_outputs) -> (aux, logs)` groups the outputs by layer
(contrastive mode: visual pass + text pass of the same layer; joint mode: one output with mixed
modality ids — the two give identical `L_mi` for the same tokens), computes per layer
`L_z = mean_t logsumexp(z_t)²`, `P_m ∝ Σ_{M(t)=m} p~(x_t)` with `p~ = 0` for unselected experts,
`L_mi = ½[KL(P_v‖P_t) + KL(P_t‖P_v)]` (0 when a modality is absent, e.g. captioning), and the
Switch load-balancing loss `N_r Σ f_i P_i` (ablation arm, only computed under `use_lb`; `f_i` is the
fraction of *tokens* dispatched to expert `i`, `Σ f_i = k`, so the loss equals `k` under uniform
routing — the canonical Switch normalisation, `lambda_lb: 0.01` in `switch_lb.yaml` is on that
scale). Terms are reduced
over layers by `layer_reduction` (`mean` default) and weighted by `lambda_*` under the `use_*`
switches. `logs = {loss_z, loss_mi, loss_lb}` are unweighted. Tasks add `aux` to their loss.

## Upcycling (`vivlmoe/upcycling/`, `scripts/upcycle.py`)

`upcycle_state_dict(dense_sd, moe_cfg, noise_std_rel=0.01, noise_on_bias=False, seed=0,
router_init_std=None)`: every dense FFN key (regex `^encoder\.layers\.(\d+)\.ffn\.(fc1|fc2)\.(weight|bias)$`)
is copied exactly into the shared experts and, with noise `N(0, (noise_std_rel·std(W))²)` on the
weights, into the routed experts; `router.proj ~ N(0, router_init_std²)`, `expert_bias = 0`; every
other tensor verbatim. Seeds are `zlib.crc32(f"{seed}/{layer}/{expert}/{tensor}")` (never Python
`hash()`), reproducible across processes. `upcycle_model(dense, moe_cfg, ...)` builds the MoE
backbone from `dense.cfg` and loads with `strict=True`; `verify_equivalence(dense, moe, batch,
atol)` compares eval-mode contrastive embeddings and encoder outputs (valid tokens) and raises
`UpcycleError` above `atol` — with noise 0 and `N_s ≥ 1` the difference is < 1e-5; with `N_s = 0` the
FFN contribution is halved by design.

```
python scripts/upcycle.py --config configs/moe/upcycle.yaml --out runs/moe_upcycled/upcycled.pt \
    [--dense PATH] [--noise-std-rel 0.01] [--seed 0] [--verify-atol 1e-5] [model_overrides.moe.num_shared=0]
```

The config shape is `model: {from_checkpoint: <dense>}` + `model_overrides: {moe: {...}}` (the
loader inherits the dense model section from the checkpoint sidecar; other `model` keys are not
allowed next to `from_checkpoint`). The schema has no `upcycle` section, so noise/seed/output are CLI
flags. The script refuses to save unless the noise-0 equivalence check passes, then writes a full
checkpoint (no optimizer) with `upcycle_info` and the inherited `tower_provenance`, and logs the
parameter counts (paper: 1.29B → 1.75B).

## Configs

* `configs/moe/upcycle.yaml`, `configs/moe/sparse_train.yaml` (contrastive task, `from_checkpoint` +
  `init_from` = the upcycled checkpoint, `freeze_preset: sparse`, lr 1e-5, 128 × accum 4, 20K steps,
  eval/probe every 500 steps, checkpoints every 1000 with `keep_last_n: 3`; the data files
  `configs/data/{pretrain_corpus,evals_retrieval}.yaml` are shared with the dense stages).
* `configs/ablations/`: `ns0..ns5`, `nr1_k1` (`top_k: 1`; `MoECfg` rejects `top_k > num_routed`),
  `nr2..nr8` — each needs its own upcycled checkpoint (command in the file header);
  `no_zloss`, `no_miloss`, `no_auxfree_bias`, `switch_lb` set `task.allow_stabilizer_override: true`
  because the loader otherwise refuses stabilizer switches that differ from the checkpoint (paper:
  a removed stabilizer stays off downstream too — carry the same overrides into the downstream run).
* `configs/_smoke/upcycle_tiny.yaml` (dense `runs/_smoke/stage3_tiny/checkpoints/best.pt` → 1 + 4
  experts, top-2) and `configs/_smoke/moe_tiny.yaml` (4 sparse steps on synthetic pairs, eval every
  4 steps, probe every 2 on its own cadence): the CPU chain stage3_tiny → upcycle → moe_tiny → analyze_routing
  (`tests/integration/test_cli_smoke_moe.py`, marked `slow`, builds its own tiny dense checkpoint).

## Routing analysis (`vivlmoe/analysis/routing_stats.py`, `tasks/evaluators/routing.py`)

`analysis.routing_probe: {split, every_steps: 500, max_steps: 5000, max_batches}` adds a
`RoutingProbeEvaluator` to the split's evaluator (`maybe_add_routing_probe` wraps it in a
`CompositeEvaluator`). **Cadence**: the Trainer runs the probe on its own step cadence
(`Trainer._run_routing_probe`, every `every_steps` up to `max_steps`, logged as
`{split}/routing/...` and appended to `eval_history.jsonl`), independently of `eval.every_steps` /
`every_epochs` — downstream runs evaluate per epoch (≈471 steps on KTVIC) and still get the probe at
500, 1000, …, 5000. When a full evaluation falls on a probe step the composite runs it once (the
probe returns `{}` when not due). `maybe_add_routing_probe` warns when `every_steps` is not a
multiple of `eval.every_steps`. Every due step it runs the task's forward mode (`contrastive` =
separate visual/text passes; `joint_caption` / `joint_vqa` / `joint_cls`) over the dev loader,
gathers the per-token top-k sets in dataset order (`gather_ordered`, on every rank) into a
`RoutingSnapshot` kept in `state_dict` (survives `--resume`), and reports:

* `routing/change_rate` (+ `_visual`, `_text`, per layer): fraction of tokens whose top-k **set**
  changed since the previous snapshot (`expert_change_rate`);
* `routing/sym_kl` on the layer-averaged expert × modality heatmap (Eq. 23 numerators, columns
  sum to 1) and `sym_kl_layer_mean`; per-layer `sym_kl`;
* `routing/margin_full` (top1 − top2 of `softmax(z)`, bias-free; default) and `margin_topk`
  (`p~₁ − p~₂`), by modality and per layer;
* `expert_frac_max|min`, `bias_range`.

Files: `run_dir/routing/step_<N>.json`, `step_<N>_heatmap.png` (matplotlib, optional),
`snapshot_step_<N>.pt`. Eval forwards never touch the dispatch counters.

**Fixed tokens**: `max_batches` bounds the probe by a *global* uid range
(`max_batches × eval_batch_size` dataset uids, taken before sharding), so the probed token stream —
and therefore the change rate — does not depend on the number of processes; `data.evals.<split>.limit`
bounds the split the same way. Snapshots are stored compactly (`int16` expert ids, `int8` modality)
since the previous one travels in every checkpoint's `evaluator_state`. If a restored snapshot covers
a different token stream (split, limit or `max_batches` changed between runs) the probe logs a
warning and restarts the change-rate series instead of aborting training.

```
python scripts/analyze_routing.py --config configs/moe/sparse_train.yaml \
    --checkpoint runs/moe_sparse/checkpoints/best.pt --split ktvic_dev \
    [--max-batches 20] [--prev-snapshot runs/moe_sparse/routing/snapshot_step_0001000.pt] [--out-dir DIR]
```

## Tests (M6, default-selected unless noted)

* `tests/unit/models/test_moe_router.py` — selection uses the bias but `p~` does not, `MoECfg`
  top-k validation, counters only in training, fp32 `RouterOutput` under CPU bf16 autocast, padded
  tokens never routed, Eq. 18 incl. the `N_s = 0` half rule, router init std after backbone
  construction, state-dict keys, MoE buffers through `build_checkpoint` / `load_model_state`.
* `tests/unit/models/test_moe_dispatch.py` — naive-loop parity, `gradcheck` in double, idle expert
  gradients. `test_moe_balancing.py` — Eq. 22 sign rule, load redistribution, edge cases.
  `test_moe_gloo.py` (`gloo`) — DDP + `no_sync` accumulation with an idle expert, identical grads and
  biases across ranks, eval leaves the counters untouched. `test_moe_configs.py` — every
  `configs/moe`, `configs/ablations` and MoE `_smoke` config resolves against a sidecar.
* `tests/unit/losses/test_router_losses.py` — Eq. 19/20/23/24 values, contrastive ≡ joint `L_mi`,
  layer mean vs sum, flags/lambdas, router gradients change with `lambda_mi`.
* `tests/unit/upcycling/` — key mapping, exact copies + noise statistics, crc32 seeds across
  `PYTHONHASHSEED` (subprocess), equivalence for `N_s ∈ {1, 2, 3}`, half rule for `N_s = 0`.
* `tests/unit/analysis/` — change rate / heatmap / sym-KL / margins / load collectors and the
  `RoutingProbeEvaluator` (files, resume via `state_dict`, split loaders, joint modes, composite).
