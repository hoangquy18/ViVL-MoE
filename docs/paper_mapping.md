# Paper -> code mapping

*One Tower, Many Experts: Benchmarking and Scaling Sparse Vision-Language Models for Vietnamese* (ViVL-MoE).
Started at milestone M3 (dense backbone + losses); later milestones append their rows (MoE / router
losses / upcycling / downstream tasks / metrics).

## Dense backbone (Sec. 4.2, Fig. 3)

| Paper | Code | Notes |
|---|---|---|
| `V = f_v(I)` frozen CLIP ViT, raw last hidden state | `vivlmoe/models/towers/vision.py::VisionTower` | `apply_post_layernorm: false` (default) returns the pre-`post_layernorm` tokens; the unused `post_layernorm` is then kept permanently frozen so a trainable ViT (Stage 3) has no parameter outside the graph. Frozen tower: eval kept via `train()` override, `set_grad_enabled`, reduced-precision weights on CUDA (`frozen_dtype: auto` -> the AMP dtype, bf16 by default), fp32 on CPU; trainable: non-reentrant gradient checkpointing. The weight dtype changes only in `VisionTower.apply_dtype_policy` (via `set_trainable` or `freezing.apply_frozen_dtype_policy(model, amp_dtype)` after `model.to(device)`), never inside `forward`, so the checkpoint fingerprint is stable across the first step. |
| `W_v: d_v -> 768` | `vivlmoe/models/projections.py::VisionProjection` | paper `[768, 1280]`, 730 tokens (`VisionTower.assert_dims`) |
| Perceiver resampler `R_v`, 64 queries, 2 layers, 8 x 64 | `vivlmoe/models/resampler.py::PerceiverResampler` | pre-norm residual, `to_q 768->512`, `to_kv 768->1024`, `to_out 512->768`, FFN 4x GELU, final LN; keys/values = projected patch tokens only (`concat_latents_to_kv` opt-in) |
| `T = f_t(c)` XLM-R | `vivlmoe/models/towers/text.py::TextTower` | `add_pooling_layer=False` |
| `W_t: d_t -> 768` | `projections.py::TextProjection` | |
| modality embeddings `e_v`, `e_t` | `projections.py::ModalityEmbedding` (`modality_embed`, row 0 = visual) | also added in the single-modality contrastive passes (`model.modality_embed_in_single_pass: true`, plan decision 10) |
| `f_enc`: 12 post-LN blocks (MHSA -> Add&LN -> FFN -> Add&LN), 768 / 12 heads / 3072 GELU, dropout 0.1 | `vivlmoe/models/block.py::TransformerBlock`, `attention.py::MultiHeadSelfAttention`, `ffn.py::DenseFFN`, `encoder.py::MultimodalEncoder` | `DenseFFN = fc1 -> GELU -> Dropout(ffn_inner_dropout) -> fc2` is also the expert unit (plan decision 4); no positional embeddings in `f_enc` (assumption) |
| joint input `X = [V~ + e_v ; T~ + e_t]` | `backbone.py::ViVLBackbone.forward_joint` | used by the downstream heads |
| random init of `W_v`, `W_t`, `R_v`, `f_enc`, `e_*` | `ViVLBackbone.init_new_weights` (`projections.py::init_module_`) | Normal(0, `model.init_std`=0.02); towers untouched |

## Contrastive objective (Sec. 4.2, Eq. 13)

| Paper | Code | Notes |
|---|---|---|
| separate single-modality passes, masked mean pooling | `ViVLBackbone.forward(batch, "contrastive")` -> `forward_single`, `masked_mean` | one `forward` per micro-batch (plan decision 2); embeddings fp32 L2-normalised |
| temperature `tau` (init 0.07) | `ViVLBackbone.logit_scale` = `log(1/tau)`; `scaled_logit_scale()` | clamped at `ln(model.logit_scale_max)` = ln 100 |
| symmetric InfoNCE with global-batch negatives | `vivlmoe/losses/contrastive.py::ContrastiveLoss` | grad-flowing `torch.distributed.nn.functional.all_gather`, local labels `arange(B) + rank*B`; **deviation:** `symmetric_reduction: mean` = 0.5 x (CE_i2t + CE_t2i) by default (calibration of the paper's lambdas; `sum` available) |
| 2048 negatives per InfoNCE step (Stage 2: 256 x 8 GPUs; sparse: 512 x 4) | `configs/pretrain/stage2_contrastive.yaml` (64 x 4 GPUs = 256), `stage3_end2end.yaml` (32 x 4 = 128), `configs/moe/sparse_train.yaml` (128 x 4 = 512) | **deviation:** negatives are the per-step global batch `data.batch_size x world` (plan decision 15); `grad_accum_steps` (8 / 16 / 4) recovers the 2048 optimizer batch but not the negatives, so the Eq. 13 objective is easier and R@K may differ. Logged at startup by `ContrastiveTask` (`contrastive negatives per step = B x world`); on the A40s prefer `grad_accum_steps: 1` with the largest per-GPU batch (frozen-ViT feature cache, M9) |

## Stage 1 teacher learning (Sec. 4.1, Eq. 12)

| Paper | Code | Notes |
|---|---|---|
| frozen CLIP text encoder `z_en` | `vivlmoe/models/towers/clip_teacher.py::CLIPTextTeacher` | owned by the task, not the student; `teacher_target: projected` (CLIP joint space, Appendix A) or `eos_hidden` |
| student XLM-R `<s>` -> FC1 -> `z_vi` | `vivlmoe/models/stage1.py::Stage1Student` (`text_encoder`, `fc1`) | `fc1_hidden` gives an optional 2-layer FC1 (paper: single Linear) |
| `L_1 = (1/N) sum ||z_vi - z_en||^2` | `vivlmoe/losses/teacher.py::TeacherMSELoss("eq12")` | `legacy` preset = normalise + `F.mse_loss` (legacy code) |
| Appendix-A dev retrieval in CLIP space | `clip_teacher.py::CLIPImageTeacher` | image embeddings cached once by the task |

## Training stages / freezing (Sec. 4.1-4.3, 5.4, 6.1)

| Paper | Code |
|---|---|
| Stage 1: train XLM-R + FC1 | `vivlmoe/models/freezing.py::FREEZE_PRESETS["stage1"] = []` |
| Stage 2: freeze ViT and `W_v` | `FREEZE_PRESETS["stage2"] = [vision_encoder, vision_proj]` |
| Stage 3: unfreeze everything | `FREEZE_PRESETS["stage3"] = []` |
| sparse pretraining: ViT frozen | `FREEZE_PRESETS["sparse"] = [vision_encoder]` |
| downstream: all trainable except ViT | `FREEZE_PRESETS["downstream_seq2seq" / "downstream_cls"]` (+ `logit_scale`, unused downstream); `downstream_caption` additionally freezes `text_encoder` / `text_proj`, which a text-free captioning batch never touches |
| unused-parameter hard check at startup | `freezing.py::assert_no_unused_trainable_params` |
| tower lookup for provenance / partial loads (nested `backbone.*` in heads) | `vivlmoe/models/towers/__init__.py::find_tower_paths` -> `{vision_encoder: 'backbone.vision_encoder', ...}` |

## Downstream heads (Sec. 6.1, Fig. 4)

| Paper | Code | Notes |
|---|---|---|
| ViT5 decoder on `Linear(768 -> 1024)` bridge; captioning = image only; VQA = image + question; CE with padded targets ignored | `vivlmoe/models/heads/seq2seq.py::Seq2SeqVLModel` (`backbone`, `bridge`, `decoder`) | T5 encoder stack removed (`decoder.encoder.block = ModuleList()`, remaining encoder-only params frozen); `generate(batch, num_beams=5, ...)` via `encoder_outputs` |
| ViMACSA: pooled joint representation -> classifier, per-aspect CE | `vivlmoe/models/heads/classifier.py::MultiImageClassifier` (`backbone`, `cls_head`) | images encoded once per sample (valid slots only), replicated per aspect; pooling `mean_joint` (default) / `mean_text` / `concat_mean_v_t` (legacy parity); loss = sum over aspects of batch-mean CE (legacy scaling) |

## Factory

`vivlmoe/models/factory.py::build_task_model(kind, cfg)` with `kind in {backbone, stage1, seq2seq, classifier}`,
`build_teacher`, `build_image_teacher`, `load_model_cfg(yaml)`.

## MoE seam (Sec. 5) - filled in at milestone M6

`MultimodalEncoder` builds `vivlmoe.models.moe.moe_ffn.MoEFFN` per layer when `model.moe` is set and exposes
`moe_layers()`, `dispatch_stats() -> int64 [L, N_r+1]`, `apply_router_bias_update(f, eta)` (Eq. 22 sign rule on
`router.expert_bias`), `reset_dispatch_stats()`, `router_bias_state()`. No collectives inside modules.

## Mixture-of-Experts (Sec. 5) — details in `docs/moe.md`

| Paper | Code | Notes |
|---|---|---|
| Eq. 14-15 router: `p = softmax(z)`, `p~` renormalised over `S(x)` | `vivlmoe/models/moe/router.py::Router` | fp32 with autocast disabled; `p~ = softmax(z[S])` from `z`, never from `z + b`; counts accumulate only in training |
| Eq. 21 selection `S(x) = Top-k(z + b)` | `Router.forward` | `b = expert_bias` (persistent buffer), used at selection only |
| Eq. 22 `b_i += eta * sign(k/N_r - f_i)` | `vivlmoe/models/moe/balancing.py::aux_free_bias_step` -> `MultimodalEncoder.apply_router_bias_update` | called once per optimizer step from `Task.post_optimizer_step` after ONE all-reduce of the packed `[L, N_r+1]` counts (`moe.bias_update_every: step`; `micro` updates per micro-batch); `eta = model.moe.bias_update_rate` (1e-3, not given in the paper) |
| Eq. 18 `h = 1/2 [ mean_s E^sh_s(x) + sum_{i in S} p~_i E^rt_i(x) ]` | `vivlmoe/models/moe/moe_ffn.py::MoEFFN.forward_block`, `dispatch.py::dispatch` | valid tokens only; `N_s = 0` -> half the routed path; idle experts still called on an empty slice (DDP) |
| Eq. 20 z-loss | `vivlmoe/losses/router_losses.py::z_loss` | `lambda_z = 1e-3` |
| Eq. 23-24 modality-importance loss (sym. KL of per-modality routed mass) | `router_losses.py::modality_distributions`, `sym_kl`, `RouterLossComputer` | `p~ = 0` for unselected experts; contrastive mode groups the visual-pass and text-pass outputs of the same layer; `lambda_mi = 5e-3`; **per-layer mean** (`loss.router.layer_reduction`) |
| Eq. 19 Switch load-balance loss (ablation arm) | `router_losses.py::switch_lb_loss` | `f_i` = fraction of tokens per expert (= `k` under uniform routing); `configs/ablations/switch_lb.yaml` |
| Eq. 25 total loss, stabilizers active downstream | `ContrastiveTask.train_step`, `DownstreamTask.finish_train_step` | aux added to every task loss; downstream configs inherit the switches from the checkpoint (`model.from_checkpoint`, stabilizer guard in the loader) |
| Sparse upcycling (Sec. 5.3) | `vivlmoe/upcycling/upcycle.py`, `scripts/upcycle.py` | shared experts exact copies, routed = copy + `N(0, (0.01 std W)^2)` (crc32-seeded), router `N(0, 0.02^2)`, bias 0; noise-0 equivalence verified before saving |
| Sec. 6.5 routing analysis (change rate every 500 steps, expert x modality heatmap + sym. KL, router margin) | `vivlmoe/analysis/routing_stats.py`, `vivlmoe/tasks/evaluators/routing.py::RoutingProbeEvaluator`, `scripts/analyze_routing.py` | own cadence in the Trainer (`analysis.routing_probe`); `margin_full` (softmax over `N_r`, no bias) is the reported margin, `margin_topk` is logged too |
| Ablations: `N_s in 0..5`, `N_r in 1..8` (`k = 1` for `N_r = 1`), each stabilizer removed, Switch-LB | `configs/ablations/*.yaml` (+ `downstream_*` overlays) | `nr1_k1.yaml` sets `top_k: 1` explicitly (`top_k > num_routed` is a `ConfigError`) |

## Downstream and metrics (Sec. 3.3, 6.1) — details in `docs/downstream.md`

| Paper | Code | Notes |
|---|---|---|
| ViT5-large decoder, beam 5, lr 3e-5, batch 32, 5 seeds | `vivlmoe/tasks/{captioning,vqa}.py`, `configs/downstream/*.yaml`, `scripts/sweep_seeds.py` | batch = 8 per GPU x 4; `max_new_tokens` 40 (captioning) / 24 (short-answer VQA) / 64 (OpenViVQA) |
| ViMACSA classification head, Macro-F1 | `vivlmoe/tasks/vimacsa.py`, `vivlmoe/metrics/classification.py::aspect_macro_f1` | legacy protocol: 6 aspects x {None, Negative, Neutral, Positive}, per-aspect macro over the 4 labels then mean (`task.exclude_none_in_f1` switch) |
| BLEU-4 (multi-reference) | `vivlmoe/metrics/bleu.py` (vendored HF `evaluate` BLEU) | tokenizer `13a` by default (paper-compat), `vi_safe` optional |
| ROUGE-L | `vivlmoe/metrics/rouge.py` | `vi_safe` tokenizer (keeps Vietnamese diacritics) **and** `rougeL_paper_compat` (rouge_score default tokenizer) both reported |
| CIDEr (OpenViVQA) | `vivlmoe/metrics/cider.py` (pycocoevalcap) | |
| token-F1 / EM (ViTextVQA, ViOCRVQA) | `vivlmoe/metrics/qa.py` | SQuAD-style with a Vietnamese normaliser |
| Appendix A retrieval R@K on KTVIC / UIT-ViIC dev | `vivlmoe/metrics/retrieval.py::recall_at_k`, `tasks/evaluators/retrieval.py` | dev = seeded 20 % holdout by image id (`configs/data/splits/*.json`, sha1 stored in checkpoints); I->T hit = any reference caption in the top-k |

## Documented deviations and assumptions (all configurable)

| Topic | Choice | Where |
|---|---|---|
| Contrastive reduction | `mean` = 0.5 x (CE_i2t + CE_t2i) (matches the code behind the paper's numbers); `sum` = literal Eq. 13 | `loss.contrastive.symmetric_reduction` |
| Contrastive negatives | per-step global batch (`batch_size x world`), not the accumulated 2048 | `docs/training.md`; prefer `grad_accum_steps: 1` with the largest per-GPU batch on the training box |
| Per-layer router losses | mean over the 12 layers | `loss.router.layer_reduction` |
| Bias update cadence / `eta` | once per optimizer step on all-reduced counts, `eta = 1e-3` | `model.moe.bias_update_every`, `bias_update_rate` |
| Expert dropout | inside the FFN (`fc1 -> GELU -> Dropout -> fc2`), also in the dense backbone | `model.encoder.ffn_inner_dropout` |
| Stage-1 teacher target | projected CLIP text embedding (CLIP joint space, Appendix A) | `model.stage1.teacher_target` |
| Modality embeddings in single-modality passes | yes (so `e_v`, `e_t` train from Stage 2) | `model.modality_embed_in_single_pass` |
| Vision tokens | raw last hidden state (`apply_post_layernorm: false`) | `model.vision.apply_post_layernorm` |
| Gradient clipping 1.0, no weight decay on norms / biases | legacy-code conventions | `train.max_grad_norm`, `optim.no_decay_norm_and_bias` |
| Generation defaults | `min_new_tokens: 1`, `early_stopping: true` (beam search stops when `num_beams` hypotheses finished) | `task.generation` (recorded in every `metrics/*.json`) |
| ViMACSA loss / pooling | sum over aspects of batch-mean CE; masked mean over the joint sequence (`concat_mean_v_t` = legacy head) | `model.classifier.pool` |
| Dev splits | KTVIC / UIT-ViIC dev = seeded 20 % holdouts of train. The official UIT-ViIC dev (924 images) is missing from the local copy; its ids are recovered from the corpus' `vilc_*` images and only used to exclude it from pretraining | `configs/data/splits/` |
| Stage-1 data | PhoMT, raw-text (`detokenization`) variant, first 2M of the 2.98M training pairs (`max_train_samples`); restored to `dataset/PhoMT` from the original PhoMT release because the legacy 20M-line training file was lost | `configs/pretrain/stage1_teacher.yaml` |
| Pretraining corpus | the legacy dump in `dataset/` (1.15M Vietnamese captions: COCO-vi, localization, ViOCRVQA, Oxford-102, EVJVQA, ViVQA, ...; 185K English captions) instead of the paper's COCO-vi / local / LAION-EN / crawled mix. Decontamination by name: the ViTextVQA / OpenViVQA / ViOCRVQA conversions are dropped whole (splits unknown) and the UIT-ViIC test, official-dev and dev-holdout COCO ids are dropped from coco2017 / vilc / vivqa; content duplicates under other names are not detected | `scripts/build_pretrain_corpus.py`, `vivlmoe/data/corpus_prep.py`, report next to the corpus |

## v2: changes beyond the paper (`docs/architecture_v2.md`)

v2 is an additive path: every v1 row above still holds when the new keys keep their defaults
(`vision.kind: clip`, `text.arch: xlm-roberta`, `stage1.teacher_kind: clip`, `contrastive_head: infonce`,
`generation.kind: t5`).

| Paper | v2 | Code / config |
|---|---|---|
| DFN CLIP ViT-H/14 @378, unfrozen in Stage 3 | C-RADIOv4-SO400M @512 (1024 tokens x 1152), frozen throughout; SigLIP 2 g / So400m as alternatives | `towers/vision_radio.py::RadioVisionTower`, `towers/vision_siglip.py::SiglipVisionTower`, `model.vision.{kind,image_size,revision}`, `data.transform.source: tower` |
| XLM-R large text tower | mmBERT-base (ModernBERT) | `towers/text.py::TextTower` (`model.text.arch: modernbert`) |
| Stage 1, Eq. 12: XLM-R `<s>` regresses the English CLIP text embedding on PhoMT | the student's masked-mean state regresses the SigLIP 2 g text embedding (normalised MSE, `loss_preset: legacy`) on corpus captions (caption -> same caption) + PhoMT | `towers/siglip_teacher.py`, `towers/teachers.py`, `stage1.py` (`student_pooling: mean`), adapters `caption_text` + named `mixture`, `configs/v2/distill_text.yaml` |
| Appendix-A dev retrieval in CLIP space | the same evaluator in the SigLIP 2 joint space (the teacher's image tower and preprocessing) | `tasks/stage1_teacher.py::_retrieval_loaders` |
| Eq. 13 symmetric InfoNCE, tau 0.07 | SigLIP sigmoid loss with a learnable bias (t = 10, b = -10 at init); texts all-gathered, `B_local` normalisation | `losses/contrastive.py::SigmoidContrastiveLoss`, `model.contrastive_head: sigmoid`, `ViVLBackbone.logit_bias` |
| Stages 2 and 3 (W_v frozen, then everything unfrozen) | one dense alignment step, vision frozen, W_v trainable from the start (preset `sparse`) | `configs/v2/align.yaml` |
| Upcycling + sparse training (Sec. 5) | unchanged | `configs/v2/{upcycle,sparse}.yaml` include `configs/moe/*` |
| ViT5-large decoder on the bridged joint states | Qwen3.5-0.8B fed `[projected joint states ; prompt ; answer]`, left-padded, loss on answer tokens only | `models/heads/causal_lm.py::CausalLMVLModel`, `heads/llm_loader.py`, `data/collators.py::CausalLMCollator`, `generation.kind: causal_lm`, `task.llm_prompt_template` |
