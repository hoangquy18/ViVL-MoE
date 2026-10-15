# ViVL-MoE

Clean, reviewable re-implementation of **One Tower, Many Experts: Benchmarking and Scaling Sparse
Vision-Language Models for Vietnamese** (ViVL-MoE): a dense Vietnamese CLIP-style backbone pretrained in
three stages, sparse-upcycled into a shared + routed Mixture-of-Experts encoder, and fine-tuned on the
ViMMU-Bench downstream tasks. One `Trainer`, YAML configs, native `torchrun` DDP, hermetic CPU tests.

```
Stage 1 teacher  ->  Stage 2 contrastive  ->  Stage 3 end-to-end  ->  upcycle  ->  MoE sparse training  ->  downstream
(PhoMT, Eq.12)       (image-text, Eq.13)      (ViT unfrozen)          (Sec.5.3)   (z-loss, L_mi, bias rule)   (ViT5 / classifier)
```

## Layout

```
configs/      _base (defaults, data paths, model paths) · model/{paper,small,tiny} · pretrain/ · moe/ · ablations/ · downstream/ · _smoke/
scripts/      fetch_models · train · run_pretrain.sh · upcycle · evaluate · analyze_routing · sweep_seeds · make_dev_split
              build_pretrain_corpus · hf_corpus · build_manifest · bench_loader · env_check
vivlmoe/      config/ data/ models/ losses/ upcycling/ analysis/ metrics/ training/ tasks/ utils/
tests/        unit/ (config, data, models, losses, training, tasks, metrics, upcycling, analysis) · integration/ (CLI smoke chains)
docs/         configs · training · distributed · checkpoints · tasks · moe · downstream · paper_mapping
```

* `vivlmoe/models` — CLIP ViT tower, XLM-R tower, Perceiver resampler, 12-layer post-LN shared encoder with a
  pluggable FFN (`DenseFFN` or `MoEFFN` = shared + routed experts, Eq. 18), ViT5 seq2seq head, ViMACSA classifier.
* `vivlmoe/losses` — symmetric InfoNCE with grad-flowing all-gather (Eq. 13), teacher MSE (Eq. 12), router z-loss /
  modality-importance / Switch-LB (Eq. 19-24).
* `vivlmoe/upcycling` — exact-copy sparse upcycling with seeded expert noise and an equivalence check.
* `vivlmoe/data` — dataset adapters (PhoMT, image-text corpora, KTVIC, UIT-ViIC, ViMACSA, generic VQA, synthetic),
  CLIP-consistent transforms, Vietnamese normalisation, samplers (one caption per image, ordered eval shards), collators.
* `vivlmoe/training` — DDP bootstrap, param groups / freezing, step-based cosine schedule, AMP, provenance-aware
  checkpoints, the `Trainer`.
* `vivlmoe/tasks` — one task per objective (`stage1_teacher`, `contrastive`, `captioning`, `vqa`, `vimacsa`) and the
  evaluators (retrieval R@K, generation BLEU/ROUGE/CIDEr/F1/EM, aspect macro-F1, routing probe).

## Setup

```bash
conda activate moe                      # python 3.11, torch >= 2.2.2, transformers 4.57
pip install -e ".[dev]"                 # or: pip install -r requirements-dev.txt
python scripts/env_check.py
python scripts/fetch_models.py --only small   # clip-vit-base-patch32 + xlm-roberta-base + vit5-base (smoke tests)
python scripts/fetch_models.py --only paper   # DFN ViT-H/14-378 + xlm-roberta-large + vit5-large (training box)
```

`fetch_models.py` converts `.bin`-only repos to safetensors under `models/` and registers them in
`configs/_base/model_paths.yaml` (transformers refuses `pytorch_model.bin` on torch < 2.6). Configs keep the
Hugging Face ids; `vivlmoe.utils.hf.resolve_model_path` redirects them.

Point `configs/_base/data_paths.yaml`, `configs/data/pretrain_corpus.yaml` and `configs/downstream/*.yaml` at your
dataset roots, then create the frozen dev holdouts once (`configs/data/splits/*.json` are committed for KTVIC and
UIT-ViIC):

```bash
python scripts/make_dev_split.py --ktvic-root /data/ktvic_dataset --uitviic-root /data/uitvic_dataset
python scripts/build_pretrain_corpus.py   # once: raw dump in dataset/ -> decontaminated corpus JSONL (docs/runbook.md 2.2)
python scripts/hf_corpus.py push --repo USER/NAME ...   # ship it to the GPU box via a private HF dataset repo, `pull` there (runbook 2.3)
python scripts/build_manifest.py --config configs/pretrain/stage2_contrastive.yaml --level decode   # once, validates corpus images
```

## Pipeline

Every command accepts `key.path=value` overrides and `--dry-run` (2 optimizer steps + a memory / throughput report).
Use `torchrun --nproc_per_node 4` on the GPU box; a plain `python` launch is single-process.

`bash scripts/run_pretrain.sh` runs Phases 1-2 end to end on all GPUs of the box (skips finished stages, resumes
interrupted ones; `PREFLIGHT=1` first runs every stage for a few steps into `runs_preflight/`, see `docs/runbook.md` 3.0).
`PIPELINE=v2` runs the modernised backbone instead (C-RADIOv4 + mmBERT distilled from SigLIP 2, sigmoid loss, same MoE,
Qwen3.5 head downstream; `requirements-v2.txt`, `docs/architecture_v2.md`, runbook 2.5 and 3.2).
The individual commands:

```bash
# Phase 1 - dense pretraining (paper Sec. 4)
torchrun --nproc_per_node 4 scripts/train.py --config configs/pretrain/stage1_teacher.yaml
torchrun --nproc_per_node 4 scripts/train.py --config configs/pretrain/stage2_contrastive.yaml
torchrun --nproc_per_node 4 scripts/train.py --config configs/pretrain/stage3_end2end.yaml

# Phase 2 - sparse upcycling + MoE pretraining (Sec. 5)
python scripts/upcycle.py --config configs/moe/upcycle.yaml --out runs/moe_upcycled/upcycled.pt
torchrun --nproc_per_node 4 scripts/train.py --config configs/moe/sparse_train.yaml
torchrun --nproc_per_node 4 scripts/train.py --config configs/ablations/ns0.yaml        # N_s / N_r / stabilizer ablations

# Phase 3 - downstream (Sec. 6), 5 seeds
python scripts/sweep_seeds.py --config configs/downstream/ktvic_caption.yaml --seeds 0 1 2 3 4 --nproc 4
python scripts/evaluate.py --config configs/downstream/ktvic_caption.yaml \
    --checkpoint runs/ktvic_caption_seed0/checkpoints/best.pt --splits dev,test
python scripts/analyze_routing.py --config configs/moe/sparse_train.yaml \
    --checkpoint runs/moe_sparse/checkpoints/best.pt --split ktvic_dev
```

Stage hand-offs use `init_from` (strict per-prefix partial loads); downstream and ablation configs inherit the model
section from the checkpoint they load (`model.from_checkpoint`), so the MoE topology and stabilizer switches can never
silently drift (see `docs/configs.md`, `docs/checkpoints.md`).

## Tests

```bash
make test        # unit + gloo 2-process tests, CPU, no downloads (~3 min)
make test-all    # + slow CLI smoke chains (stage1 -> stage2 -> stage3 -> upcycle -> MoE -> downstream -> evaluate)
python -m pytest -o addopts="" -m hf_cache   # real small models + real KTVIC images (after fetch_models --only small)
make lint
```

The smoke chains use `configs/_smoke/*_tiny.yaml` (random tiny towers, synthetic data). Paper invariants covered by
tests include: upcycled MoE == dense at noise 0 (Eq. 18), bias on logits with the sign rule and all-reduced counts
(Eq. 21-22), `L_mi` symmetric and identical under separate-pass and joint routing (Eq. 23-24), router losses reach the
optimizer, separate contrastive passes leak nothing, scheduler once per optimizer step, DDP gradients identical across
ranks, ordered evaluation gathers cover every sample exactly once.

## Documentation

**Tiếng Việt:** `docs/runbook.md` (toàn bộ lệnh chạy theo từng phase + chỗ cần chỉnh + lỗi thường gặp), `docs/architecture.md` (kiến trúc, shape tensor, luồng dữ liệu / huấn luyện, chuỗi stage) và `docs/architecture_v2.md` (kiến trúc v2: C-RADIOv4, mmBERT, sigmoid loss, head Qwen3.5).

`docs/paper_mapping.md` (equation / section -> code, and every documented deviation), `docs/configs.md` (generated
from the schema), `docs/training.md`, `docs/distributed.md`, `docs/checkpoints.md`, `docs/tasks.md` (Task protocol,
adding a dataset), `docs/moe.md`, `docs/downstream.md`.
