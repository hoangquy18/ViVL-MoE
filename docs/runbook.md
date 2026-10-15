# Runbook — lệnh chạy và cách chỉnh

Tài liệu này liệt kê **toàn bộ lệnh** để chạy pipeline ViVL-MoE theo 3 phase và **chỗ cần chỉnh** trong
config cho từng bước. Các tài liệu kỹ thuật khác (tiếng Anh): `configs.md` (mọi key config, sinh tự động),
`training.md`, `distributed.md`, `checkpoints.md`, `tasks.md`, `moe.md`, `downstream.md`,
`paper_mapping.md`, `architecture.md` (kiến trúc + luồng chạy, tiếng Việt).

## 0. Quy ước chung

| việc | cách làm |
|---|---|
| môi trường | `conda activate moe` (Python 3.11, torch ≥ 2.2.2, transformers 4.57). Trên máy GPU nên dùng torch ≥ 2.6. |
| chạy 1 GPU / CPU | `python scripts/train.py --config <yaml> [key.path=value ...]` |
| chạy DDP | `torchrun --nproc_per_node 4 scripts/train.py --config <yaml> [overrides]` |
| override tạm thời | thêm `key.path=value` ở cuối lệnh, ví dụ `train.max_steps=100 data.batch_size=16 run.name=thu_nghiem`. Giá trị được parse như YAML (`1e-5`, `true`, `[a,b]`, `null`). |
| chỉnh cố định | sửa file YAML tương ứng (mục 2–5 bên dưới). Mọi key đều có mô tả trong `docs/configs.md`. |
| thử trước khi chạy thật | thêm `--dry-run`: dựng model + data, chạy 2 optimizer step, in bảng param-group, effective batch, peak memory vào `dry_run_report.json`. |
| tiếp tục run bị ngắt | `--resume auto` (đọc `runs/<name>/checkpoints/last.pt`) hoặc `--resume <path>`. Config phải giống hệt (fingerprint), khác thì dùng `init_from`. |
| kết quả | `runs/<run.name>/`: `config.resolved.yaml`, `command.txt`, `train.log`, `tb/` (TensorBoard), `checkpoints/{step_*.pt, last.pt, best.pt, *.meta.json}`, `metrics/*.json`, `predictions/*.jsonl`, `routing/`, `eval_history.jsonl`, `final_metrics.json`. |
| xem log | `tensorboard --logdir runs` |

Lưu ý YAML: **không** dùng `on/off/yes/no` làm giá trị chuỗi (PyYAML hiểu là boolean); các giá trị chuỗi
của repo dùng `none`, `disabled`, ...

## 1. Chuẩn bị môi trường và model

```bash
pip install -e ".[dev]"                 # hoặc pip install -r requirements-dev.txt
python scripts/env_check.py             # kiểm tra import, torch, cache HF, model đã convert

python scripts/fetch_models.py --only small   # clip-vit-base-patch32 + xlm-roberta-base + vit5-base (smoke test)
python scripts/fetch_models.py --only paper   # DFN5B-CLIP-ViT-H-14-378 + xlm-roberta-large + vit5-large (train thật)
```

`fetch_models.py` tải model và **convert `pytorch_model.bin` → `model.safetensors`** vào `models/<tên>/`
(transformers 4.57 từ chối `.bin` khi torch < 2.6), rồi ghi mapping vào `configs/_base/model_paths.yaml`.
Config vẫn ghi HF id (`apple/DFN5B-CLIP-ViT-H-14-378`, `VietAI/vit5-large`, ...); hàm
`vivlmoe.utils.hf.resolve_model_path` tự đổi sang thư mục local.

Chỉnh gì:

- Máy không có internet: chạy `fetch_models.py` ở nơi có mạng, copy thư mục `models/` + `~/.cache/huggingface`
  sang, đặt `export HF_HUB_OFFLINE=1`.
- Đổi model khác: sửa `configs/model/paper.yaml` (`model.vision.name`, `model.text.name`,
  `model.generation.name`) hoặc override `model.vision.name=...`. Model nhỏ để thử: `configs/model/small.yaml`.
- Thư mục `models/` ở chỗ khác: `--models-dir /path` khi fetch, hoặc sửa `configs/_base/model_paths.yaml`,
  hoặc `export VIVLMOE_MODEL_PATHS=/path/model_paths.yaml`.

## 2. Chuẩn bị dữ liệu

### 2.1 Đường dẫn cần sửa (theo máy)

| dữ liệu | file | key |
|---|---|---|
| KTVIC, UIT-ViIC (dev retrieval cho Stage 1–3 và sparse) | `configs/_base/data_paths.yaml` | `data.evals.ktvic_dev.root`, `data.evals.uitviic_dev.root` |
| PhoMT (Stage 1) | `configs/pretrain/stage1_teacher.yaml` | `data.train.root`, `data.evals.phomt_dev.root`, `data.train.options.variant` (`tokenization` \| `detokenization`), `max_train_samples` |
| corpus ảnh–caption (Stage 2/3, sparse) | `configs/data/pretrain_corpus.yaml` | `data.train.root` (mặc định `dataset`, tương đối so với thư mục chạy lệnh = gốc repo); file `corpus/pretrain_corpus.jsonl` do `scripts/build_pretrain_corpus.py` tạo (mục 2.2) |
| KTVIC / UIT-ViIC downstream | `configs/downstream/ktvic_caption.yaml`, `uitviic_caption.yaml` | `data.train.root`, `data.evals.dev.root`, `data.evals.test.root` |
| ViMACSA | `configs/downstream/vimacsa.yaml` | `data.*.root` (`train/dev/test.json`, `image/`), `options.image_aspect_json` |
| OpenViVQA, ViTextVQA, ViOCRVQA | `configs/downstream/{openvivqa,vitextvqa,viocrvqa}.yaml` | `data.*.root`, `splits.<split>.file`, `splits.<split>.image_root`, và `options` mô tả layout JSON (`records_path`, `image_key`, `question_key`, `answer_key`/`answers_key`, `image_lookup`, `image_filename_template`) — **kiểm tra lại với file thật** |

Kiểm tra một dataset mới trước khi train (đọc 8 record, không cần checkpoint chuẩn):

```bash
python scripts/evaluate.py --config configs/downstream/openvivqa.yaml \
    --checkpoint runs/moe_sparse/checkpoints/best.pt --splits dev --limit 8
```

### 2.2 Corpus pretrain từ `dataset/` (chạy một lần)

Dữ liệu thô nằm trong `dataset/` (đã git-ignore, khoảng 100 GB sau khi giải nén). Layout sau khi chuẩn bị:

| đường dẫn | nội dung |
|---|---|
| `images/` | mọi ảnh trong một thư mục phẳng, tên `<nguồn>_<id>.<ext>`: coco2017, viocrvqa, vitextvqa, openvivqa, oxford102, vivqa, vilc (= UIT-ViIC), evjvqa, localization, english |
| `filtered_data.json` | 1.153.967 caption tiếng Việt / 252.985 ảnh: `{"<idx>": {"image_id": "images/<tên>", "caption": ...}}` |
| `english.jsonl` | 185.651 caption tiếng Anh: `{"image": "images/english_<n>.<ext>", "caption": ...}` (chép từ checkout cũ `/Users/gos/Documents/Moe/moe/english.jsonl`; 2 file jsonl đi kèm ảnh English chỉ phủ 63.585 ảnh) |
| `corpus/` | `pretrain_corpus.jsonl` + `pretrain_corpus.report.json` do bước 4 sinh ra |
| `_raw/` | phần còn lại, không dùng khi train: RAR5 12 volume (50 GB; trong archive còn code, `.git` và `.env` của project cũ), `moe_dataset.json` (bản cũ của `filtered_data.json`), zip tải từ Drive. Xoá được sau khi corpus đã kiểm tra xong |

Dump gốc từ Google Drive để ảnh ở ba nơi: archive RAR (mọi nguồn trừ localization), `localization/images/` và
`english_dataset/`. Các bước chuẩn bị:

```bash
# 1) Chỉ giải nén thư mục ảnh. Google Drive đổi tên volume thành *.partN-00X.rar, khiến trình giải nén
#    dừng sau volume 1, nên phải đổi về *.partN.rar trước. Cần công cụ đọc được RAR5 đa volume (brew install unar).
cd dataset/moe_dataset
for f in MoE_dataset.part*-[0-9][0-9][0-9].rar; do mv -n "$f" "${f%-[0-9][0-9][0-9].rar}.rar"; done
unar -f -D -o . MoE_dataset.part1.rar 'MoE_dataset/moe_dataset/images/*'
cd ../..

# 2) Dồn ảnh vào dataset/images/ (mv trên cùng ổ đĩa chỉ là đổi tên, không chép dữ liệu; -n: không ghi đè)
mkdir -p dataset/images
for d in dataset/moe_dataset/MoE_dataset/moe_dataset/images dataset/localization/images dataset/english_dataset; do
  find "$d" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
    -exec sh -c 'mv -n "$@" dataset/images/' _ {} +
done
cp /Users/gos/Documents/Moe/moe/english.jsonl dataset/

# 3) Id ảnh test + dev chính thức của UIT-ViIC (dev khôi phục từ vilc_* trong corpus). Đã commit sẵn trong configs/data/splits/.
python scripts/make_dev_split.py --only uitviic_official --uitviic-corpus dataset/filtered_data.json

# 4) Dựng corpus đã khử nhiễm -> dataset/corpus/pretrain_corpus.jsonl + pretrain_corpus.report.json
python scripts/build_pretrain_corpus.py
```

Kết quả trên dump hiện tại: corpus còn **1.113.113 caption / 378.087 ảnh** (từ 1.339.618 / 438.636), không thiếu ảnh nào.
Số liệu chi tiết theo nguồn nằm trong `dataset/corpus/pretrain_corpus.report.json`.

Sự cố đã gặp khi giải nén: `unar` dừng gần cuối archive với lỗi "Archive parsing failed", để thiếu 4.097 ảnh `vivqa_*`
(id ≥ 137734). Bộ RAR vẫn đủ (volume 12 có block kết thúc với cờ volume cuối). Ảnh ViVQA chính là ảnh COCO, trùng
từng byte với `coco2017_<id>` (đã kiểm 19.675 cặp), nên các ảnh thiếu được dựng lại bằng APFS clone, ví dụ
`cp -c dataset/images/coco2017_000000137734.jpg dataset/images/vivqa_000000137734.jpg`. Bước 4 sẽ báo lỗi nếu còn
thiếu ảnh nào.

**Khử nhiễm.** Paper §4.2 loại khỏi corpus pretrain các ảnh thuộc split đánh giá của ViMMU-Bench, nhưng dump
này lại chứa chính dữ liệu benchmark. Mặc định của `build_pretrain_corpus.py`:

| nguồn trong dump | vấn đề | xử lý mặc định |
|---|---|---|
| `vilc_*` (UIT-ViIC: 3.850 ảnh / 19.255 caption) | gồm cả 231 ảnh test và 924 ảnh dev chính thức, kèm caption tham chiếu | bỏ ảnh test, dev chính thức và dev holdout (539) theo COCO id |
| `coco2017_*` (toàn bộ COCO train2017), `vivqa_*` | chứa lại đúng các ảnh UIT-ViIC trên (cùng COCO id) | bỏ theo cùng danh sách COCO id |
| `vitextvqa_*`, `openvivqa_*`, `viocrvqa_*` | cặp hỏi–đáp VQA đã chuyển thành câu; file bị đánh số lại nên không biết ảnh nào thuộc dev/test | bỏ cả nguồn |

Chỉnh gì:

- Giữ một nguồn VQA (khi đã có file split và tự loại dev/test của nó): `--exclude-sources openvivqa,viocrvqa`.
- Đổi danh sách COCO id bị loại: `--exclude-coco-ids FILE` (lặp lại được; khi truyền thì **thay** danh sách mặc định,
  nên phải liệt kê đủ cả ba file `uitviic_{test,official_dev,dev}_image_ids.json`).
- Tạo lại corpus cũ không khử nhiễm (chỉ để so sánh): `--no-decontamination`.
- Ảnh còn thiếu (giải nén dở): build dừng với lỗi, vẫn ghi report. Chấp nhận thiếu một ít: `--max-missing 0.001`.
- Dataset ở chỗ khác: `--root /path/dataset`. Đường dẫn ảnh trong corpus tương đối so với root (`images/<tên>`), nên
  chỉ cần chép `images/` + `corpus/` sang máy GPU rồi chạy với `data.train.root=/path/dataset`.
- Ảnh để theo thư mục khác (ví dụ mỗi nguồn một thư mục): `--image-dir DIR` (lặp lại được), ảnh được tìm theo tên file.
- So khớp chỉ dựa theo tên file/COCO id: ảnh trùng nội dung nhưng mang tên khác không bị phát hiện. KTVIC và
  ViMACSA không có trong dump.

### 2.3 Chuyển dữ liệu lên máy GPU qua HuggingFace

`scripts/hf_corpus.py` chỉ đóng gói những ảnh mà corpus dùng (378.087 ảnh, 104 GB) thành 53 file tar khoảng 2 GB
(không nén, vì JPEG/PNG đã nén sẵn), kèm `corpus/`. Script upload shard này trong lúc đóng shard kế tiếp, rồi xoá
shard tạm ngay sau khi upload: RAM gần như không đổi, đĩa chỉ cần trống thêm khoảng 4 GB. Chạy lại đúng lệnh thì
các shard đã lên repo được bỏ qua. File ẩn (`.env`, `.git`, `.DS_Store`) không bao giờ được đóng gói.

Có hai repo:

| repo | nội dung | chế độ |
|---|---|---|
| `nhq188/vivlmoe-pretrain` | ảnh + corpus (~104 GB) | public (`--public`; tài khoản free chỉ có 100 GB private). Là bản tạm, xoá sau khi train |
| `nhq188/vivlmoe-pretrain-extras` | PhoMT, KTVIC, UIT-ViIC (1,7 GB) | **luôn private**: PhoMT cấm phát tán, KTVIC/UIT-ViIC có tập test benchmark. Script kiểm tra lại sau khi tạo repo và từ chối repo public |

```bash
# Trên Mac: đăng nhập một lần bằng token quyền write (script không đọc hay ghi token)
hf auth login

# Upload (thêm --dry-run để chỉ xem kế hoạch); caffeinate giữ máy không ngủ trong lúc upload.
# HF_HUB_DISABLE_XET=1: dùng đường LFS cổ điển (xem ghi chú bên dưới)
HF_HUB_DISABLE_XET=1 caffeinate -i nice -n 10 python scripts/hf_corpus.py push --repo nhq188/vivlmoe-pretrain --public \
    --extras-repo nhq188/vivlmoe-pretrain-extras \
    --extra PhoMT=dataset/PhoMT \
    --extra ktvic_dataset=/Users/gos/Documents/Moe/moe/ktvic_dataset \
    --extra uitvic_dataset=/Users/gos/Documents/Moe/moe/uit-vilc/uitvic_dataset

# Trên máy GPU (cần đăng nhập vì repo extras là private) -> /data/vivlmoe/{dataset,PhoMT,ktvic_dataset,uitvic_dataset}
hf auth login
python scripts/hf_corpus.py pull --repo nhq188/vivlmoe-pretrain --root /data/vivlmoe/dataset \
    --extras-repo nhq188/vivlmoe-pretrain-extras --extras-dir /data/vivlmoe --workers 8

# Train xong: xoá repo public (không hoàn tác được, tự chạy khi chắc chắn; hoặc Settings -> Delete trên web)
python -c "from huggingface_hub import HfApi; HfApi().delete_repo('nhq188/vivlmoe-pretrain', repo_type='dataset')"
```

Ghi chú upload (2026-10): đường upload mặc định `hf_xet` nhiều lần chạy vọt lên rồi tụt còn vài chục KB/s, kể cả
khi mạng không mất gói. Với `HF_HUB_DISABLE_XET=1` (LFS cổ điển, đẩy thẳng lên S3), tốc độ giữ đều ~9 MB/s. Nếu upload
hay pull bị treo, cứ dừng rồi chạy lại đúng lệnh, vì cả hai đều tự resume.

Chỉnh gì: `--shard-gb` (kích thước shard), `--work-dir` (nơi đặt shard tạm, mặc định `dataset/.hf_upload`),
`--workers` (số shard tải song song khi pull). Pull bỏ qua shard đã có đủ ảnh trên đĩa, nên chạy lại được khi bị ngắt.
Bỏ `--public` thì repo ảnh là private, và extras có thể đi chung repo đó (không cần `--extras-repo`), nhưng tài khoản
free sẽ vượt hạn mức 100 GB.

### 2.4 Dev split và manifest ảnh (chạy một lần)

```bash
# 20 % KTVIC / UIT-ViIC train giữ làm dev (Appendix A). Đã có sẵn trong configs/data/splits/ — chỉ chạy lại khi đổi dữ liệu.
python scripts/make_dev_split.py --ktvic-root /data/ktvic_dataset --uitviic-root /data/uitvic_dataset --fraction 0.2 --seed 0

# Kiểm tra/decode toàn bộ ảnh corpus (sau mục 2.2) một lần, cache kết quả (ảnh hỏng/thiếu bị loại, không thay ảnh xám)
python scripts/build_manifest.py --config configs/pretrain/stage2_contrastive.yaml --level decode

# Đo tốc độ DataLoader để chọn data.num_workers (ViT-H đông lạnh rất nhanh, loader phải theo kịp)
python scripts/bench_loader.py --config configs/pretrain/stage2_contrastive.yaml --workers 8 --batches 50
```

Chỉnh gì: `data.num_workers`, `prefetch_factor`, `persistent_workers` trong `configs/_base/train_defaults.yaml`
(hoặc override); `image_check: exists|decode|none` và `missing_image_policy: skip|error` cho từng source
(corpus crawl → `skip`; benchmark → `error`).

## 2.5 v2: hai phép kiểm tra chạy trước trên máy GPU

Kiến trúc v2 (C-RADIOv4 + mmBERT + head Qwen3.5, xem `docs/architecture_v2.md`) dựa trên hai điều chỉ kiểm chứng được
trên máy GPU. Chạy hai lệnh này trước khi train v2; kết quả quyết định mặc định trong config v2.

```bash
# Môi trường v2 (transformers >= 5.2, timm); v1 vẫn chạy được trong môi trường này
pip install -r requirements-v2.txt && pip install -e .

# 1) Truy xuất zero-shot tiếng Việt trên KTVIC/UIT-ViIC dev: SigLIP2-g, SigLIP2-So400m, C-RADIOv4 (adaptor siglip2-g).
#    Cột rsum chọn vision encoder; số của siglip2-g cũng cho biết teacher của bước distill đọc tiếng Việt tốt đến đâu.
python scripts/probe_vision.py data.evals.ktvic_dev.root=/data/vivlmoe/ktvic_dataset \
    data.evals.uitviic_dev.root=/data/vivlmoe/uitvic_dataset

# 2) Head LLM: Qwen3.5-0.8B load đủ trọng số, perplexity tiếng Việt hợp lý, mọi tham số có gradient,
#    sinh theo batch (pad trái) trùng sinh từng mẫu. Mọi dòng phải là PASS.
python scripts/check_llm_head.py --model Qwen/Qwen3.5-0.8B --llm-class Qwen3_5ForCausalLM
```

Nếu `check_llm_head.py` báo FAIL ở bước load hoặc perplexity thì dùng tạm `--model Qwen/Qwen3-1.7B --llm-class ""`
(chạy được cả trên transformers 4.57) và báo lại. Dòng "fast path" chỉ là thông tin: cài `flash-linear-attention` và
`causal-conv1d` sẽ tăng tốc các lớp attention tuyến tính của Qwen3.5.

## 3. Phase 1 — dense pretraining (paper §4)

### 3.0 Chạy cả chuỗi pretrain bằng một lệnh (Phase 1 + Phase 2)

`scripts/run_pretrain.sh` chạy lần lượt stage1 → stage2 → stage3 → upcycle → sparse bằng `torchrun --standalone`
trên mọi GPU của máy. Bước đã xong (có `final_metrics.json` hoặc `upcycled.pt`) được bỏ qua, bước dở dang chạy tiếp
từ `last.pt`, nên khi bị ngắt chỉ cần chạy lại đúng lệnh cũ. Trước Stage 2, script dựng sẵn manifest ảnh (cache
decode) bằng một tiến trình, để các rank không phải chờ nhau lúc load data.

```bash
# Đường dẫn trên máy GPU (bỏ trống = giữ giá trị trong config)
export DATA_ROOT=/data/vivlmoe/dataset PHOMT_ROOT=/data/vivlmoe/PhoMT \
       KTVIC_ROOT=/data/vivlmoe/ktvic_dataset UITVIIC_ROOT=/data/vivlmoe/uitvic_dataset   # bố cục sau khi pull (mục 2.3)

# 1) Chạy thử cả chuỗi, mỗi bước 4 step, ghi vào runs_preflight/ (khoảng 15-30 phút): bắt lỗi đường dẫn, OOM, config
PREFLIGHT=1 bash scripts/run_pretrain.sh

# 2) Chạy thật trong tmux (mất vài ngày). Log của từng bước: runs/<run>/stdout.log và runs/<run>/train.log
mkdir -p runs && tmux new -s pretrain 'bash scripts/run_pretrain.sh 2>&1 | tee -a runs/pretrain_pipeline.log'

# Chỉ chạy vài bước (vẫn theo thứ tự stage1 stage2 stage3 upcycle sparse)
bash scripts/run_pretrain.sh stage3 upcycle sparse
```

Chỉnh gì: `NPROC` (mặc định đếm bằng `nvidia-smi`), `RUN_DIR` (mặc định `runs/`), `PREFLIGHT_STEPS`, và
`EXTRA="key=value ..."` để thêm override cho mọi bước train (ví dụ `EXTRA="logging.wandb.enabled=true"`). Muốn đổi
siêu tham số của riêng một stage thì sửa file config của stage đó (bảng batch/memory ở `docs/training.md`).

### 3.1 Từng bước bằng tay

```bash
# Stage 1: XLM-R <s> -> FC1 khớp CLIP text encoder trên PhoMT (Eq. 12)
torchrun --nproc_per_node 4 scripts/train.py --config configs/pretrain/stage1_teacher.yaml

# Stage 2: contrastive, ViT + W_v đông lạnh; text tower lấy từ Stage 1 (init_from text_encoder)
torchrun --nproc_per_node 4 scripts/train.py --config configs/pretrain/stage2_contrastive.yaml

# Stage 3: mở đông toàn bộ, 2 learning rate (1e-5 encoders / 5e-5 projections)
torchrun --nproc_per_node 4 scripts/train.py --config configs/pretrain/stage3_end2end.yaml
```

Luôn chạy `--dry-run` trước mỗi stage trên máy GPU để xem memory.

| muốn chỉnh | ở đâu | ghi chú |
|---|---|---|
| số GPU khác 4 | `data.batch_size`, `train.grad_accum_steps` | effective batch = `batch_size × grad_accum_steps × số GPU`. Paper: Stage 1 = 2048, Stage 2 = 2048, Stage 3 = 2048. Ví dụ 2 GPU Stage 2: `data.batch_size=64 train.grad_accum_steps=16`. Số **negative** của InfoNCE = `batch_size × số GPU` (accum không thêm negative) → ưu tiên batch lớn, accum nhỏ. |
| OOM Stage 3 | `model.vision.gradient_checkpointing`, `model.text.gradient_checkpointing` (đã bật), giảm `data.batch_size`, tăng `grad_accum_steps` | Stage 2/sparse: ViT đông lạnh chạy bf16 tự động (`model.vision.frozen_dtype: auto`). |
| số epoch / dừng sớm | `train.epochs`, `train.early_stopping.{enabled,patience}`, `eval.selection_metric` | Stage 1 chọn `phomt_dev/mse` (min); Stage 2/3 chọn `ktvic_dev/rsum` (max). Đổi sang `uitviic_dev/rsum` nếu muốn. |
| chạy theo số step | `train.max_steps=<N> train.epochs=null` | chỉ được đặt một trong hai. |
| learning rate | `optim.lr`; Stage 3 dùng `optim.groups` (module list + lr riêng) | `optim.no_decay_norm_and_bias`, `train.max_grad_norm` theo code gốc. |
| đông lạnh | `optim.freeze_preset` (`stage1`, `stage2`, `stage3`, `sparse`, `downstream_seq2seq`, `downstream_caption`, `downstream_cls`) + `optim.freeze: [module.path]` | tên module: `vision_encoder`, `vision_proj`, `resampler`, `text_encoder`, `text_proj`, `modality_embed`, `encoder`, `logit_scale`. |
| checkpoint đầu vào | `init_from: {path, load: [text_encoder]}` hoặc `load: [all]` | strict theo prefix; sai key là lỗi ngay. |
| tần suất eval / lưu | `eval.every_steps` \| `eval.every_epochs`, `checkpoint.every_steps` \| `every_epochs`, `checkpoint.keep_last_n`, `checkpoint.save_optimizer` (`last_only` mặc định) | |
| precision | `train.amp.dtype: bf16` (A40) \| `fp16` | fp16 + `model.vision.frozen_dtype: bf16` bị cấm. |
| Stage-1 teacher | `model.stage1.teacher_target: projected|eos_hidden`, `model.stage1.loss_preset: eq12|legacy` | |
| hàm contrastive | `loss.contrastive.symmetric_reduction: mean|sum`, `loss.contrastive.gather` | `mean` = 0.5×(CE + CE) như code gốc. |

Đầu ra dùng cho bước sau: `runs/stage1_teacher/checkpoints/best.pt` → Stage 2;
`runs/stage2_contrastive/checkpoints/best.pt` → Stage 3; `runs/stage3_end2end/checkpoints/best.pt` → upcycle.

## 4. Phase 2 — upcycling và MoE pretraining (paper §5)

```bash
# 4.1 Upcycle dense Stage-3 -> MoE (N_s=1, N_r=8, top-2), kiểm tra tương đương khi noise = 0 rồi mới lưu
python scripts/upcycle.py --config configs/moe/upcycle.yaml --out runs/moe_upcycled/upcycled.pt

# 4.2 Sparse training 20K step, ViT đông lạnh, z-loss + L_mi + aux-free bias, probe routing mỗi 500 step
torchrun --nproc_per_node 4 scripts/train.py --config configs/moe/sparse_train.yaml

# 4.3 Phân tích routing offline (heatmap expert x modality, sym-KL, margin, change rate)
python scripts/analyze_routing.py --config configs/moe/sparse_train.yaml \
    --checkpoint runs/moe_sparse/checkpoints/best.pt --split ktvic_dev \
    --prev-snapshot runs/moe_sparse/routing/snapshot_step_0019500.pt
```

| muốn chỉnh | ở đâu |
|---|---|
| topology (N_s, N_r, k) | `configs/moe/upcycle.yaml` → `model_overrides.moe.{num_shared,num_routed,top_k}` hoặc override khi upcycle: `model_overrides.moe.num_shared=2`. `top_k` phải ≤ `num_routed` (N_r = 1 → `top_k=1`). |
| noise / seed upcycle | flag của `upcycle.py`: `--noise-std-rel 0.01 --seed 0 --noise-on-bias --verify-atol 1e-5`; `--dense <ckpt>` để đổi checkpoint nguồn. |
| bias aux-free | `model_overrides.moe.aux_free_bias`, `bias_update_rate` (η, mặc định 1e-3), `bias_update_every: step|micro` |
| trọng số loss ổn định | `loss.router.{lambda_z,lambda_mi,lambda_lb,use_z,use_mi,use_lb,layer_reduction}` trong `sparse_train.yaml` |
| ngân sách / lr | `train.max_steps` (20000), `optim.lr` (1e-5), `data.batch_size` (128 × accum 4 × 4 GPU = 2048) |
| probe routing | `analysis.routing_probe: {split, every_steps, max_steps, max_batches}`; đặt `null` để tắt |
| baseline dense | bỏ qua phase 2, downstream trỏ vào `runs/stage3_end2end/checkpoints/best.pt` |

`sparse_train.yaml` dùng `model: {from_checkpoint: runs/moe_upcycled/upcycled.pt}`: **toàn bộ** section `model`
(topology, stabilizer) được kế thừa từ sidecar `upcycled.pt.meta.json`; muốn đổi thì ghi vào `model_overrides`,
không thêm key vào `model`. Loader từ chối nếu `loss.router.use_*` / `moe.aux_free_bias` khác checkpoint,
trừ khi `task.allow_stabilizer_override: true` (dùng cho ablation).

### 4.4 Ablation (paper §6.4)

```bash
# N_s / N_r sweep: mỗi topology cần upcycle riêng (lệnh nằm ở đầu từng file ablation)
python scripts/upcycle.py --config configs/moe/upcycle.yaml --out runs/moe_upcycled_ns0/upcycled.pt model_overrides.moe.num_shared=0
torchrun --nproc_per_node 4 scripts/train.py --config configs/ablations/ns0.yaml
python scripts/upcycle.py --config configs/moe/upcycle.yaml --out runs/moe_upcycled_nr1/upcycled.pt model_overrides.moe.num_routed=1 model_overrides.moe.top_k=1
torchrun --nproc_per_node 4 scripts/train.py --config configs/ablations/nr1_k1.yaml

# Tắt từng stabilizer (dùng chung checkpoint upcycle mặc định)
torchrun --nproc_per_node 4 scripts/train.py --config configs/ablations/no_zloss.yaml
torchrun --nproc_per_node 4 scripts/train.py --config configs/ablations/no_miloss.yaml
torchrun --nproc_per_node 4 scripts/train.py --config configs/ablations/no_auxfree_bias.yaml
torchrun --nproc_per_node 4 scripts/train.py --config configs/ablations/switch_lb.yaml

# Downstream từ checkpoint ablation (stabilizer đã tắt phải tắt luôn ở downstream)
torchrun --nproc_per_node 4 scripts/train.py --config configs/ablations/downstream_no_miloss.yaml

# 5 seed upcycle khác nhau cho cùng topology (figure sweep)
python scripts/sweep_seeds.py --config configs/moe/sparse_train.yaml --seeds 0 1 2 3 4 --nproc 4 --vary-upcycle-seed
```

Tạo ablation mới: copy một file trong `configs/ablations/`, đổi `run.name`, `model.from_checkpoint`,
`init_from.path`, `model_overrides.moe.*` hoặc `loss.router.*`. Overlay downstream cho dataset khác: copy
`downstream_no_miloss.yaml` và đổi `_include` (không override được `_include` từ dòng lệnh).

## 5. Phase 3 — downstream (paper §6)

```bash
# Captioning (KTVIC, UIT-ViIC): 5 seed, batch 8 x 4 GPU = 32, lr 3e-5, dừng sớm trên dev, test bằng best.pt
python scripts/sweep_seeds.py --config configs/downstream/ktvic_caption.yaml   --seeds 0 1 2 3 4 --nproc 4
python scripts/sweep_seeds.py --config configs/downstream/uitviic_caption.yaml --seeds 0 1 2 3 4 --nproc 4

# VQA (sau khi sửa root + layout trong file)
python scripts/sweep_seeds.py --config configs/downstream/openvivqa.yaml --seeds 0 1 2 3 4 --nproc 4
python scripts/sweep_seeds.py --config configs/downstream/vitextvqa.yaml --seeds 0 1 2 3 4 --nproc 4
python scripts/sweep_seeds.py --config configs/downstream/viocrvqa.yaml  --seeds 0 1 2 3 4 --nproc 4

# ViMACSA (classification head, macro-F1)
python scripts/sweep_seeds.py --config configs/downstream/vimacsa.yaml --seeds 0 1 2 3 4 --nproc 4

# Chạy đơn lẻ một seed
torchrun --nproc_per_node 4 scripts/train.py --config configs/downstream/ktvic_caption.yaml run.seed=0 run.name=ktvic_seed0

# Đánh giá lại một checkpoint (in bảng, ghi predictions JSONL + metrics JSON)
python scripts/evaluate.py --config configs/downstream/ktvic_caption.yaml \
    --checkpoint runs/ktvic_caption_seed0/checkpoints/best.pt --splits dev,test
```

Kết quả sweep: `runs/<name>_sweep/sweep_summary.json` và `sweep_summary.md` (mean ± std của 5 seed).

| muốn chỉnh | ở đâu |
|---|---|
| encoder đầu vào (MoE hay dense baseline) | `model.from_checkpoint` và `init_from.path` trong `configs/downstream/_downstream_defaults.yaml` (mặc định `runs/moe_sparse/checkpoints/best.pt`; dense baseline = `runs/stage3_end2end/checkpoints/best.pt`). Override 2 key cùng lúc: `model.from_checkpoint=... init_from.path=...`. |
| lr / batch / epoch | `_downstream_defaults.yaml`: `optim.lr` 3e-5, `data.batch_size` 8 (×4 GPU = 32), `train.epochs` 10 (ViMACSA 20), `train.early_stopping.patience` 3 |
| decoder sinh văn bản | `task.generation.{num_beams,max_new_tokens,length_penalty,early_stopping,min_new_tokens}`; `task.max_target_len`, `data.tokenizer.decoder_max_len` |
| metric | `task.metrics` (`bleu4`, `rougeL`, `cider`, `f1`, `em`, `macro_f1`), `task.bleu_tokenizer` (`13a` = giống HF evaluate), `task.metric_tokenizer.mode` (`vi_safe` giữ dấu tiếng Việt), `task.report_paper_compat_rouge` |
| chọn model | `eval.selection_metric: dev/<metric>`, `eval.final_splits: [test]` |
| VQA prompt / câu trả lời | `task.prompt_template` (`"{question}"`), `task.train_on_all_answers`, `task.max_source_len` |
| ViMACSA | `task.aspects`, `task.use_image_tags`, `task.max_images`, `task.legacy_parity`, `task.exclude_none_in_f1`, `model_overrides.classifier.pool` (`mean_joint` \| `mean_text` \| `concat_mean_v_t`) |
| decoder khác ViT5-large | `model_overrides.generation.name=VietAI/vit5-base` (hoặc BARTpho/mBART qua `AutoModelForSeq2SeqLM`, cần kiểm tra `decoder_start_token_id`) |
| OOM decoder | `model_overrides.generation.gradient_checkpointing=true`, giảm `data.batch_size` và tăng `train.grad_accum_steps` để giữ batch 32 |
| probe routing downstream | `analysis.routing_probe` trong `_downstream_defaults.yaml` (mỗi 500 step, 5000 step đầu, split dev) |

## 6. Smoke test trên máy không có GPU

```bash
make test        # unit + gloo 2-process (~3 phút, không tải gì)
make test-all    # + chuỗi CLI: stage1 -> stage2 -> stage3 -> upcycle -> MoE -> downstream -> evaluate (~5 phút)
python -m pytest -o addopts="" -m hf_cache     # model small thật + ảnh KTVIC thật (sau fetch_models --only small)
make lint

# Chạy tay từng bước với model tí hon + dữ liệu synthetic
python scripts/train.py --config configs/_smoke/stage1_tiny.yaml run.dir=/tmp/smoke
python scripts/train.py --config configs/_smoke/stage2_tiny.yaml run.dir=/tmp/smoke init_from.path=/tmp/smoke/stage1_tiny/checkpoints/best.pt
python scripts/train.py --config configs/_smoke/stage3_tiny.yaml run.dir=/tmp/smoke init_from.path=/tmp/smoke/stage2_tiny/checkpoints/best.pt
python scripts/upcycle.py --config configs/_smoke/upcycle_tiny.yaml --dense /tmp/smoke/stage3_tiny/checkpoints/best.pt --out /tmp/smoke/upcycle_tiny/upcycled.pt model.from_checkpoint=/tmp/smoke/stage3_tiny/checkpoints/best.pt
python scripts/train.py --config configs/_smoke/moe_tiny.yaml run.dir=/tmp/smoke model.from_checkpoint=/tmp/smoke/upcycle_tiny/upcycled.pt init_from.path=/tmp/smoke/upcycle_tiny/upcycled.pt
python scripts/train.py --config configs/_smoke/caption_tiny.yaml run.dir=/tmp/smoke

# Dry-run config thật với model nhỏ (cần fetch_models --only small; PhoMT + KTVIC local)
python scripts/train.py --config configs/pretrain/stage1_teacher.yaml --dry-run \
    model.vision.name=openai/clip-vit-base-patch32 model.text.name=FacebookAI/xlm-roberta-base \
    data.train.options.max_train_samples=256 data.num_workers=0 train.amp.enabled=false
```

## 7. Cheat-sheet override hay dùng

```bash
run.name=abc run.dir=runs run.seed=3                       # tên/thư mục/seed
data.batch_size=16 train.grad_accum_steps=8                # đổi batch, giữ effective batch
train.max_steps=200 train.epochs=null                      # chạy ngắn để thử
eval.every_steps=100 eval.every_epochs=null checkpoint.every_steps=100 checkpoint.every_epochs=null
train.amp.enabled=false                                    # tắt AMP (CPU tự tắt)
data.num_workers=0                                         # debug loader
logging.wandb.enabled=true logging.wandb.project=vivlmoe   # bật W&B (pip install wandb)
loss.router.use_mi=false task.allow_stabilizer_override=true    # tắt L_mi (ablation)
model_overrides.moe.num_routed=4 model_overrides.moe.top_k=2    # đổi topology khi upcycle
optim.freeze=[vision_encoder,text_encoder]                 # đông lạnh thêm module
dist.find_unused_parameters=true                           # chỉ khi debug DDP (bình thường không cần)
```

Thêm dataset mới: viết một adapter (`vivlmoe/data/adapters/<kind>.py`, `@register_dataset`) + một YAML,
theo mục "Adding a dataset" trong `docs/tasks.md`. VQA có layout JSON khác thường chỉ cần chỉnh `options`
của `vqa_generic`, không cần code.

## 8. Lỗi thường gặp

| thông báo | nguyên nhân / cách sửa |
|---|---|
| `ValueError: ... require users to upgrade torch to at least v2.6` | model còn ở dạng `.bin`; chạy `scripts/fetch_models.py` (convert sang safetensors) hoặc nâng torch ≥ 2.6. |
| `ConfigError: Unknown config key(s) [...] at 'train'` | gõ sai key; xem `docs/configs.md`. |
| `ConfigError: train: set exactly one of max_steps / epochs` | khi đổi sang step phải đặt `train.epochs=null`. |
| `ConfigError: model.moe.top_k must be in [1, num_routed=1]` | N_r = 1 thì phải `top_k=1`. |
| `ConfigError: stabilizer settings differ from checkpoint ...` | `loss.router.use_*`/`moe.aux_free_bias` khác checkpoint; đúng ý thì thêm `task.allow_stabilizer_override=true`. |
| `ConfigError: model.from_checkpoint is set; move the other model keys ... into 'model_overrides'` | file có `model.from_checkpoint` không được include `configs/model/*.yaml`; đổi model qua `model_overrides`. |
| `ConfigError: eval.selection_metric refers to split 'x' which is not in data.evals` | tên split trong `data.evals` và `selection_metric` phải khớp. |
| `CheckpointError: model fingerprint ... != current` khi `--resume` | config model đã đổi; resume chỉ dành cho cùng run. Dùng `init_from` cho run mới. |
| `CheckpointError: ... missing in checkpoint` khi `init_from` | prefix trong `init_from.load` không có trong checkpoint (ví dụ nạp head vào backbone); xem `docs/checkpoints.md`. |
| `RuntimeError: trainable parameters received no gradient: ...` | module trainable nhưng không dùng trong forward (ví dụ captioning mà không đông lạnh text tower) → dùng đúng `freeze_preset` (`downstream_caption`). |
| `DataIntegrityError` / `ImageLoadError` | ảnh thiếu hoặc hỏng; benchmark dùng `missing_image_policy: error` (sửa dữ liệu), corpus crawl dùng `skip` + `build_manifest.py`. |
| loss `nan` với bf16/fp32 | dừng ngay theo thiết kế; giảm `optim.lr`, kiểm tra dữ liệu. Với fp16 step bị skip và tự hạ scale. |
| eval quá chậm | giảm `data.evals.<split>.limit`, `analysis.routing_probe.max_batches`, tăng `data.eval_batch_size`. |
| `last.pt` là symlink hỏng | file step đã bị prune; resume từ `step_*.pt` còn lại hoặc `best.pt` chỉ để `init_from`. |
