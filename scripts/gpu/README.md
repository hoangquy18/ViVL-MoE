# Chạy ViVL-MoE v2 trên máy GPU (1 A40), từ đầu đến cuối

Mỗi bước là một script, chạy từ thư mục repo theo đúng thứ tự. Đường dẫn và tên repo HF nằm ở `env.sh`: sửa một lần,
hoặc ghi đè khi chạy (`DATA_DIR=/mnt/data bash scripts/gpu/1_pull_data.sh`). Log của từng bước ghi vào `logs/`.
`DRY_RUN=1` chỉ in các lệnh nặng mà không chạy.

```bash
git clone https://github.com/hoangquy18/ViVL-MoE.git && cd ViVL-MoE
```

| bước | script | làm gì | thời gian (1 A40) |
|---|---|---|---|
| 0 | `0_setup.sh` | tạo `.venv` Python 3.11, cài `requirements-v2.txt` + repo, kiểm tra GPU và môi trường | 10–20 phút |
| 1 | `1_pull_data.sh` | kéo ảnh + corpus (repo public) và PhoMT, KTVIC, UIT-ViIC (repo private) về `/data/vivlmoe` (~110 GB); cần `hf auth login` trước | tuỳ mạng |
| 1b | `1b_shrink_images.sh` | ổ < ~200 GB: thu nhỏ ảnh corpus để cạnh ngắn còn 512 px (C-RADIO ép về 512, model không mất gì); 104 GB → ~60 GB | 10–30 phút |
| 2 | `2_fetch_models.sh` | tải C-RADIOv4, mmBERT, SigLIP2-g (thêm SigLIP2-So400m, Qwen3.5 khi `PRETRAIN_ONLY=0`) vào cache HF; ghim commit C-RADIO vào `configs/model/v2.yaml` | 10–30 phút |
| 3 | `3_checks.sh` | `probe_vision` (chọn vision encoder) + `check_llm_head` (Qwen3.5 dùng được không); **gửi log `logs/3_*.log` cho Claude** | 20–40 phút |
| 3b | `3b_bench_speed.sh` | đo tốc độ: C-RADIO forward (GPU, ảnh ngẫu nhiên) và DataLoader (CPU, ảnh thật) tách riêng; **gửi log `logs/3b_bench_speed.log` cho Claude** | ~5 phút |
| 4 | `4_preflight.sh` | chạy thử cả chuỗi pretrain, mỗi bước 4 step, vào `runs_preflight/` (bắt lỗi đường dẫn, OOM) | 30–60 phút |
| 5 | `5_pretrain.sh` | pretrain thật: distill → align → upcycle → sparse vào `runs/v2_*` (chạy trong tmux), kèm worker 5c ở nền | ~1,5 ngày |
| 5b | `5b_backup_checkpoints.sh` | sao lưu `best.pt` từng bước + log lên repo model **private** `CKPT_REPO` (cần token có quyền write); trước khi trả máy, khi 5c đã dừng | tuỳ mạng (~10 GB) |
| 5c | `5c_autobackup.sh` | worker sao lưu chạy song song bước 5 (bước 5 tự chạy): mỗi checkpoint mới (`last.pt` có optimizer, `best.pt`, `upcycled.pt`) + log lên `CKPT_REPO` trong vòng 5 phút | suốt lúc train |
| 6 | `6_downstream.sh` | caption, VQA, ViMACSA trên backbone sparse; `SEEDS="0 1 2 3 4"` cho số liệu cuối | caption ~2 giờ/seed, VQA 3–10 giờ/seed |
| 7 | `7_delete_public_data.sh` | train xong hẳn: xoá repo HF public (gõ lại tên repo để xác nhận) | – |

```bash
bash scripts/gpu/0_setup.sh
hf auth login
bash scripts/gpu/1_pull_data.sh
bash scripts/gpu/1b_shrink_images.sh   # chỉ khi ổ < ~200 GB
bash scripts/gpu/2_fetch_models.sh
bash scripts/gpu/3_checks.sh
bash scripts/gpu/3b_bench_speed.sh
bash scripts/gpu/4_preflight.sh
hf auth login                               # token có quyền GHI vào CKPT_REPO (bước 5 kiểm tra trước khi train)
tmux new -s v2 'bash scripts/gpu/5_pretrain.sh'
bash scripts/gpu/5b_backup_checkpoints.sh   # trước khi trả máy (worker 5c đã đẩy gần hết)
tmux new -s ds 'bash scripts/gpu/6_downstream.sh'
```

Ghi chú:
- Ổ đĩa (dữ liệu, model và `runs/` thường nằm chung một ổ): dữ liệu ~106 GB (~62 GB sau bước 1b), model ~11 GB
  (~17 GB khi `PRETRAIN_ONLY=0`), checkpoint ~30 GB lúc cao nhất, môi trường Python ~10 GB. Kiểm tra: `df -h /data
  /workspace ~/.cache`.
- `PRETRAIN_ONLY=1` (mặc định trong `env.sh`): chỉ tải model pretrain cần, bước 3 so C-RADIO với SigLIP2-g và bỏ
  `check_llm_head`. Đặt `PRETRAIN_ONLY=0` trước khi làm downstream.
- Cài thêm gói thì dùng `.venv/bin/pip`, đừng dùng `pip` của conda đang bật: các script luôn chạy trong `.venv`.
- GPU chưa có bf16 thật (compute capability < 8.0: Quadro RTX 8000, Titan RTX, V100, T4): bf16 ở đó là giả lập,
  chậm ~10 lần. Pretrain tự chuyển sang fp16 (`pretrain_overrides` trong `env.sh`: AMP fp16 + GradScaler, teacher
  của distill cũng fp16); C-RADIO và SigLIP2-g đã kiểm ở fp16: không tràn số, lệch fp32 < 0,3%. Nên thuê GPU từ
  Ampere trở lên (A40, A6000, L40S, 3090/4090) nếu có.
- Pretrain dùng mọi GPU của máy (DDP). Batch mỗi GPU tự chỉnh theo số GPU và bộ nhớ, giữ 256 cặp mỗi step cho
  sigmoid loss và 512 mẫu mỗi bước tối ưu: 1 A40 → 256 × accum 2; 2× RTX 3090 → 128 × 2 GPU × accum 2.
  Downstream (bước 6) với head Qwen3.5 cần card ≥ 40 GB; trên card 24 GB thì chưa vừa.
- Bước 5 và 6 chạy lâu: luôn chạy trong tmux (`tmux attach -t v2` để xem lại). Bị ngắt thì chạy lại đúng lệnh:
  bước đã xong được bỏ qua, bước dở dang chạy tiếp từ `last.pt`.
- Sao lưu khi train (`AUTO_BACKUP=1`, mặc định): bước 5 kiểm tra token ghi được vào `CKPT_REPO` rồi chạy worker
  5c ở nền (`logs/5c_autobackup.log`). Mỗi checkpoint lên HF cùng sidecar trong một commit; đè bản cũ thì lịch sử
  repo được squash, nhưng HF vẫn giữ bản bị thay và tính vào quota: tài khoản free (100 GB private) đầy sau vài
  chục bản checkpoint, rồi lượt sao lưu nào cũng báo `Private storage limit reached`. `BACKUP_PRUNE=1` cho worker
  xoá hẳn các bản bị thay (không khôi phục được); xoá tay: `python scripts/hf_checkpoints.py prune --repo
  $CKPT_REPO` liệt kê, thêm `--yes` để xoá. Máy sập: thuê máy mới, chạy bước 0-2,
  `.venv/bin/python scripts/hf_checkpoints.py pull --repo $CKPT_REPO`, rồi chạy lại bước 5 (train tiếp từ
  `last.pt`). `BACKUP_INTERVAL` đổi chu kỳ (giây); `AUTO_BACKUP=0` tắt.
- Hết bộ nhớ (OOM) ở pretrain: giảm nửa batch mỗi GPU, gấp đôi accum (ghi đè giá trị tự chỉnh), ví dụ trên
  2× 3090: `EXTRA="data.batch_size=64 train.grad_accum_steps=4" bash scripts/gpu/4_preflight.sh --fresh`, rồi
  dùng đúng `EXTRA` đó cho `5_pretrain.sh`. Ở downstream:
  `EXTRA_OVERRIDES="model_overrides.generation.gradient_checkpointing=true" bash scripts/gpu/6_downstream.sh`.
- VQA và ViMACSA không có trên HF: chép dữ liệu lên máy, đặt `OPENVIVQA_ROOT`, `VITEXTVQA_ROOT`, `VIOCRVQA_ROOT`,
  `VIMACSA_ROOT` trong `env.sh`, và kiểm tra layout JSON trong `configs/downstream/<dataset>.yaml` với file thật.
- Chi tiết từng bước và cách chỉnh: `docs/runbook.md` (mục 2.3, 2.5, 3.2 và 5).
