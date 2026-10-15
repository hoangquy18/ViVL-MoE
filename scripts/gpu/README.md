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
| 2 | `2_fetch_models.sh` | tải C-RADIOv4, mmBERT, SigLIP 2, Qwen3.5 vào cache HF; ghim commit C-RADIO vào `configs/model/v2.yaml` | 10–30 phút |
| 3 | `3_checks.sh` | `probe_vision` (chọn vision encoder) + `check_llm_head` (Qwen3.5 dùng được không); **gửi log `logs/3_*.log` cho Claude** | 20–40 phút |
| 4 | `4_preflight.sh` | chạy thử cả chuỗi pretrain, mỗi bước 4 step, vào `runs_preflight/` (bắt lỗi đường dẫn, OOM) | 30–60 phút |
| 5 | `5_pretrain.sh` | pretrain thật: distill → align → upcycle → sparse vào `runs/v2_*` (chạy trong tmux) | ~1,5 ngày |
| 6 | `6_downstream.sh` | caption, VQA, ViMACSA trên backbone sparse; `SEEDS="0 1 2 3 4"` cho số liệu cuối | caption ~2 giờ/seed, VQA 3–10 giờ/seed |
| 7 | `7_delete_public_data.sh` | train xong hẳn: xoá repo HF public (gõ lại tên repo để xác nhận) | – |

```bash
bash scripts/gpu/0_setup.sh
hf auth login
bash scripts/gpu/1_pull_data.sh
bash scripts/gpu/2_fetch_models.sh
bash scripts/gpu/3_checks.sh
bash scripts/gpu/4_preflight.sh
tmux new -s v2 'bash scripts/gpu/5_pretrain.sh'
tmux new -s ds 'bash scripts/gpu/6_downstream.sh'
```

Ghi chú:
- Pretrain dùng mọi GPU của máy (DDP). Batch mỗi GPU tự chỉnh theo số GPU và bộ nhớ, giữ 256 cặp mỗi step cho
  sigmoid loss và 512 mẫu mỗi bước tối ưu: 1 A40 → 256 × accum 2; 2× RTX 3090 → 128 × 2 GPU × accum 2.
  Downstream (bước 6) với head Qwen3.5 cần card ≥ 40 GB; trên card 24 GB thì chưa vừa.
- Bước 5 và 6 chạy lâu: luôn chạy trong tmux (`tmux attach -t v2` để xem lại). Bị ngắt thì chạy lại đúng lệnh:
  bước đã xong được bỏ qua, bước dở dang chạy tiếp từ `last.pt`.
- Hết bộ nhớ (OOM) ở pretrain: giảm nửa batch mỗi GPU, gấp đôi accum (ghi đè giá trị tự chỉnh), ví dụ trên
  2× 3090: `EXTRA="data.batch_size=64 train.grad_accum_steps=4" bash scripts/gpu/4_preflight.sh --fresh`, rồi
  dùng đúng `EXTRA` đó cho `5_pretrain.sh`. Ở downstream:
  `EXTRA_OVERRIDES="model_overrides.generation.gradient_checkpointing=true" bash scripts/gpu/6_downstream.sh`.
- VQA và ViMACSA không có trên HF: chép dữ liệu lên máy, đặt `OPENVIVQA_ROOT`, `VITEXTVQA_ROOT`, `VIOCRVQA_ROOT`,
  `VIMACSA_ROOT` trong `env.sh`, và kiểm tra layout JSON trong `configs/downstream/<dataset>.yaml` với file thật.
- Chi tiết từng bước và cách chỉnh: `docs/runbook.md` (mục 2.3, 2.5, 3.2 và 5).
