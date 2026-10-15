#!/usr/bin/env bash
# Bước 1: kéo dữ liệu từ HuggingFace về $DATA_DIR (mặc định /data/vivlmoe, ~110 GB):
#   dataset/{images,corpus}  (repo ảnh, public)   +   PhoMT/, ktvic_dataset/, uitvic_dataset/  (repo extras, private)
#
#   hf auth login                          # một lần, token quyền read (bạn tự nhập, script không đụng tới token)
#   bash scripts/gpu/1_pull_data.sh
# Bị ngắt thì chạy lại đúng lệnh: shard đã có đủ ảnh được bỏ qua. Dòng "read operation timed out ... Trying to
# resume" chỉ là đọc chậm, tải tiếp từ chỗ dở. Mạng mạnh: PULL_WORKERS=8; mạng chập chờn: PULL_WORKERS=2.
source "$(dirname "$0")/env.sh"
activate

hf_logged_in() { hf auth whoami >/dev/null 2>&1 || huggingface-cli whoami >/dev/null 2>&1; }
if [ "${DRY_RUN:-0}" != 1 ] && ! hf_logged_in; then
  die "chưa đăng nhập HuggingFace: chạy 'hf auth login' (tài khoản xem được $HF_EXTRAS_REPO) rồi chạy lại"
fi
mkdir -p "$DATA_DIR" 2>/dev/null || die "không tạo được $DATA_DIR: đặt DATA_DIR=<thư mục bạn có quyền ghi>"
free_gb=$(df -Pk "$DATA_DIR" | awk 'NR == 2 {print int($4 / 1048576)}')
log "$DATA_DIR còn trống ${free_gb} GB (cần ~110 GB)"
[ "$free_gb" -ge 115 ] || [ -d "$DATA_ROOT/images" ] || die "không đủ chỗ trống ở $DATA_DIR"

run python scripts/hf_corpus.py pull --repo "$HF_REPO" --root "$DATA_ROOT" \
  --extras-repo "$HF_EXTRAS_REPO" --extras-dir "$DATA_DIR" --workers "$PULL_WORKERS" \
  2>&1 | tee -a "$LOG_DIR/1_pull_data.log"

if [ "${DRY_RUN:-0}" != 1 ]; then
  need "$DATA_ROOT/corpus/pretrain_corpus.jsonl" "corpus pretrain"
  need "$DATA_ROOT/images" "ảnh corpus"
  need "$PHOMT_ROOT/detokenization/train" "PhoMT"
  need "$KTVIC_ROOT/train_data.json" "KTVIC"
  need "$UITVIIC_ROOT" "UIT-ViIC"
  log "corpus: $(wc -l < "$DATA_ROOT/corpus/pretrain_corpus.jsonl") caption (đúng là 1113113)"
fi
log "xong bước 1. Tiếp theo: bash scripts/gpu/2_fetch_models.sh"
