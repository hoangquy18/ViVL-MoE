#!/usr/bin/env bash
# Bước 4: chạy thử cả chuỗi v2 (distill -> align -> upcycle -> sparse), mỗi bước 4 step, vào runs_preflight/:
# bắt lỗi đường dẫn, hết bộ nhớ (OOM), config trước khi chạy thật. Khoảng 30-60 phút.
#
#   bash scripts/gpu/4_preflight.sh            # bước đã xong ở lần trước được bỏ qua
#   bash scripts/gpu/4_preflight.sh --fresh    # xoá runs_preflight/ (chỉ là kết quả chạy thử) rồi chạy lại
# Batch mỗi GPU tự chỉnh theo số GPU và bộ nhớ (pretrain_overrides trong env.sh). Vẫn OOM: ghi đè bằng EXTRA
# (đặt sau nên thắng), ví dụ EXTRA="data.batch_size=64 train.grad_accum_steps=4", rồi chạy lại với --fresh.
source "$(dirname "$0")/env.sh"
activate

if [ "${1:-}" = "--fresh" ]; then
  run rm -rf runs_preflight
fi
batch=$(pretrain_overrides)
log "batch theo phần cứng: ${batch:-theo config}"
run env RUN_DIR=runs_preflight PREFLIGHT=1 PIPELINE=v2 EXTRA="$batch ${EXTRA:-}" bash scripts/run_pretrain.sh \
  2>&1 | tee -a "$LOG_DIR/4_preflight.log"
log "xong bước 4 (log: $LOG_DIR/4_preflight.log). Tiếp theo, trong tmux: bash scripts/gpu/5_pretrain.sh"
