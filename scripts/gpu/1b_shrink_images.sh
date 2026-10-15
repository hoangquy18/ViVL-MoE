#!/usr/bin/env bash
# Bước 1b (ổ đĩa < ~200 GB): thu nhỏ ảnh corpus tại chỗ để cạnh ngắn còn 512 px, chừa chỗ cho model và checkpoint.
# C-RADIO ép ảnh về 512 x 512, nên ảnh có cạnh ngắn > 512 không mất gì với model; ảnh nhỏ hơn giữ nguyên
# (ảnh COCO 640x480 không đổi). Ước tính 104 GB -> ~60 GB. Chạy lại an toàn: ảnh đã nhỏ được bỏ qua.
#
#   bash scripts/gpu/1b_shrink_images.sh --dry-run     # chỉ đếm số ảnh sẽ bị thu nhỏ
#   bash scripts/gpu/1b_shrink_images.sh               # SHORT_SIDE=512 QUALITY=90 SHRINK_WORKERS=<số CPU>
source "$(dirname "$0")/env.sh"
activate

[ "${DRY_RUN:-0}" = 1 ] || need "$DATA_ROOT/images" "chưa có ảnh corpus: chạy bước 1 trước"
log "trước: $(df -h "$DATA_DIR" 2>/dev/null | awk 'NR == 2 {print $4 " trống"}')"
run python scripts/shrink_images.py --root "$DATA_ROOT" --short-side "${SHORT_SIDE:-512}" \
  --quality "${QUALITY:-90}" --workers "${SHRINK_WORKERS:-$(nproc 2>/dev/null || echo 8)}" "$@" \
  2>&1 | tee -a "$LOG_DIR/1b_shrink_images.log"
log "sau: $(df -h "$DATA_DIR" 2>/dev/null | awk 'NR == 2 {print $4 " trống"}'). Tiếp theo: bash scripts/gpu/2_fetch_models.sh"
