#!/usr/bin/env bash
# Bước 5b: sao lưu kết quả pretrain lên repo model PRIVATE trên HuggingFace trước khi trả máy thuê (máy bị xoá
# là mất hết). Đẩy best.pt của từng bước, upcycled.pt, config, log, metrics, TensorBoard (~10 GB); repo public
# bị từ chối. Chạy lại an toàn: file đã có trên repo (cùng kích thước) được bỏ qua.
#
#   bash scripts/gpu/5b_backup_checkpoints.sh --dry-run      # xem danh sách file, không gọi mạng
#   bash scripts/gpu/5b_backup_checkpoints.sh
#   WITH_LAST=1 bash scripts/gpu/5b_backup_checkpoints.sh    # thêm last.pt (có optimizer) để train tiếp ở máy khác
#
# Cần token HF có quyền GHI vào repo $CKPT_REPO (token read-only của bước 1 không đủ): tạo repo model private trên
# huggingface.co, tạo token fine-grained có quyền write cho repo đó, rồi .venv/bin/hf auth login.
# Khôi phục ở máy khác: python scripts/hf_checkpoints.py pull --repo $CKPT_REPO
source "$(dirname "$0")/env.sh"
activate

args=(push --repo "$CKPT_REPO" --runs "$RUNS")
[ "${WITH_LAST:-0}" = 1 ] && args+=(--with-last)
run python scripts/hf_checkpoints.py "${args[@]}" "$@" 2>&1 | tee -a "$LOG_DIR/5b_backup_checkpoints.log"
log "xong bước 5b: https://huggingface.co/$CKPT_REPO (private)"
