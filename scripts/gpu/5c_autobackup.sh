#!/usr/bin/env bash
# Bước 5c: worker sao lưu chạy song song với bước 5. Cứ BACKUP_INTERVAL giây (mặc định 300) đẩy lên repo model
# PRIVATE $CKPT_REPO mọi file mới hoặc vừa đổi trong runs/v2_*: last.pt (có optimizer, để train tiếp), best.pt,
# upcycled.pt, kèm log và metrics. Máy sập chỉ mất checkpoint đang đẩy dở; checkpoint trước đó đã nằm trên HF.
# Bước 5 tự chạy worker này ở nền (AUTO_BACKUP=1, mặc định) và đợi nó đẩy nốt khi train xong hoặc bị dừng. Chỉ
# chạy tay khi train đang chạy mà chưa có worker:
#
#   tmux new -s backup 'bash scripts/gpu/5c_autobackup.sh'     # Ctrl-C: đẩy nốt một lượt rồi thoát
#
# Mỗi checkpoint lên cùng sidecar trong một commit. Đè checkpoint đã có trên repo thì lịch sử repo được squash,
# nhưng HF vẫn giữ bản bị thay và tính vào quota (tài khoản free đầy sau vài chục bản checkpoint).
# BACKUP_PRUNE=1: worker xoá hẳn các bản bị thay (không khôi phục được), repo chỉ còn cỡ các file hiện tại.
# Cần token có quyền GHI (xem bước 5b).
# Log: logs/5c_autobackup.log. Khôi phục ở máy mới (sau bước 0-2):
#   python scripts/hf_checkpoints.py pull --repo $CKPT_REPO   # rồi chạy lại bước 5: train tiếp từ last.pt
source "$(dirname "$0")/env.sh"
activate
export HF_HUB_OFFLINE=0  # run_pretrain.sh tắt mạng HF cho tiến trình train; worker thì cần mạng

if [ "${BACKUP_CHECK:-1}" = 1 ]; then  # bước 5 kiểm tra trước rồi mới chạy worker ở nền (BACKUP_CHECK=0)
  run python scripts/hf_checkpoints.py check --repo "$CKPT_REPO" \
    || die "không đẩy được lên $CKPT_REPO: đăng nhập token có quyền GHI (.venv/bin/hf auth login)"
fi
args=(watch --repo "$CKPT_REPO" --runs "$RUNS" --interval "${BACKUP_INTERVAL:-300}")
# chạy ở nền thì bước 5 đã chuyển output vào chính file log này
[ -t 2 ] && args+=(--log-file "$LOG_DIR/5c_autobackup.log")
[ "${BACKUP_PRUNE:-0}" = 1 ] && args+=(--prune)
run exec nice -n 10 python scripts/hf_checkpoints.py "${args[@]}" "$@"
