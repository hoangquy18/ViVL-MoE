#!/usr/bin/env bash
# Bước 5: pretrain v2 thật: distill (~4-5 giờ) -> align (~10-12 giờ) -> upcycle -> sparse (~13-15 giờ) trên 1 A40
# hoặc 2x RTX 3090 (tương đương), vào runs/v2_*. Dùng mọi GPU của máy (DDP), batch mỗi GPU tự chỉnh theo phần
# cứng (env.sh: pretrain_overrides). Chạy trong tmux để mất kết nối SSH không giết tiến trình:
#
#   tmux new -s v2 'bash scripts/gpu/5_pretrain.sh'
#   bash scripts/gpu/5_pretrain.sh align upcycle sparse     # chỉ một số bước (vẫn theo thứ tự)
#   DISTILL_PHOMT=0 bash scripts/gpu/5_pretrain.sh          # distill chỉ trên caption (bỏ PhoMT)
# Bị ngắt thì chạy lại đúng lệnh: bước đã xong được bỏ qua, bước dở dang chạy tiếp từ last.pt.
# Theo dõi: tail -f runs/<run>/train.log, hoặc tensorboard --logdir runs
#
# Sao lưu tự động (AUTO_BACKUP=1, mặc định): trước khi train, kiểm tra token HF ghi được vào repo private
# $CKPT_REPO, rồi chạy worker bước 5c ở nền: checkpoint nào ghi xong cũng lên HF trong vòng BACKUP_INTERVAL giây
# (mặc định 300), nên máy sập không mất tiến độ. Train xong hoặc bị dừng (kể cả Ctrl-C) thì đợi worker đẩy nốt.
# AUTO_BACKUP=0: train không sao lưu (nhớ chạy bước 5b trước khi trả máy).
source "$(dirname "$0")/env.sh"
activate

if [ -z "${TMUX:-}" ] && [ -t 0 ] && [ "${DRY_RUN:-0}" != 1 ]; then
  echo "Không ở trong tmux: mất kết nối SSH sẽ dừng việc train. Nên chạy: tmux new -s v2 'bash $0 $*'"
  read -r -p "Vẫn chạy ở đây? [y/N] " ans
  [ "$ans" = y ] || exit 1
fi
mkdir -p "$RUNS"

backup_pid=
stop_backup() {
  [ -n "$backup_pid" ] || return 0
  local pid=$backup_pid
  backup_pid=
  if kill -0 "$pid" 2>/dev/null; then
    log "đợi worker sao lưu đẩy nốt lượt cuối (log: $LOG_DIR/5c_autobackup.log)"
    kill -TERM "$pid" 2>/dev/null || true
  fi
  wait "$pid" || log "CẢNH BÁO: worker sao lưu báo lỗi (xem $LOG_DIR/5c_autobackup.log); đẩy tay bằng" \
    "WITH_LAST=1 bash scripts/gpu/5b_backup_checkpoints.sh"
}
if [ "${AUTO_BACKUP:-1}" = 1 ]; then
  run env HF_HUB_OFFLINE=0 python scripts/hf_checkpoints.py check --repo "$CKPT_REPO" \
    || die "không đẩy được lên $CKPT_REPO: đăng nhập token có quyền GHI (.venv/bin/hf auth login), hoặc" \
      "AUTO_BACKUP=0 để train không sao lưu"
  if [ "${DRY_RUN:-0}" != 1 ]; then
    RUNS="$RUNS" LOG_DIR="$LOG_DIR" BACKUP_CHECK=0 bash scripts/gpu/5c_autobackup.sh \
      >>"$LOG_DIR/5c_autobackup.log" 2>&1 &
    backup_pid=$!
    trap stop_backup EXIT
    log "worker sao lưu chạy nền (pid $backup_pid) -> https://huggingface.co/$CKPT_REPO," \
      "log: $LOG_DIR/5c_autobackup.log"
  fi
fi

batch=$(pretrain_overrides)
log "batch theo phần cứng: ${batch:-theo config} (giữ 256 cặp mỗi step, 512 mẫu mỗi bước tối ưu)"
run env RUN_DIR="$RUNS" PIPELINE=v2 EXTRA="$batch ${EXTRA:-}" bash scripts/run_pretrain.sh "$@" \
  2>&1 | tee -a "$LOG_DIR/5_pretrain.log"
stop_backup
log "xong bước 5: $RUNS/v2_moe_sparse/checkpoints/best.pt. Tiếp theo: bash scripts/gpu/6_downstream.sh"
