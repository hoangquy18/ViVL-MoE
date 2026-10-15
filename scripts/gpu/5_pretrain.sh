#!/usr/bin/env bash
# Bước 5: pretrain v2 thật trên 1 A40: distill (~4-5 giờ) -> align (~10-12 giờ) -> upcycle -> sparse (~13-15 giờ),
# vào runs/v2_*. Chạy trong tmux để mất kết nối SSH không giết tiến trình:
#
#   tmux new -s v2 'bash scripts/gpu/5_pretrain.sh'
#   bash scripts/gpu/5_pretrain.sh align upcycle sparse     # chỉ một số bước (vẫn theo thứ tự)
#   DISTILL_PHOMT=0 bash scripts/gpu/5_pretrain.sh          # distill chỉ trên caption (bỏ PhoMT)
# Bị ngắt thì chạy lại đúng lệnh: bước đã xong được bỏ qua, bước dở dang chạy tiếp từ last.pt.
# Theo dõi: tail -f runs/<run>/train.log, hoặc tensorboard --logdir runs
source "$(dirname "$0")/env.sh"
activate

if [ -z "${TMUX:-}" ] && [ -t 0 ] && [ "${DRY_RUN:-0}" != 1 ]; then
  echo "Không ở trong tmux: mất kết nối SSH sẽ dừng việc train. Nên chạy: tmux new -s v2 'bash $0 $*'"
  read -r -p "Vẫn chạy ở đây? [y/N] " ans
  [ "$ans" = y ] || exit 1
fi
mkdir -p "$RUNS"
run env RUN_DIR="$RUNS" PIPELINE=v2 bash scripts/run_pretrain.sh "$@" 2>&1 | tee -a "$LOG_DIR/5_pretrain.log"
log "xong bước 5: $RUNS/v2_moe_sparse/checkpoints/best.pt. Tiếp theo: bash scripts/gpu/6_downstream.sh"
