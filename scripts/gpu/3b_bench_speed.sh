#!/usr/bin/env bash
# Bước 3b: đo tốc độ trước khi pretrain thật (~5 phút, khi GPU đang rảnh). Mỗi step align/sparse cần 512 ảnh qua
# C-RADIO; nếu chỉ được ~5 ảnh/s (như preflight) thì align 6 epoch mất ~5 ngày. Script tách hai nghi phạm:
#   - GPU: C-RADIO forward trên ảnh ngẫu nhiên 512 px (không đọc đĩa), kèm các op tốn thời gian nhất;
#   - CPU: DataLoader của align trên ảnh thật (giải mã + resize bằng các worker), không chạy model.
# Bước nào chậm hơn quyết định tốc độ train. Gửi Claude toàn bộ output (logs/3b_bench_speed.log).
#
#   bash scripts/gpu/3b_bench_speed.sh
source "$(dirname "$0")/env.sh"
activate
export HF_HUB_OFFLINE=1

{
  log "máy"
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,clocks.sm,clocks.max.sm,power.draw,power.limit,utilization.gpu \
      --format=csv
  fi
  echo "nproc=$(nproc 2>/dev/null || echo '?') cgroup cpu.max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo '?')"
  echo "/dev/shm: $(df -h /dev/shm 2>/dev/null | tail -1)"
  echo "ảnh: $DATA_ROOT ($(df -h "$DATA_ROOT" 2>/dev/null | tail -1))"
  log "GPU: C-RADIO forward, ảnh ngẫu nhiên"
  run python scripts/bench_vision.py --config configs/v2/align.yaml
  log "CPU: DataLoader của align, ảnh thật (batch 256)"
  run python scripts/bench_loader.py --config configs/v2/align.yaml --batches 12 "data.train.root=$DATA_ROOT"
} 2>&1 | tee -a "$LOG_DIR/3b_bench_speed.log"
log "xong bước 3b: gửi Claude $LOG_DIR/3b_bench_speed.log"
