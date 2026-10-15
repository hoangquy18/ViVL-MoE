#!/usr/bin/env bash
# Bước 2: tải model v2 vào cache HF: C-RADIOv4-SO400M, mmBERT-base, SigLIP 2 g (~11 GB; PRETRAIN_ONLY=0 thêm
# SigLIP 2 So400m và Qwen3.5-0.8B, ~17 GB tổng), nạp sẵn code remote của C-RADIO, rồi ghim commit của C-RADIO
# vào configs/model/v2.yaml (model.vision.revision) nếu đang để null, để mọi bước train dùng cùng một phiên bản.
#
#   bash scripts/gpu/2_fetch_models.sh
source "$(dirname "$0")/env.sh"
activate

group=v2
[ "$PRETRAIN_ONLY" != 1 ] || group=v2-pretrain   # chỉ pretrain: bỏ SigLIP2-So400m, Qwen3.5
run python scripts/fetch_models.py --only "$group" --pin 2>&1 | tee -a "$LOG_DIR/2_fetch_models.log"
log "xong bước 2. Tiếp theo: bash scripts/gpu/3_checks.sh"
