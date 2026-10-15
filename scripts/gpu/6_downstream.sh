#!/usr/bin/env bash
# Bước 6: downstream v2 trên backbone runs/v2_moe_sparse (head Qwen3.5 cho caption/VQA, head phân loại cho
# ViMACSA). Mỗi dataset chạy scripts/sweep_seeds.py; kết quả: runs/v2_<dataset>_sweep/sweep_summary.md.
# Dataset chưa có dữ liệu trên máy được bỏ qua (KTVIC, UIT-ViIC có sẵn từ bước 1; VQA/ViMACSA đặt *_ROOT).
#
#   tmux new -s ds 'bash scripts/gpu/6_downstream.sh'                       # 1 seed (seed 0) mọi dataset
#   SEEDS="0 1 2 3 4" bash scripts/gpu/6_downstream.sh                       # đủ 5 seed cho số liệu cuối
#   DATASETS="ktvic_caption uitviic_caption" bash scripts/gpu/6_downstream.sh
#   OPENVIVQA_ROOT=/data/openvivqa bash scripts/gpu/6_downstream.sh          # thêm dataset VQA khi đã có file
# Layout JSON của VQA/ViMACSA nằm trong configs/downstream/<dataset>.yaml (kiểm tra lại với file thật).
# 1 A40: caption ~2 giờ/seed, VQA 3-10 giờ/seed. OOM: EXTRA_OVERRIDES="model_overrides.generation.gradient_checkpointing=true"
source "$(dirname "$0")/env.sh"
activate

SEEDS=${SEEDS:-0}
DATASETS=${DATASETS:-ktvic_caption uitviic_caption openvivqa vitextvqa viocrvqa vimacsa}
BACKBONE=$RUNS/v2_moe_sparse/checkpoints/best.pt
[ "${DRY_RUN:-0}" = 1 ] || need "$BACKBONE" "chưa xong bước 5 (sparse)"
read -r -a EXTRA_ARR <<< "${EXTRA_OVERRIDES:-}"

root_of() {
  case $1 in
    ktvic_caption) echo "$KTVIC_ROOT" ;;
    uitviic_caption) echo "$UITVIIC_ROOT" ;;
    openvivqa) echo "$OPENVIVQA_ROOT" ;;
    vitextvqa) echo "$VITEXTVQA_ROOT" ;;
    viocrvqa) echo "$VIOCRVQA_ROOT" ;;
    vimacsa) echo "$VIMACSA_ROOT" ;;
    *) die "dataset không biết: $1" ;;
  esac
}

for name in $DATASETS; do
  root=$(root_of "$name")
  if [ -z "$root" ] || [ ! -d "$root" ]; then
    log "bỏ qua $name: chưa có dữ liệu (đặt đường dẫn cho biến *_ROOT tương ứng trong scripts/gpu/env.sh)"
    continue
  fi
  log "downstream $name (seed: $SEEDS) <- $root"
  # shellcheck disable=SC2086
  run python scripts/sweep_seeds.py --config "configs/v2/downstream/$name.yaml" --seeds $SEEDS --nproc 1 \
    "run.dir=$RUNS" "model.from_checkpoint=$BACKBONE" "init_from.path=$BACKBONE" \
    "data.train.root=$root" "data.evals.dev.root=$root" "data.evals.test.root=$root" \
    ${EXTRA_ARR[@]+"${EXTRA_ARR[@]}"} 2>&1 | tee -a "$LOG_DIR/6_downstream_$name.log"
done
log "xong bước 6. Bảng kết quả: $RUNS/v2_*_sweep/sweep_summary.md"
