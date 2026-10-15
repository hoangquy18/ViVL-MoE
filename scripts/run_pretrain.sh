#!/usr/bin/env bash
# Toàn bộ pretraining trên một máy nhiều GPU, theo đúng thứ tự:
#   PIPELINE=v1 (paper §4-5, mặc định):
#     stage1 (teacher, PhoMT) -> stage2 (contrastive) -> stage3 (end-to-end) -> upcycle -> sparse (MoE)
#   PIPELINE=v2 (docs/architecture_v2.md; môi trường requirements-v2.txt):
#     distill (mmBERT <- text SigLIP 2) -> align (sigmoid, dense) -> upcycle -> sparse (MoE)
#
#   bash scripts/run_pretrain.sh                  # cả chuỗi; bước đã xong (có final_metrics.json / upcycled.pt)
#                                                 # được bỏ qua, bước dở dang chạy tiếp từ last.pt (--resume auto)
#   bash scripts/run_pretrain.sh stage2 stage3    # chỉ các bước liệt kê (vẫn theo thứ tự trên)
#   PREFLIGHT=1 bash scripts/run_pretrain.sh      # chạy thử cả chuỗi, mỗi bước vài step, vào runs_preflight/
#   PIPELINE=v2 bash scripts/run_pretrain.sh      # chuỗi v2 (run riêng: runs/v2_*, không đụng run v1)
#
# Biến môi trường (để trống = giữ giá trị trong config):
#   PIPELINE      v1 | v2 (mặc định v1)
#   DATA_ROOT     thư mục dataset (images/ + corpus/pretrain_corpus.jsonl)   [configs/data/pretrain_corpus.yaml]
#   PHOMT_ROOT    PhoMT cho Stage 1 / distill v2                             [configs/pretrain/stage1_teacher.yaml]
#   DISTILL_PHOMT 0 = distill v2 chỉ trên caption (bỏ PhoMT), mặc định 1
#   KTVIC_ROOT    KTVIC (dev retrieval để chọn checkpoint)                   [configs/_base/data_paths.yaml]
#   UITVIIC_ROOT  UIT-ViIC (dev retrieval)                                   [configs/_base/data_paths.yaml]
#   NPROC         số GPU (mặc định: đếm bằng nvidia-smi)
#   RUN_DIR       nơi ghi các run (mặc định runs/, PREFLIGHT=1 -> runs_preflight/)
#   EXTRA         override thêm cho mọi bước train, ví dụ EXTRA="logging.wandb.enabled=true"
#   PYTHON        trình thông dịch (mặc định python)
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
PREFLIGHT=${PREFLIGHT:-0}
PIPELINE=${PIPELINE:-v1}
if [ "$PREFLIGHT" = 1 ]; then RUN_DIR=${RUN_DIR:-runs_preflight}; else RUN_DIR=${RUN_DIR:-runs}; fi
NPROC=${NPROC:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)}
[ "${NPROC:-0}" -ge 1 ] 2>/dev/null || NPROC=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

case "$PIPELINE" in
  v1) ALL_STAGES=(stage1 stage2 stage3 upcycle sparse) ;;
  v2) ALL_STAGES=(distill align upcycle sparse) ;;
  *) echo "PIPELINE phải là v1 hoặc v2 (đang là $PIPELINE)" >&2; exit 2 ;;
esac
if [ $# -gt 0 ]; then WANT=" $* "; else WANT=" ${ALL_STAGES[*]} "; fi
for s in $WANT; do
  case " ${ALL_STAGES[*]} " in *" $s "*) ;; *) echo "bước không hợp lệ: $s (chọn trong: ${ALL_STAGES[*]})" >&2; exit 2 ;; esac
done
wanted() { case "$WANT" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

S1="$RUN_DIR/stage1_teacher"; S2="$RUN_DIR/stage2_contrastive"; S3="$RUN_DIR/stage3_end2end"
UP="$RUN_DIR/moe_upcycled/upcycled.pt"; SP="$RUN_DIR/moe_sparse"
D2="$RUN_DIR/v2_distill_text"; A2="$RUN_DIR/v2_align"
UP2="$RUN_DIR/v2_moe_upcycled/upcycled.pt"; SP2="$RUN_DIR/v2_moe_sparse"

# ------------------------------------------------------------------ override chung cho mọi bước
read -r -a EXTRA_ARR <<< "${EXTRA:-}"
COMMON=("run.dir=$RUN_DIR")
[ -n "${KTVIC_ROOT:-}" ] && COMMON+=("data.evals.ktvic_dev.root=$KTVIC_ROOT")
[ -n "${UITVIIC_ROOT:-}" ] && COMMON+=("data.evals.uitviic_dev.root=$UITVIIC_ROOT")
CORPUS=("${COMMON[@]}")
[ -n "${DATA_ROOT:-}" ] && CORPUS+=("data.train.root=$DATA_ROOT")
PF=()
if [ "$PREFLIGHT" = 1 ]; then
  n=${PREFLIGHT_STEPS:-4}
  PF=("train.max_steps=$n" "train.epochs=null" "train.early_stopping.enabled=false"
      "eval.every_steps=$n" "eval.every_epochs=null" "checkpoint.every_steps=$n" "checkpoint.every_epochs=null"
      "data.train.limit=4096" "data.train.image_check=none" "logging.wandb.enabled=false")
fi

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
need() { [ -e "$1" ] || { echo "thiếu $1 ($2)" >&2; exit 1; }; }

# train_stage <run dir> <config> [overrides...]: bỏ qua nếu đã xong, resume nếu có last.pt
train_stage() {
  local dir=$1 cfg=$2; shift 2
  if [ -f "$dir/final_metrics.json" ]; then log "bỏ qua $dir (đã có final_metrics.json)"; return; fi
  local resume=()
  [ -f "$dir/checkpoints/last.pt" ] && resume=(--resume auto)
  mkdir -p "$dir"
  log "train $cfg -> $dir (${NPROC} GPU${resume[@]+, resume})"
  torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py --config "$cfg" \
    ${resume[@]+"${resume[@]}"} "$@" ${PF[@]+"${PF[@]}"} ${EXTRA_ARR[@]+"${EXTRA_ARR[@]}"} \
    2>&1 | tee -a "$dir/stdout.log"
}

# upcycle_step <dense best.pt> <upcycle config> <out>: MoE checkpoint, bỏ qua nếu đã có
upcycle_step() {
  local dense=$1 cfg=$2 out=$3
  if [ -f "$out" ]; then log "bỏ qua upcycle ($out đã có)"; return; fi
  need "$dense" "bước dense trước upcycle chưa xong"
  mkdir -p "$(dirname "$out")"
  log "upcycle $dense -> $out (kiểm tra tương đương MoE == dense khi noise = 0)"
  "$PYTHON" scripts/upcycle.py --config "$cfg" --dense "$dense" --out "$out" "run.dir=$RUN_DIR" \
    2>&1 | tee -a "$(dirname "$out")/stdout.log"
}

# ------------------------------------------------------------------ kiểm tra trước khi chạy
log "PIPELINE=$PIPELINE RUN_DIR=$RUN_DIR NPROC=$NPROC PREFLIGHT=$PREFLIGHT bước:$WANT"
if [ "$PIPELINE" = v2 ]; then
  "$PYTHON" scripts/env_check.py --v2
  # Tải model v2 và nạp code remote của C-RADIO một lần, trước torchrun (các rank không ghi cache cùng lúc).
  log "model v2 (cache HF) + code remote C-RADIO"
  "$PYTHON" scripts/fetch_models.py --only v2-pretrain
  # Mọi model đã nằm trong cache: các rank load offline (không gọi Hub; không chạy luồng auto-conversion
  # safetensors của transformers cho repo chỉ có pytorch_model.bin như mmBERT)
  export HF_HUB_OFFLINE=1
  MANIFEST_CFG=configs/v2/align.yaml
else
  "$PYTHON" scripts/env_check.py
  MANIFEST_CFG=configs/pretrain/stage2_contrastive.yaml
fi
if wanted stage2 || wanted stage3 || wanted sparse || wanted distill || wanted align; then
  need "${DATA_ROOT:-dataset}/corpus/pretrain_corpus.jsonl" "chạy scripts/build_pretrain_corpus.py, xem docs/runbook.md mục 2.2"
  if [ "$PREFLIGHT" != 1 ] && { [ "$PIPELINE" = v1 ] || wanted align || wanted sparse; }; then
    # Kiểm tra/decode ảnh một lần trước khi torchrun: cache trúng thì vài chục giây; nếu để từng rank tự
    # dựng cache lúc load data, các rank chờ nhau cả giờ và có thể vượt timeout NCCL.
    log "manifest ảnh corpus (decode, có cache)"
    "$PYTHON" scripts/build_manifest.py --config "$MANIFEST_CFG" --level decode "${CORPUS[@]}"
  fi
fi

# ------------------------------------------------------------------ Phase 1: dense pretraining
if wanted stage1; then
  s1=("${COMMON[@]}")
  [ -n "${PHOMT_ROOT:-}" ] && s1+=("data.train.root=$PHOMT_ROOT" "data.evals.phomt_dev.root=$PHOMT_ROOT")
  [ "$PREFLIGHT" = 1 ] && s1+=("data.train.options.max_train_samples=20000")
  train_stage "$S1" configs/pretrain/stage1_teacher.yaml "${s1[@]}"
fi
if wanted stage2; then
  need "$S1/checkpoints/best.pt" "Stage 1 chưa xong"
  train_stage "$S2" configs/pretrain/stage2_contrastive.yaml "${CORPUS[@]}" \
    "init_from.path=$S1/checkpoints/best.pt"
fi
if wanted stage3; then
  need "$S2/checkpoints/best.pt" "Stage 2 chưa xong"
  train_stage "$S3" configs/pretrain/stage3_end2end.yaml "${CORPUS[@]}" \
    "init_from.path=$S2/checkpoints/best.pt"
fi

# ------------------------------------------------------------------ v2 Phase 1: distill + dense align
if wanted distill; then
  d=("${COMMON[@]}")
  [ -n "${DATA_ROOT:-}" ] && d+=("data.train.options.sources.captions.root=$DATA_ROOT")
  [ -n "${PHOMT_ROOT:-}" ] && d+=("data.train.options.sources.phomt.root=$PHOMT_ROOT" "data.evals.phomt_dev.root=$PHOMT_ROOT")
  [ "${DISTILL_PHOMT:-1}" = 0 ] && d+=("data.train.options.sources.phomt=null")
  [ "$PREFLIGHT" = 1 ] && [ "${DISTILL_PHOMT:-1}" != 0 ] && d+=("data.train.options.sources.phomt.options.max_train_samples=20000")
  train_stage "$D2" configs/v2/distill_text.yaml "${d[@]}"
fi
if wanted align; then
  need "$D2/checkpoints/best.pt" "distill chưa xong"
  train_stage "$A2" configs/v2/align.yaml "${CORPUS[@]}" "init_from.path=$D2/checkpoints/best.pt"
fi

# ------------------------------------------------------------------ Phase 2: upcycling + MoE
if wanted upcycle; then
  if [ "$PIPELINE" = v2 ]; then
    upcycle_step "$A2/checkpoints/best.pt" configs/v2/upcycle.yaml "$UP2"
  else
    upcycle_step "$S3/checkpoints/best.pt" configs/moe/upcycle.yaml "$UP"
  fi
fi
if wanted sparse; then
  if [ "$PIPELINE" = v2 ]; then up=$UP2 sp_dir=$SP2 sp_cfg=configs/v2/sparse.yaml
  else up=$UP sp_dir=$SP sp_cfg=configs/moe/sparse_train.yaml; fi
  need "$up" "chưa upcycle"
  sp=("${CORPUS[@]}" "model.from_checkpoint=$up" "init_from.path=$up")
  [ "$PREFLIGHT" = 1 ] && sp+=("analysis.routing_probe.every_steps=${PREFLIGHT_STEPS:-4}")
  train_stage "$sp_dir" "$sp_cfg" "${sp[@]}"
fi

log "xong. Kết quả: $RUN_DIR/*/final_metrics.json, TensorBoard: tensorboard --logdir $RUN_DIR"
