# Cấu hình dùng chung cho scripts/gpu/*.sh (được `source`, không chạy trực tiếp).
# Sửa đường dẫn một lần cho máy GPU, hoặc ghi đè khi chạy: DATA_DIR=/mnt/data bash scripts/gpu/1_pull_data.sh
# DRY_RUN=1 chỉ in các lệnh nặng (train, tải, cài đặt) mà không chạy.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

# ---------------------------------------------------------------- dữ liệu (kéo từ HuggingFace ở bước 1)
export DATA_DIR=${DATA_DIR:-/data/vivlmoe}            # cần ~110 GB trống
export DATA_ROOT=${DATA_ROOT:-$DATA_DIR/dataset}       # images/ + corpus/pretrain_corpus.jsonl
export PHOMT_ROOT=${PHOMT_ROOT:-$DATA_DIR/PhoMT}
export KTVIC_ROOT=${KTVIC_ROOT:-$DATA_DIR/ktvic_dataset}
export UITVIIC_ROOT=${UITVIIC_ROOT:-$DATA_DIR/uitvic_dataset}
# Dataset downstream không có trên HF: đặt đường dẫn khi đã có file trên máy (để trống = bước 6 bỏ qua).
export OPENVIVQA_ROOT=${OPENVIVQA_ROOT:-}
export VITEXTVQA_ROOT=${VITEXTVQA_ROOT:-}
export VIOCRVQA_ROOT=${VIOCRVQA_ROOT:-}
export VIMACSA_ROOT=${VIMACSA_ROOT:-}

export HF_REPO=${HF_REPO:-nhq188/vivlmoe-pretrain}                  # ảnh + corpus (public, tạm thời)
export HF_EXTRAS_REPO=${HF_EXTRAS_REPO:-nhq188/vivlmoe-pretrain-extras}  # PhoMT, KTVIC, UIT-ViIC (private)
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}                  # LFS cổ điển: ổn định hơn hf_xet

# ---------------------------------------------------------------- môi trường, run, log
export VENV=${VENV:-$REPO/.venv}
RUNS=${RUNS:-runs}                         # thư mục run thật (bước 4 luôn dùng runs_preflight/)
LOG_DIR=${LOG_DIR:-$REPO/logs}
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p "$LOG_DIR"

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { echo "LỖI: $*" >&2; exit 1; }
need() { [ -e "$1" ] || die "thiếu $1 ($2)"; }
run() { echo "+ $*"; if [ "${DRY_RUN:-0}" != 1 ]; then "$@"; fi; }

activate() {
  if [ "${DRY_RUN:-0}" = 1 ] && [ ! -f "$VENV/bin/activate" ]; then return 0; fi
  [ -f "$VENV/bin/activate" ] || die "chưa có môi trường $VENV: chạy bash scripts/gpu/0_setup.sh"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  export PYTHON=python
}
