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
export CKPT_REPO=${CKPT_REPO:-nhq188/vivlmoe-v2-checkpoints}         # bước 5b: sao lưu checkpoint (private)
# Tải xuống: Xet (hf_xet) nhanh và bền hơn đường xet-bridge dùng khi tắt Xet (đọc hay quá hạn 10 s rồi resume).
# Upload từ Mac mới cần HF_HUB_DISABLE_XET=1 (runbook 2.3).
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-0}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-60}       # giây chờ mỗi lần đọc (mặc định HF: 10)
export PULL_WORKERS=${PULL_WORKERS:-4}                            # số shard tải song song ở bước 1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=${HF_XET_CHUNK_CACHE_SIZE_BYTES:-0}  # tắt chunk cache Xet (mặc định 10 GB)

# Chỉ pretrain (mặc định): không tải SigLIP2-So400m và Qwen3.5 (~6 GB), bước 3 so C-RADIO với SigLIP2-g và
# bỏ check_llm_head. PRETRAIN_ONLY=0 cho cả downstream.
export PRETRAIN_ONLY=${PRETRAIN_ONLY:-1}

# ---------------------------------------------------------------- môi trường, run, log
export VENV=${VENV:-$REPO/.venv}
RUNS=${RUNS:-runs}                         # thư mục run thật (bước 4 luôn dùng runs_preflight/)
LOG_DIR=${LOG_DIR:-$REPO/logs}
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}  # less fragmentation
mkdir -p "$LOG_DIR"

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { echo "LỖI: $*" >&2; exit 1; }
need() { [ -e "$1" ] || die "thiếu $1 ($2)"; }
run() { echo "+ $*"; if [ "${DRY_RUN:-0}" != 1 ]; then "$@"; fi; }

# Precision theo GPU, dùng chung cho pretrain, downstream và check_llm_head. GPU chưa có bf16 thật (compute
# capability < 8.0: Turing như Quadro RTX 8000, Volta): bf16 ở đó là giả lập, chậm ~10 lần fp16, nên train fp16
# (GradScaler). GradScaler bắt đầu ở 2^16 và giảm nửa sau mỗi step tràn số; gradient của sigmoid loss (tổng trên 256
# text, nhân t = 10) cần ~5-7 lần giảm ở đầu align/sparse, quá ngưỡng mặc định 5 step bị bỏ liên tiếp, nên nới lên 16
# (vẫn bắt được run hỏng thật: sau 16 lần, scale < 1). GPU_CC ghi đè phần tự dò.
gpu_cc() {  # compute capability nhỏ nhất trong máy; rỗng khi không có nvidia-smi
  if [ -n "${GPU_CC:-}" ]; then echo "$GPU_CC"; return 0; fi
  command -v nvidia-smi >/dev/null || return 0
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -n | head -1 || true
}
amp_dtype() {  # dtype AMP cho compute capability $1: fp16 dưới 8.0, còn lại (kể cả không rõ) bf16 như config
  if [ -n "${1:-}" ] && [ "${1%%.*}" -lt 8 ]; then echo fp16; else echo bf16; fi
}
amp_overrides() {  # override cho train.py theo compute capability $1 (rỗng: giữ bf16 của config)
  [ "$(amp_dtype "${1:-}")" = fp16 ] || return 0
  echo "train.amp.dtype=fp16 train.max_consecutive_nonfinite=16"
}

# Batch pretrain theo phần cứng, giữ đúng công thức của config: 256 cặp mỗi step cho sigmoid loss (text được
# all-gather giữa các GPU, nên số cặp = batch mỗi GPU x số GPU) và 512 mẫu mỗi bước tối ưu (x accum).
# Card < 40 GB nhận tối đa 128 mẫu (256 không vừa 24 GB). Ví dụ: 1 A40 -> 256 x accum 2; 2x 3090 -> 128 x 2 GPU
# x accum 2; 1x 3090 -> 128 x accum 4 (còn 128 cặp mỗi step). NGPU / GPU_MEM_GB / GPU_CC ghi đè phần tự dò.
# Với fp16, teacher của distill cũng chạy fp16 (C-RADIO và SigLIP2-g đã kiểm: không tràn số, lệch fp32 < 0,3%).
pretrain_overrides() {
  local ngpu=${NGPU:-} mem=${GPU_MEM_GB:-} cc=${GPU_CC:-}
  if [ -z "$ngpu" ] || [ -z "$mem" ]; then
    command -v nvidia-smi >/dev/null || { echo ""; return 0; }
    ngpu=${NPROC:-$(nvidia-smi -L | grep -c '^GPU')}
    mem=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1) / 1024 ))
    cc=${cc:-$(gpu_cc)}
  fi
  local cap=256 per accum out amp
  [ "$mem" -ge 40 ] || cap=128
  per=$((256 / ngpu))
  [ "$per" -le "$cap" ] || per=$cap
  [ "$per" -ge 1 ] || per=1
  accum=$(( (512 + per * ngpu - 1) / (per * ngpu) ))
  out="data.batch_size=$per train.grad_accum_steps=$accum"
  amp=$(amp_overrides "$cc")
  [ -z "$amp" ] || out="$out $amp train.amp.teacher_autocast=true"
  echo "$out"
}

activate() {
  if [ "${DRY_RUN:-0}" = 1 ] && [ ! -f "$VENV/bin/activate" ]; then return 0; fi
  [ -f "$VENV/bin/activate" ] || die "chưa có môi trường $VENV: chạy bash scripts/gpu/0_setup.sh"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  export PYTHON=python
}
