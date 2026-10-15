#!/usr/bin/env bash
# Bước 0: môi trường Python 3.11 cho v2 (torch >= 2.6, transformers >= 5.2, timm) trong .venv của repo.
#
#   bash scripts/gpu/0_setup.sh
#   INSTALL_FAST_KERNELS=1 bash scripts/gpu/0_setup.sh   # thêm kernel nhanh cho attention tuyến tính của
#                                                        # Qwen3.5 (cần nvcc; không bắt buộc)
# Cần: driver NVIDIA (nvidia-smi), python3.11 (hoặc conda để tạo Python 3.11).
source "$(dirname "$0")/env.sh"

command -v nvidia-smi >/dev/null || die "không thấy nvidia-smi: máy chưa có driver GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Python 3.11 (pyproject: >=3.11,<3.12)
PY=${PYTHON311:-}
if [ -z "$PY" ]; then
  for cand in python3.11 python3; do
    if command -v "$cand" >/dev/null && "$cand" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))'; then
      PY=$cand
      break
    fi
  done
fi
if [ -z "$PY" ] && command -v conda >/dev/null; then
  log "không có python3.11 trên PATH -> tạo Python 3.11 bằng conda ở $REPO/.conda311"
  [ -x "$REPO/.conda311/bin/python" ] || run conda create -y -p "$REPO/.conda311" python=3.11
  PY=$REPO/.conda311/bin/python
fi
[ -n "$PY" ] || die "cần Python 3.11: cài python3.11 (vd. apt install python3.11 python3.11-venv) hoặc conda"

if [ ! -x "$VENV/bin/python" ]; then
  log "tạo môi trường $VENV bằng $PY"
  run "$PY" -m venv "$VENV"
fi
run "$VENV/bin/pip" install -U pip wheel
run "$VENV/bin/pip" install -r requirements-v2.txt
run "$VENV/bin/pip" install -e ".[dev]"      # dev: pytest (env_check cần), ruff
if [ "${INSTALL_FAST_KERNELS:-0}" = 1 ]; then
  run "$VENV/bin/pip" install flash-linear-attention causal-conv1d \
    || log "không cài được kernel nhanh (không bắt buộc: Qwen3.5 sẽ dùng bản torch, chậm hơn)"
fi

log "kiểm tra torch + GPU và môi trường v2"
run "$VENV/bin/python" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
run "$VENV/bin/python" scripts/env_check.py --v2
log "xong bước 0. Tiếp theo: bash scripts/gpu/1_pull_data.sh"
