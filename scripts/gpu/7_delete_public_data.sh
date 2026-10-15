#!/usr/bin/env bash
# Bước 7 (sau khi train xong hẳn): xoá repo HF public chứa ảnh + corpus ($HF_REPO). KHÔNG hoàn tác được.
# Repo extras (private: PhoMT, KTVIC, UIT-ViIC) không bị đụng tới. Phải gõ lại đúng tên repo để xác nhận.
#
#   bash scripts/gpu/7_delete_public_data.sh
source "$(dirname "$0")/env.sh"
activate

echo "Sắp XOÁ VĨNH VIỄN repo dataset HuggingFace: $HF_REPO"
read -r -p "Gõ lại đúng tên repo để xác nhận: " typed
[ "$typed" = "$HF_REPO" ] || die "tên không khớp, không xoá gì"
run python -c "from huggingface_hub import HfApi; HfApi().delete_repo('$HF_REPO', repo_type='dataset')"
log "đã xoá $HF_REPO"
