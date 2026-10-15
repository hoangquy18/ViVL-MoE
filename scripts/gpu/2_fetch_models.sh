#!/usr/bin/env bash
# Bước 2: tải model v2 vào cache HF (C-RADIOv4-SO400M, mmBERT-base, SigLIP 2 g + So400m, Qwen3.5-0.8B; ~15 GB),
# nạp sẵn code remote của C-RADIO, rồi ghim commit của C-RADIO vào configs/model/v2.yaml (model.vision.revision)
# nếu đang để null, để mọi bước train dùng đúng một phiên bản code remote.
#
#   bash scripts/gpu/2_fetch_models.sh
source "$(dirname "$0")/env.sh"
activate

run python scripts/fetch_models.py --only v2 2>&1 | tee -a "$LOG_DIR/2_fetch_models.log"

[ "${DRY_RUN:-0}" = 1 ] && { log "DRY_RUN: bỏ qua bước ghim revision"; exit 0; }
python - <<'EOF'
import re
from pathlib import Path

from huggingface_hub import snapshot_download

cfg = Path("configs/model/v2.yaml")
text = cfg.read_text(encoding="utf-8")
m = re.search(r"name: nvidia/C-RADIOv4-SO400M, revision: ([^,}\s]+)", text)
if m is None:
    raise SystemExit(f"{cfg}: không thấy dòng vision của C-RADIO")
if m.group(1) != "null":
    print(f"C-RADIO đã được ghim: revision {m.group(1)}")
else:
    sha = Path(snapshot_download("nvidia/C-RADIOv4-SO400M", local_files_only=True)).name
    cfg.write_text(text.replace(m.group(0), m.group(0).replace("null", sha)), encoding="utf-8")
    print(f"đã ghim C-RADIO revision {sha} vào {cfg} (nên commit + push thay đổi này)")
EOF
log "xong bước 2. Tiếp theo: bash scripts/gpu/3_checks.sh"
