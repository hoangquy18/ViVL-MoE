#!/usr/bin/env bash
# Bước 3: hai phép kiểm tra trước khi train (docs/runbook.md mục 2.5). Gửi lại log cho Claude.
#   a) probe_vision: truy xuất zero-shot tiếng Việt trên KTVIC/UIT-ViIC dev của SigLIP2-g, SigLIP2-So400m và
#      C-RADIOv4 -> chọn vision encoder (cột rsum); số của siglip2-g cũng là chất lượng teacher bước distill.
#   b) check_llm_head: Qwen3.5-0.8B load đủ trọng số, perplexity tiếng Việt, gradient, sinh theo batch.
#
#   bash scripts/gpu/3_checks.sh          # ~20-40 phút trên 1 A40
source "$(dirname "$0")/env.sh"
activate

run python scripts/env_check.py --v2 2>&1 | tee "$LOG_DIR/3_env_check.log"

log "a) probe_vision (kết quả: $RUNS/probe_vision.json)"
run python scripts/probe_vision.py --out "$RUNS/probe_vision.json" \
  "data.evals.ktvic_dev.root=$KTVIC_ROOT" "data.evals.uitviic_dev.root=$UITVIIC_ROOT" \
  2>&1 | tee "$LOG_DIR/3_probe_vision.log"

log "b) check_llm_head (mọi dòng phải là PASS)"
llm_ok=1
run python scripts/check_llm_head.py --model Qwen/Qwen3.5-0.8B --llm-class Qwen3_5ForCausalLM \
  2>&1 | tee "$LOG_DIR/3_check_llm_head.log" || llm_ok=0

log "tóm tắt"
grep -E "rsum" "$LOG_DIR/3_probe_vision.log" || true
grep -E "^(PASS|FAIL)" "$LOG_DIR/3_check_llm_head.log" || true
[ "$llm_ok" = 1 ] || die "check_llm_head có FAIL: gửi $LOG_DIR/3_check_llm_head.log cho Claude trước khi train"
log "xong bước 3 (log trong $LOG_DIR/3_*.log). Tiếp theo: bash scripts/gpu/4_preflight.sh"
