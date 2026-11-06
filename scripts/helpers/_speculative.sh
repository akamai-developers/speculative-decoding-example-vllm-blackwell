#!/usr/bin/env bash
set -euo pipefail

# Settings come from config.env via start_vllm.sh (pick a profile there).
# SPEC_EXTRA_ARGS lets you add flags for one profile only (e.g. per-request metrics).
vllm serve "$TARGET_MODEL" \
  --host 0.0.0.0 \
  --port 8001 \
  --gpu-memory-utilization "$SPEC_GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --speculative-config '{"method":"draft_model","model":"'"$DRAFT_MODEL"'","num_speculative_tokens":'"$NUM_SPEC_TOKENS"'}' \
  ${SPEC_EXTRA_ARGS:-}
