#!/usr/bin/env bash
set -euo pipefail

# Settings come from config.env via start_vllm.sh (pick a profile there).
vllm serve "$TARGET_MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization "$BASELINE_GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS"
