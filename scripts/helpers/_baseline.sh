#!/usr/bin/env bash
set -euo pipefail

GPU_MEMORY_UTILIZATION=0.35

vllm serve "$TARGET_MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN"