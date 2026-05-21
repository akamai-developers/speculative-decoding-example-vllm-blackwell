#!/usr/bin/env bash
set -euo pipefail

source config/baseline.env

vllm serve "$TARGET_MODEL" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --enable-metrics