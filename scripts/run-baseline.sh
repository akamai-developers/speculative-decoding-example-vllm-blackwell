#!/usr/bin/env bash
set -euo pipefail

source config/baseline.env

python -m vllm.entrypoints.openai.api_server \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --model "$TARGET_MODEL" \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN"