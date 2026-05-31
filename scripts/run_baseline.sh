#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

DTYPE="bfloat16"
MAX_MODEL_LEN=2048
GPU_MEMORY_UTILIZATION=0.35 # Fixed static slice of your 96GB VRAM

vllm serve "$TARGET_MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN"