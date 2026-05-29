#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

set -a
source .env
set +a

vllm serve "$TARGET_MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype "$DTYPE" \
  --gpu-memory-utilization 0.25 \
  --max-model-len "$MAX_MODEL_LEN" \