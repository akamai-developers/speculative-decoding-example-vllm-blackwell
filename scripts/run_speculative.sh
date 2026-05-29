#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

set -a
source .env
set +a

SPECULATIVE_CONFIG=$(cat <<EOF
{
  "method": "draft_model",
  "model": "$DRAFT_MODEL",
  "num_speculative_tokens": $NUM_SPECULATIVE_TOKENS
}
EOF
)

vllm serve "$TARGET_MODEL" \
  --host 0.0.0.0 \
  --port 8001 \
  --speculative-config "$SPECULATIVE_CONFIG" \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN"