#!/usr/bin/env bash
set -euo pipefail

source config/speculative.env

SPECULATIVE_CONFIG=$(cat <<EOF
{
  "method": "draft_model",
  "model": "$DRAFT_MODEL",
  "num_speculative_tokens": $NUM_SPECULATIVE_TOKENS
}
EOF
)

vllm serve "$TARGET_MODEL" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --speculative-config "$SPECULATIVE_CONFIG" \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN"