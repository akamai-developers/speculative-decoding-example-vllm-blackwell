#!/usr/bin/env bash
set -euo pipefail

GPU_MEMORY_UTILIZATION=0.52
NUM_SPECULATIVE_TOKENS=5

vllm serve "$TARGET_MODEL" \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --speculative-config "{\"method\":\"draft_model\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS}"