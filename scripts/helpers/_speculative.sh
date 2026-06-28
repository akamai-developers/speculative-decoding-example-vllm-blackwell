#!/usr/bin/env bash
set -euo pipefail

NUM_SPECULATIVE_TOKENS=5

vllm serve "$TARGET_MODEL" \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype bfloat16 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048 \
  --speculative-config "{\"method\":\"draft_model\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS}"