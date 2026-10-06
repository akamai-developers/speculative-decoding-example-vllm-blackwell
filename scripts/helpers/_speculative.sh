#!/usr/bin/env bash
set -euo pipefail


# # DEMO 1-2
# vllm serve "$TARGET_MODEL" \
#   --host 0.0.0.0 \
#   --port 8001 \
#   --gpu-memory-utilization 0.45 \
#   --max-model-len 36864 \
#   --max-num-seqs 1 \
#   --speculative-config "{\"method\":\"draft_model\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":5}"

# DEMO 3
vllm serve "$TARGET_MODEL" \
  --host 0.0.0.0 \
  --port 8001 \
  --gpu-memory-utilization 0.50 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --speculative-config '{"method":"draft_model","model":"'"$DRAFT_MODEL"'","num_speculative_tokens":5}' \
  --per-request-spec-decode-metrics summary