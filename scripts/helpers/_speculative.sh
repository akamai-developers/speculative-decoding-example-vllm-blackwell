#!/usr/bin/env bash
set -euo pipefail

# # DEMO1
# vllm serve "$TARGET_MODEL" \
#   --host 0.0.0.0 \
#   --port 8001 \
#   --gpu-memory-utilization 0.50 \
#   --max-model-len 8192 \
#   --max-num-seqs 1 \
#   --speculative-config '{"method":"draft_model","model":"'"$DRAFT_MODEL"'","num_speculative_tokens":5}'


# DEMO 1-2
vllm serve "$TARGET_MODEL" \
  --host 0.0.0.0 \
  --port 8001 \
  --gpu-memory-utilization 0.50 \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --speculative-config '{"method":"draft_model","model":"'"$DRAFT_MODEL"'","num_speculative_tokens":5}'
#   --speculative-config "{\"method\":\"draft_model\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":5}"


# # DEMO 3
# vllm serve "$TARGET_MODEL" \
#   --host 0.0.0.0 \
#   --port 8001 \
#   --gpu-memory-utilization 0.50 \
#   --max-model-len 8192 \
#   --max-num-seqs 16 \
#   --speculative-config '{"method":"draft_model","model":"'"$DRAFT_MODEL"'","num_speculative_tokens":5}'