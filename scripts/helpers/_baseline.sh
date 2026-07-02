#!/usr/bin/env bash
set -euo pipefail

# # DEMO1
# vllm serve "$TARGET_MODEL" \
#   --host 0.0.0.0 \
#   --port 8000 \
#   --gpu-memory-utilization 0.40 \
#   --max-model-len 8192 \
#   --max-num-seqs 1


# DEMO 1-2
vllm serve "$TARGET_MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.40 \
  --max-model-len 32768 \
  --max-num-seqs 1


# # DEMO 3
# vllm serve "$TARGET_MODEL" \
#   --host 0.0.0.0 \
#   --port 8000 \
#   --gpu-memory-utilization 0.40 \
#   --max-model-len 8192 \
#   --max-num-seqs 16