#!/usr/bin/env bash
set -euo pipefail


# demo 1-2
vllm serve "$TARGET_MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.44 \
  --max-model-len 32000 \
  --max-num-seqs 1 \

# demo 3
# vllm serve "$TARGET_MODEL" \
#   --host 0.0.0.0 \
#   --port 8000 \
#   --gpu-memory-utilization 0.44 \
#   --max-model-len 1024 \
#   --max-num-seqs 32 \