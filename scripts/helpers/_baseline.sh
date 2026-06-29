#!/usr/bin/env bash
set -euo pipefail


# demo 1-2
vllm serve "$TARGET_MODEL" \
  --served-model-name qwen3-32b-fp4-baseline \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.44 \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --kv-cache-dtype auto \

# demo 3
# vllm serve "$TARGET_MODEL" \
#   --served-model-name qwen3-32b-fp4-baseline \
#   --host 0.0.0.0 \
#   --port 8000 \
#   --gpu-memory-utilization 0.44 \
#   --max-model-len 1024 \
#   --max-num-seqs 32 \
#   --kv-cache-dtype auto