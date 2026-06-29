#!/usr/bin/env bash
set -euo pipefail

# demo 1-2
vllm serve "$TARGET_MODEL"  \
  --served-model-name qwen3-32b-fp4-spec \
  --host 0.0.0.0 \
  --port 8001 \
  --gpu-memory-utilization 0.46 \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --kv-cache-dtype auto \
  --speculative-config "{\"method\":\"draft_model\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":4}"


# demo 3
# vllm serve "$TARGET_MODEL" \
#   --served-model-name qwen3-32b-fp4-spec \
#   --host 0.0.0.0 \
#   --port 8001 \
#   --gpu-memory-utilization 0.46 \
#   --max-model-len 1024 \
#   --max-num-seqs 32 \
#   --kv-cache-dtype auto \
#   --speculative-config "{\"method\":\"draft_model\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":4}"