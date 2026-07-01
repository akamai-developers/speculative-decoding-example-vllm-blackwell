#!/usr/bin/env bash
set -euo pipefail

# demo 1-2
vllm serve "$TARGET_MODEL"  \
  --host 0.0.0.0 \
  --port 8001 \
  --gpu-memory-utilization 0.46 \
  --max-model-len 32000 \
  --max-num-seqs 1 \
  --speculative-config "{\"method\":\"draft_model\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":5}"


# demo 3
# vllm serve "$TARGET_MODEL" \
#   --host 0.0.0.0 \
#   --port 8001 \
#   --gpu-memory-utilization 0.46 \
#   --max-model-len 1024 \
#   --max-num-seqs 32 \
#   --speculative-config "{\"method\":\"draft_model\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":4}"