#!/usr/bin/env bash
set -euo pipefail

vllm serve "$TARGET_MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048