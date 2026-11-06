#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/start_vllm.sh [demo12|demo3]
PROFILE="${1:-demo12}"

source ./scripts/common.sh
activate_venv

case "$PROFILE" in
  demo12)
    export BASELINE_GPU_UTIL="$DEMO12_BASELINE_GPU_UTIL"
    export SPEC_GPU_UTIL="$DEMO12_SPEC_GPU_UTIL"
    export MAX_MODEL_LEN="$DEMO12_MAX_MODEL_LEN"
    export MAX_NUM_SEQS="$DEMO12_MAX_NUM_SEQS"
    export SPEC_EXTRA_ARGS=""
    ;;
  demo3)
    export BASELINE_GPU_UTIL="$DEMO3_BASELINE_GPU_UTIL"
    export SPEC_GPU_UTIL="$DEMO3_SPEC_GPU_UTIL"
    export MAX_MODEL_LEN="$DEMO3_MAX_MODEL_LEN"
    export MAX_NUM_SEQS="$DEMO3_MAX_NUM_SEQS"
    export SPEC_EXTRA_ARGS="--per-request-spec-decode-metrics summary"
    ;;
  *)
    echo "Unknown profile: $PROFILE (use demo12 or demo3)"
    exit 1
    ;;
esac

echo "Profile: $PROFILE"
echo "  max-model-len=$MAX_MODEL_LEN  max-num-seqs=$MAX_NUM_SEQS  k=$NUM_SPEC_TOKENS"
echo "  baseline gpu-util=$BASELINE_GPU_UTIL  speculative gpu-util=$SPEC_GPU_UTIL"

# Stop any servers left over from a previous profile.
pkill -f "vllm serve" 2>/dev/null && echo "Stopping existing vLLM servers..." && sleep 10 || true

wait_for_health() {
  local name="$1" port="$2" timeout="${3:-900}"
  local start=$SECONDS
  echo "⏳ Waiting for $name server on port $port..."
  until curl -sf "http://127.0.0.1:${port}/health" > /dev/null; do
    if (( SECONDS - start > timeout )); then
      echo "❌ $name server did not become healthy within ${timeout}s. Check $LOG_DIR/${name}.log"
      exit 1
    fi
    sleep 5
  done
  echo "✅ $name server is ready ($(( SECONDS - start ))s)"
}

# Start one at a time so the two servers don't race while profiling GPU memory.
echo "Starting baseline vLLM server..."
nohup ./scripts/helpers/_baseline.sh > "$LOG_DIR/baseline.log" 2>&1 &
wait_for_health baseline 8000

echo "Starting speculative vLLM server..."
nohup ./scripts/helpers/_speculative.sh > "$LOG_DIR/speculative.log" 2>&1 &
wait_for_health speculative 8001

echo "========================================"
echo "vLLM servers ready ($PROFILE)"
echo "========================================"
echo "Baseline vLLM:    http://${INSTANCE_IP}:8000"
echo "Speculative vLLM: http://${INSTANCE_IP}:8001"
echo "vLLM version:     $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)"
