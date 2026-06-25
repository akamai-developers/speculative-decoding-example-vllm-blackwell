#!/usr/bin/env bash
set -euo pipefail

./scripts/setup_env.sh
./scripts/start_monitor.sh
./scripts/start_vllm.sh

echo "========================================"
echo "Full demo stack is running"
echo "========================================"