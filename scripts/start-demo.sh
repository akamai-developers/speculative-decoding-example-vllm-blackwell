#!/usr/bin/env bash
set -euo pipefail

cd /root/vllm-speculative-demo
chmod +x scripts/*.sh # make all scripts executable
mkdir -p logs

# Create Python .venv if it doesn't exist
if [ ! -d ".venv" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv .venv
else
  echo "Using existing Python virtual environment..."
fi

# Activate the .venv and install GPU/server dependencies
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt

# Hugging Face auth
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set."
  echo "Run: export HF_TOKEN='hf_xxx'"
  exit 1
fi

hf auth login --token "$HF_TOKEN"

# Download models
echo "Downloading models..."
./scripts/download-models.sh

# Start vLLM servers in background
echo "Starting baseline vLLM server..."
nohup ./scripts/run-baseline.sh > logs/baseline.log 2>&1 &

echo "Starting speculative vLLM server..."
nohup ./scripts/run-speculative.sh > logs/speculative.log 2>&1 &

INSTANCE_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "vLLM servers started."
echo "Baseline vLLM:    http://${INSTANCE_IP}:8000"
echo "Speculative vLLM: http://${INSTANCE_IP}:8001"
echo ""
echo "Check logs:"
echo "tail -f logs/baseline.log"
echo "tail -f logs/speculative.log"