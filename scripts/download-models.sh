#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/root/models}"

mkdir -p "$MODEL_DIR"

echo "Checking Hugging Face auth..."
hf auth whoami

echo "Downloading draft model..."
hf download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir "$MODEL_DIR/Llama-3.2-1B-Instruct"

echo "Downloading target model..."
hf download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir "$MODEL_DIR/Llama-3.1-8B-Instruct"

echo "Done. Models downloaded to $MODEL_DIR"