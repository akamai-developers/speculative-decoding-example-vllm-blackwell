#!/usr/bin/env bash
set -euo pipefail

source config/baseline.env
source config/speculative.env

mkdir -p models

hf auth whoami

echo "Downloading target model..."
hf download "$TARGET_MODEL_HF" --local-dir "$TARGET_MODEL"

echo "Downloading draft model..."
hf download "$DRAFT_MODEL_HF" --local-dir "$DRAFT_MODEL"