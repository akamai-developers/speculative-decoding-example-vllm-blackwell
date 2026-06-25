#!/usr/bin/env bash
set -euo pipefail

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

echo "Downloading target model: $TARGET_MODEL_HF..."
hf download "$TARGET_MODEL_HF" \
  --local-dir "$TARGET_MODEL"

echo "Downloading draft model: $DRAFT_MODEL_HF..."
hf download "$DRAFT_MODEL_HF" \
  --local-dir "$DRAFT_MODEL"