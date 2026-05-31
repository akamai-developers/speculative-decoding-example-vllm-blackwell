#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

mkdir -p "$MODELS_DIR"

hf download "$TARGET_MODEL_HF" \
  --local-dir "$TARGET_MODEL"

hf download "$DRAFT_MODEL_HF" \
  --local-dir "$DRAFT_MODEL"