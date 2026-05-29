#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

set -a
source .env
set +a

mkdir -p "$MODELS_DIR"

hf download "$TARGET_MODEL_HF" \
  --local-dir "$TARGET_MODEL"

hf download "$DRAFT_MODEL_HF" \
  --local-dir "$DRAFT_MODEL"