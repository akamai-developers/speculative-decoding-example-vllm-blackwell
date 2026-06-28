#!/usr/bin/env bash
set -euo pipefail

echo "TARGET_MODEL=$TARGET_MODEL"

streamlit run app/streamlit.py \
  --server.address 0.0.0.0 \
  --server.port 8501