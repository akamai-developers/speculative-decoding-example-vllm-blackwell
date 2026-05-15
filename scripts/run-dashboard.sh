#!/usr/bin/env bash
set -euo pipefail

streamlit run dashboard/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501