#!/usr/bin/env bash
set -euo pipefail


streamlit run app/dashboard.py \
  --server.address 0.0.0.0 \
  --server.port 8501