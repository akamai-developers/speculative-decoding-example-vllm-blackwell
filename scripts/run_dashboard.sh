#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

set -a
source .env
set +a

streamlit run dashboard/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501