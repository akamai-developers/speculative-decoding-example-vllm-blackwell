#!/usr/bin/env bash
python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
pip install --upgrade pip
pip install vllm huggingface_hub