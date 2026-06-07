#!/bin/bash

# Inference script for Qwen3-VL-4B-Thinking on MedThink Pairwise Test Dataset
# Using HuggingFace Transformers (no vLLM)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd /path/to/Medthink

python scripts/inference_qwen3vl_pairwise.py
