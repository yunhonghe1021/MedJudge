#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-/path/to/Medthink/scripts/llamafactory_2026jan/qwen3vl_8b_medjudge_sft.yaml}"
NUM_GPUS="${2:-4}"
LLAMA_FACTORY_DIR="/path/to/Medthink/LLaMA-Factory"

cd "$LLAMA_FACTORY_DIR"

if command -v deepspeed >/dev/null 2>&1; then
  deepspeed --num_gpus="$NUM_GPUS" src/train.py "$CONFIG_PATH"
else
  llamafactory-cli train "$CONFIG_PATH"
fi
