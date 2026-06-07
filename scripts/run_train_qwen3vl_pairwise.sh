#!/bin/bash

# Training script for Qwen3-VL-4B-Thinking on MedThink Pairwise Dataset
# Usage: bash run_train_qwen3vl_pairwise.sh [NUM_GPUS]

NUM_GPUS=${1:-4}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_FACTORY_DIR="/path/to/LLaMA-Factory"

cd $LLAMA_FACTORY_DIR

# Run training with deepspeed
deepspeed --num_gpus=$NUM_GPUS src/train.py \
    $SCRIPT_DIR/qwen3vl_4b_pairwise_sft.yaml
