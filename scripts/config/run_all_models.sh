#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-sft}"  # sft | sft_rea | rm
NUM_GPUS="${2:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$MODE" in
  sft|sft_rea|rm) ;;
  *) echo "Usage: $0 [sft|sft_rea|rm] [NUM_GPUS]" >&2; exit 2 ;;
esac

for CONFIG in "$SCRIPT_DIR"/*_medjudge_${MODE}.yaml; do
  echo "==> Training with $CONFIG"
  "$SCRIPT_DIR/run_train.sh" "$CONFIG" "$NUM_GPUS"
done
