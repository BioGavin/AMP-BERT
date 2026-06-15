#!/usr/bin/env bash
# Train then evaluate AMP-BERT end to end on a server.
#
#   bash scripts/run_amp_bert.sh                 # full run (15 epochs)
#   EPOCHS=1 bash scripts/run_amp_bert.sh        # quick smoke test
#   nohup bash scripts/run_amp_bert.sh > run.log 2>&1 &   # detached, logged
#
# Pick a GPU with CUDA_VISIBLE_DEVICES=0 if the host has several.
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-15}"
MODEL_DIR="${MODEL_DIR:-models/amp_bert}"

echo "==> Training (epochs=$EPOCHS) -> $MODEL_DIR"
python scripts/train_amp_bert.py --epochs "$EPOCHS" --model-dir "$MODEL_DIR"

echo "==> Testing"
python scripts/test_amp_bert.py --model-dir "$MODEL_DIR"
