#!/usr/bin/env bash
# Train then evaluate AMP-BERT on the ESCAPE benchmark (multilabel).
#
#   bash scripts/run_escape.sh                 # full run (15 epochs)
#   EPOCHS=1 bash scripts/run_escape.sh        # quick smoke test
#   nohup bash scripts/run_escape.sh > escape.log 2>&1 &   # detached
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-15}"
MODEL_DIR="${MODEL_DIR:-models/amp_bert_escape}"

echo "==> Training on ESCAPE (epochs=$EPOCHS) -> $MODEL_DIR"
python3 scripts/train_escape.py --epochs "$EPOCHS" --model-dir "$MODEL_DIR"

echo "==> Testing on ESCAPE"
python3 scripts/test_escape.py --model-dir "$MODEL_DIR"
