# Server / CLI scripts

Headless training & evaluation of AMP-BERT (the Part 2 paper reproduction),
for running on a GPU server instead of Colab. They reuse the `src/amp_bert`
package, so behaviour matches the notebooks.

## Setup (once)
```bash
pip install -r requirements.txt
pip install -e .          # optional; the scripts also add src/ to sys.path
```
A CUDA GPU is strongly recommended (ProtBERT-BFD is large). `fp16` is on by default.

## Train
```bash
python scripts/train_amp_bert.py --epochs 15 --model-dir models/amp_bert
```
Key flags (see `--help` for all): `--epochs`, `--lr`, `--batch-size`,
`--grad-accum` (effective batch = batch-size × grad-accum, default 1×64=64),
`--seed`, `--no-fp16`. Saves the fine-tuned model to `--model-dir`.

Mean **training loss is printed after every epoch**. By default training uses
all data (as in the paper), so there are no per-epoch validation metrics. To get
per-epoch accuracy/F1/AUC, hold out a validation split:
```bash
python scripts/train_amp_bert.py --val-frac 0.1   # 10% held out, evaluated each epoch
```

## Test
```bash
python scripts/test_amp_bert.py --model-dir models/amp_bert
```
Evaluates on the external test set (DRAMP AMPs ∪ AMPEP non-AMPs), prints
accuracy / F1 / precision / recall / MCC / ROC-AUC, and writes per-sequence
predictions to `results/external_test_predictions.csv`.

## Both, end to end
```bash
bash scripts/run_amp_bert.sh                       # full (15 epochs)
EPOCHS=1 bash scripts/run_amp_bert.sh              # smoke test
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_amp_bert.sh > run.log 2>&1 &
```

## ESCAPE benchmark (Part 3, multilabel)

Same idea for the ESCAPE multilabel task (5 labels), reading the reconstructed
CSVs in `data/escape/`.

```bash
python scripts/train_escape.py --epochs 5 --model-dir models/amp_bert_escape
python scripts/test_escape.py  --model-dir models/amp_bert_escape
bash scripts/run_escape.sh                    # both (EPOCHS=1 for smoke test)
```

- **train_escape.py**: **two-fold cross-validation** — trains one model on Fold1
  and another on Fold2, saved to `<model-dir>/fold1` and `<model-dir>/fold2`.
  Add `--val` to also evaluate on the held-out fold each epoch (slower);
  `--only-fold 1|2` to train just one. Logs to `results/escape_train.log`.
- **test_escape.py**: loads both fold models, **averages their sigmoid
  probabilities (ensemble)** on the Test split, then reports per-class AP/F1 and
  overall **mAP / F1** (ESCAPE's official metric). Writes per-class probabilities
  to `--pred-csv`, logs to `results/escape_test.log`. (Falls back to a single
  model placed directly in `<model-dir>`.)

> This matches ESCAPE's two-fold protocol (single seed, no multi-seed averaging).
> Each fold trains on ~33k sequences; training **two** models takes a while —
> start with `EPOCHS=1`, and use `--only-fold` to test the pipeline on one fold.
