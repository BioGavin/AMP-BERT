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

> ESCAPE (Part 3, multilabel) currently lives only in `notebooks/02_escape_benchmark.ipynb`.
> Ask if you want matching `train_escape.py` / `test_escape.py` server scripts.
