#!/usr/bin/env python
"""Evaluate AMP-BERT on the external test set (DRAMP AMPs ∪ AMPEP non-AMPs).

Example
-------
    python scripts/test_amp_bert.py --model-dir models/amp_bert

Prints accuracy / F1 / precision / recall / MCC / ROC-AUC and writes per-sequence
predictions to --pred-csv.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from amp_bert import AmpDataset, compute_metrics, load_dataset  # noqa: E402
from amp_bert.config import (  # noqa: E402
    MODELS_DIR,
    RESULTS_DIR,
    TEST_AMP_CSV,
    TEST_NONAMP_CSV,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-dir", default=str(MODELS_DIR / "amp_bert"), help="fine-tuned model directory")
    p.add_argument("--amp-csv", default=str(TEST_AMP_CSV), help="AMP test CSV (DRAMP)")
    p.add_argument("--nonamp-csv", default=str(TEST_NONAMP_CSV), help="non-AMP test CSV (AMPEP)")
    p.add_argument("--pred-csv", default=str(RESULTS_DIR / "external_test_predictions.csv"))
    p.add_argument("--output-dir", default=str(RESULTS_DIR / "amp_bert_test"), help="Trainer working dir")
    p.add_argument("--batch-size", type=int, default=56, help="per-device eval batch size")
    p.add_argument("--max-len", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        print(f"[test] device: cuda — {torch.cuda.get_device_name(0)}")
    else:
        print("[test] device: cpu (no CUDA detected)")

    if not pathlib.Path(args.model_dir).exists():
        sys.exit(f"[test] model not found: {args.model_dir} — run train_amp_bert.py first")

    amp_df = load_dataset(args.amp_csv)
    nonamp_df = load_dataset(args.nonamp_csv)
    test_df = pd.concat([amp_df, nonamp_df])
    print(f"[test] AMP={len(amp_df)} non-AMP={len(nonamp_df)} total={len(test_df)}")
    test_dataset = AmpDataset(test_df, max_len=args.max_len)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    eval_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
    )
    trainer = Trainer(model=model, args=eval_args, compute_metrics=compute_metrics)

    predictions, _, metrics = trainer.predict(test_dataset)
    print("[test] metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    pathlib.Path(args.pred_csv).parent.mkdir(parents=True, exist_ok=True)
    out = test_df.copy()
    out["pred"] = predictions.argmax(-1)
    out.to_csv(args.pred_csv)
    print(f"[test] wrote predictions to {args.pred_csv}")


if __name__ == "__main__":
    main()
