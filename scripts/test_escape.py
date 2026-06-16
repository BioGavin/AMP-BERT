#!/usr/bin/env python
"""Evaluate AMP-BERT (multilabel) on the ESCAPE test split (server / CLI).

Reports per-class AP/F1 and overall mAP / F1 using ESCAPE's official metric
definition, writes per-class prediction probabilities to --pred-csv, and logs
everything to --log-file. Example:

    python scripts/test_escape.py --model-dir models/amp_bert_escape
"""
import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments  # noqa: E402

from amp_bert.config import ESCAPE_TEST_CSV, MODELS_DIR, RESULTS_DIR  # noqa: E402
from amp_bert.escape import LABEL_COLUMNS, EscapeDataset, escape_scores, load_escape  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--test", default=str(ESCAPE_TEST_CSV), help="ESCAPE Test CSV")
    p.add_argument("--model-dir", default=str(MODELS_DIR / "amp_bert_escape"), help="fine-tuned model directory")
    p.add_argument("--pred-csv", default=str(RESULTS_DIR / "escape_test_predictions.csv"))
    p.add_argument("--log-file", default=str(RESULTS_DIR / "escape_test.log"))
    p.add_argument("--batch-size", type=int, default=16, help="per-device eval batch size")
    p.add_argument("--max-len", type=int, default=200)
    return p.parse_args()


def setup_logging(log_file):
    pathlib.Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [escape-test] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    return logging.getLogger("escape-test")


def main():
    args = parse_args()
    log = setup_logging(args.log_file)
    log.info("logging to %s", args.log_file)

    if torch.cuda.is_available():
        log.info("device: cuda — %s", torch.cuda.get_device_name(0))
    else:
        log.info("device: cpu (no CUDA detected)")

    # Locate the per-fold models. Two-fold CV ensembles <model-dir>/fold1 and
    # <model-dir>/fold2; fall back to a single model directly in <model-dir>.
    root = pathlib.Path(args.model_dir)
    fold_dirs = [d for d in (root / "fold1", root / "fold2") if (d / "config.json").exists()]
    if not fold_dirs:
        if (root / "config.json").exists():
            fold_dirs = [root]
        else:
            log.error("no model found under %s — run train_escape.py first", args.model_dir)
            sys.exit(1)
    log.info("ensembling %d model(s): %s", len(fold_dirs), ", ".join(d.name for d in fold_dirs))

    test_df = load_escape(args.test)
    assert test_df["Sequence"].isna().sum() == 0, "missing sequences in test data"
    log.info("test examples=%d", len(test_df))
    test_dataset = EscapeDataset(test_df, max_len=args.max_len)

    eval_args = TrainingArguments(
        output_dir=str(RESULTS_DIR / "escape_test"),
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
    )

    # Average the sigmoid probabilities across the fold models (ESCAPE ensemble).
    labels = None
    prob_sum = None
    for d in fold_dirs:
        model = AutoModelForSequenceClassification.from_pretrained(str(d))
        trainer = Trainer(model=model, args=eval_args)
        predictions, labels, _ = trainer.predict(test_dataset)
        probs = 1.0 / (1.0 + np.exp(-predictions))  # sigmoid
        prob_sum = probs if prob_sum is None else prob_sum + probs
    y_probs = prob_sum / len(fold_dirs)
    aps, f1s, mAP, macro_f1 = escape_scores(labels, y_probs)

    log.info("per-class results:")
    for name, ap, f1 in zip(LABEL_COLUMNS, aps, f1s):
        log.info("    %-14s AP=%5.1f%%  F1=%5.1f%%", name, ap * 100, f1 * 100)
    log.info("mAP = %.1f%%  |  overall F1 = %.1f%%   (paper AMP-BERT: F1 64.7 / mAP 66.9)",
             mAP * 100, macro_f1 * 100)

    pathlib.Path(args.pred_csv).parent.mkdir(parents=True, exist_ok=True)
    out = test_df.copy()
    for i, name in enumerate(LABEL_COLUMNS):
        out[f"prob_{name}"] = y_probs[:, i]
    out.to_csv(args.pred_csv, index=False)
    log.info("wrote predictions to %s", args.pred_csv)


if __name__ == "__main__":
    main()
