#!/usr/bin/env python
"""Evaluate AMP-BERT on the external test set (DRAMP AMPs ∪ AMPEP non-AMPs).

Example
-------
    python scripts/test_amp_bert.py --model-dir models/amp_bert

Reports SN / SP / F1 / ACC / AUROC / AUPR (plus precision, MCC), writes per-sequence
predictions to --pred-csv, and logs everything to --log-file.
"""
import argparse
import logging
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
from amp_bert.metrics import binary_metrics  # noqa: E402
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
    p.add_argument("--log-file", default=str(RESULTS_DIR / "amp_bert_test.log"),
                   help="write all results to this log file (also printed to console)")
    p.add_argument("--batch-size", type=int, default=56, help="per-device eval batch size")
    p.add_argument("--max-len", type=int, default=200)
    return p.parse_args()


def setup_logging(log_file):
    pathlib.Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [test] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    return logging.getLogger("test")


def main():
    args = parse_args()
    log = setup_logging(args.log_file)
    log.info("logging to %s", args.log_file)

    if torch.cuda.is_available():
        log.info("device: cuda — %s", torch.cuda.get_device_name(0))
    else:
        log.info("device: cpu (no CUDA detected)")

    if not pathlib.Path(args.model_dir).exists():
        log.error("model not found: %s — run train_amp_bert.py first", args.model_dir)
        sys.exit(1)
    log.info("model: %s", args.model_dir)

    amp_df = load_dataset(args.amp_csv)
    nonamp_df = load_dataset(args.nonamp_csv)
    test_df = pd.concat([amp_df, nonamp_df])
    log.info("AMP=%d non-AMP=%d total=%d", len(amp_df), len(nonamp_df), len(test_df))
    test_dataset = AmpDataset(test_df, max_len=args.max_len)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    # predict-only: output_dir is required by TrainingArguments but nothing is written there
    eval_args = TrainingArguments(
        output_dir=str(RESULTS_DIR / "amp_bert_test"),
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
    )
    trainer = Trainer(model=model, args=eval_args, compute_metrics=compute_metrics)

    predictions, labels, _ = trainer.predict(test_dataset)
    m = binary_metrics(labels, predictions)
    log.info("metrics:")
    log.info("    SN    (sensitivity): %.4f", m["sensitivity"])
    log.info("    SP    (specificity): %.4f", m["specificity"])
    log.info("    F1                 : %.4f", m["f1"])
    log.info("    ACC   (accuracy)   : %.4f", m["accuracy"])
    log.info("    AUROC              : %.4f", m.get("auroc", float("nan")))
    log.info("    AUPR               : %.4f", m.get("aupr", float("nan")))
    log.info("    (precision=%.4f, MCC=%.4f)", m["precision"], m["mcc"])

    pathlib.Path(args.pred_csv).parent.mkdir(parents=True, exist_ok=True)
    out = test_df.copy()
    out["pred"] = predictions.argmax(-1)
    out.to_csv(args.pred_csv)
    log.info("wrote predictions to %s", args.pred_csv)


if __name__ == "__main__":
    main()
