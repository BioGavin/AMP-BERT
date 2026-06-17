#!/usr/bin/env python
"""Fine-tune AMP-BERT (multilabel) on ESCAPE with two-fold cross-validation.

ESCAPE ships the training data pre-split into Fold1 and Fold2. Two-fold CV trains
one model per fold (the other fold is the held-out validation set):

    model "fold1": train on Fold1   (validate on Fold2 if --val)
    model "fold2": train on Fold2   (validate on Fold1 if --val)

Both models are saved under --model-dir/{fold1,fold2}; test_escape.py averages
their predictions (ensemble) on the Test split, as in the ESCAPE benchmark.

    python scripts/train_escape.py --epochs 5 --model-dir models/amp_bert_escape
"""
import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from transformers import Trainer, TrainerCallback  # noqa: E402

from amp_bert import build_training_args  # noqa: E402
from amp_bert.config import (  # noqa: E402
    ESCAPE_FOLD1_CSV,
    ESCAPE_FOLD2_CSV,
    MODELS_DIR,
    RESULTS_DIR,
)
from amp_bert.escape import (  # noqa: E402
    EscapeDataset,
    escape_compute_metrics,
    escape_model_init,
    load_escape,
)


class LogToFile(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.logger.info("step %d | %s", state.global_step, logs)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--fold1", default=str(ESCAPE_FOLD1_CSV), help="ESCAPE Fold1 CSV")
    p.add_argument("--fold2", default=str(ESCAPE_FOLD2_CSV), help="ESCAPE Fold2 CSV")
    p.add_argument("--model-dir", default=str(MODELS_DIR / "amp_bert_escape"),
                   help="parent dir; the two fold models go in <dir>/fold1 and <dir>/fold2")
    p.add_argument("--output-dir", default=str(RESULTS_DIR / "escape_train"), help="Trainer working dir")
    p.add_argument("--log-file", default=str(RESULTS_DIR / "escape_train.log"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=1, help="per-device train batch size (lower if OOM)")
    p.add_argument("--grad-accum", type=int, default=64, help="gradient accumulation (effective batch = batch-size * this)")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-fp16", action="store_true", help="disable mixed precision")
    p.add_argument("--val", action="store_true",
                   help="evaluate on the held-out fold each epoch (slower; off by default)")
    p.add_argument("--eval-batch-size", type=int, default=16, help="per-device eval batch size (with --val)")
    p.add_argument("--only-fold", choices=["1", "2"], default=None,
                   help="train only this fold's model instead of both")
    return p.parse_args()


def setup_logging(log_file):
    pathlib.Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [escape-train] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    return logging.getLogger("escape-train")


def train_one(name, train_csv, val_csv, args, log):
    """Train a single model on `train_csv`, holding out `val_csv` for validation."""
    log.info("=== model '%s': train on %s ===", name, train_csv)
    train_df = load_escape(train_csv).sample(frac=1, random_state=args.seed)
    assert train_df["Sequence"].isna().sum() == 0, "missing sequences in training data"
    train_dataset = EscapeDataset(train_df, max_len=args.max_len)
    log.info("    train examples=%d  effective_batch=%d  epochs=%d",
             len(train_dataset), args.batch_size * args.grad_accum, args.epochs)

    eval_dataset, eval_kwargs = None, {}
    if args.val:
        eval_dataset = EscapeDataset(load_escape(val_csv), max_len=args.max_len)
        eval_kwargs = dict(eval_strategy="epoch", per_device_eval_batch_size=args.eval_batch_size)
        log.info("    validate on %s (%d examples) each epoch", val_csv, len(eval_dataset))

    training_args = build_training_args(
        str(pathlib.Path(args.output_dir) / name),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=args.weight_decay,
        fp16=not args.no_fp16,
        seed=args.seed,
        logging_strategy="epoch",
        **eval_kwargs,
    )
    trainer = Trainer(
        model_init=escape_model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=escape_compute_metrics,
        callbacks=[LogToFile(log)],
    )
    trainer.train()

    save_path = pathlib.Path(args.model_dir) / name
    trainer.save_model(str(save_path))
    log.info("    saved model to %s", save_path)


def main():
    args = parse_args()
    log = setup_logging(args.log_file)
    log.info("logging to %s", args.log_file)
    if torch.cuda.is_available():
        log.info("device: cuda — %s (%d GPU visible)", torch.cuda.get_device_name(0), torch.cuda.device_count())
    else:
        log.info("device: cpu (no CUDA detected — training will be very slow)")

    # name -> (train fold, held-out/validation fold)
    plan = {"fold1": (args.fold1, args.fold2), "fold2": (args.fold2, args.fold1)}
    if args.only_fold:
        plan = {f"fold{args.only_fold}": plan[f"fold{args.only_fold}"]}

    for name, (train_csv, val_csv) in plan.items():
        train_one(name, train_csv, val_csv, args, log)

    log.info("done. models in %s", args.model_dir)


if __name__ == "__main__":
    main()
