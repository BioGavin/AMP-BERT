#!/usr/bin/env python
"""Fine-tune ProtBERT-BFD into AMP-BERT on the Veltri training set (server / CLI).

Example
-------
    python scripts/train_amp_bert.py --epochs 15 --model-dir models/amp_bert

The resulting model is what notebook 02 / scripts/test_amp_bert.py load for evaluation.
"""
import argparse
import logging
import pathlib
import sys

# Run without `pip install -e .` by putting src/ on the path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from transformers import Trainer, TrainerCallback  # noqa: E402

from amp_bert import (  # noqa: E402
    AmpDataset,
    build_training_args,
    compute_metrics,
    load_dataset,
    model_init,
)
from amp_bert.config import MODELS_DIR, RESULTS_DIR, TRAIN_CSV  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--train-csv", default=str(TRAIN_CSV), help="training CSV (cols: aa_seq, AMP)")
    p.add_argument("--model-dir", default=str(MODELS_DIR / "amp_bert"), help="where to save the fine-tuned model")
    p.add_argument("--output-dir", default=str(RESULTS_DIR / "amp_bert_train"), help="Trainer working dir")
    p.add_argument("--log-file", default=str(RESULTS_DIR / "amp_bert_train.log"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=1, help="per-device train batch size")
    p.add_argument("--grad-accum", type=int, default=64, help="gradient accumulation (effective batch = batch-size * this)")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-fp16", action="store_true", help="disable mixed precision (use on CPU / unsupported GPUs)")
    p.add_argument("--val-frac", type=float, default=0.0,
                   help="hold out this fraction for per-epoch eval metrics (0 = train on all data, as in the paper)")
    p.add_argument("--eval-batch-size", type=int, default=32, help="per-device eval batch size (only with --val-frac)")
    return p.parse_args()


class LogToFile(TrainerCallback):
    """Forward Trainer logs (per-epoch loss, eval metrics) to the Python logger."""
    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.logger.info("step %d | %s", state.global_step, logs)


def setup_logging(log_file):
    pathlib.Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [train] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    return logging.getLogger("train")


def main():
    args = parse_args()
    log = setup_logging(args.log_file)
    log.info("logging to %s", args.log_file)
    if torch.cuda.is_available():
        log.info("device: cuda — %s (%d GPU visible)", torch.cuda.get_device_name(0), torch.cuda.device_count())
    else:
        log.info("device: cpu (no CUDA detected — training will be very slow)")
    log.info("data=%s epochs=%d effective_batch=%d seed=%d",
             args.train_csv, args.epochs, args.batch_size * args.grad_accum, args.seed)

    df = load_dataset(args.train_csv, shuffle=True, random_state=args.seed)

    # Optional held-out validation set for per-epoch metrics (off by default).
    eval_dataset = None
    eval_kwargs = {}
    if args.val_frac > 0:
        n_val = int(len(df) * args.val_frac)
        val_df, train_df = df.iloc[:n_val], df.iloc[n_val:]
        eval_dataset = AmpDataset(val_df, max_len=args.max_len)
        eval_kwargs = dict(eval_strategy="epoch", per_device_eval_batch_size=args.eval_batch_size)
        log.info("%d train / %d val examples (per-epoch eval ON)", len(train_df), len(val_df))
    else:
        train_df = df
        log.info("%d examples (training on all data; no per-epoch eval)", len(train_df))
    train_dataset = AmpDataset(train_df, max_len=args.max_len)

    training_args = build_training_args(
        args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=args.weight_decay,
        fp16=not args.no_fp16,
        seed=args.seed,
        logging_strategy="epoch",   # log mean training loss after every epoch
        **eval_kwargs,
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[LogToFile(log)],
    )
    trainer.train()

    trainer.save_model(args.model_dir)
    log.info("saved model to %s", args.model_dir)


if __name__ == "__main__":
    main()
