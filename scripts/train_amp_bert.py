#!/usr/bin/env python
"""Fine-tune ProtBERT-BFD into AMP-BERT on the Veltri training set (server / CLI).

Example
-------
    python scripts/train_amp_bert.py --epochs 15 --model-dir models/amp_bert

The resulting model is what notebook 02 / scripts/test_amp_bert.py load for evaluation.
"""
import argparse
import pathlib
import sys

# Run without `pip install -e .` by putting src/ on the path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from transformers import Trainer  # noqa: E402

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
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=1, help="per-device train batch size")
    p.add_argument("--grad-accum", type=int, default=64, help="gradient accumulation (effective batch = batch-size * this)")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-fp16", action="store_true", help="disable mixed precision (use on CPU / unsupported GPUs)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[train] data={args.train_csv} epochs={args.epochs} "
          f"effective_batch={args.batch_size * args.grad_accum} seed={args.seed}")

    df = load_dataset(args.train_csv, shuffle=True, random_state=args.seed)
    train_dataset = AmpDataset(df, max_len=args.max_len)
    print(f"[train] {len(train_dataset)} examples")

    training_args = build_training_args(
        args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=args.weight_decay,
        fp16=not args.no_fp16,
        seed=args.seed,
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    trainer.save_model(args.model_dir)
    print(f"[train] saved model to {args.model_dir}")


if __name__ == "__main__":
    main()
