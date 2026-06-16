"""ESCAPE benchmark support: multilabel AMP classification with AMP-BERT.

ESCAPE (Ojeda et al., NeurIPS 2025) predicts 5 binary labels per peptide. This
module mirrors the binary helpers in ``data.py`` / ``metrics.py`` / ``model.py``
but for the multilabel setting, and replicates ESCAPE's official metric
computation (``src/test_ESCAPE.py``): per-class average precision -> mAP, and
per-class best-threshold F1 over the PR curve -> overall F1.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .model import MODEL_NAME

LABEL_COLUMNS: List[str] = [
    "Antibacterial", "Antifungal", "Antiviral", "Antiparasitic", "Antimicrobial",
]
NUM_LABELS = len(LABEL_COLUMNS)


def load_escape(path) -> pd.DataFrame:
    """Read a reconstructed ESCAPE CSV (cols: Sequence, Hash, 5 label columns)."""
    return pd.read_csv(path)


class EscapeDataset(Dataset):
    """Tokenised multilabel view over an ESCAPE DataFrame.

    Labels are a length-5 float vector (multi-hot) for ``BCEWithLogitsLoss``.
    """

    def __init__(self, df, tokenizer_name: str = MODEL_NAME, max_len: int = 200):
        self.df = df.reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False, use_fast=False)
        self.max_len = max_len
        self.seqs = list(self.df["Sequence"])
        self.labels = self.df[LABEL_COLUMNS].values.astype("float32")

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int):
        seq = " ".join("".join(str(self.seqs[idx]).split()))  # ProtBERT wants spaced residues
        ids = self.tokenizer(seq, truncation=True, padding="max_length", max_length=self.max_len)
        sample = {k: torch.tensor(v) for k, v in ids.items()}
        sample["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return sample


def escape_scores(y_true, y_probs):
    """ESCAPE official metrics: returns (per-class AP, per-class F1, mAP, macro-F1)."""
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    aps, f1s = [], []
    for i in range(y_true.shape[1]):
        aps.append(average_precision_score(y_true[:, i], y_probs[:, i]))
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
        f1s.append(np.max(2 * precision * recall / (precision + recall + 1e-8)))
    return aps, f1s, float(np.mean(aps)), float(np.mean(f1s))


def escape_compute_metrics(pred):
    """``Trainer``-compatible callback returning {'mAP', 'macro_f1'}."""
    y_true = pred.label_ids
    y_probs = 1.0 / (1.0 + np.exp(-np.asarray(pred.predictions)))  # sigmoid
    _, _, mAP, macro_f1 = escape_scores(y_true, y_probs)
    return {"mAP": mAP, "macro_f1": macro_f1}


def escape_model_init():
    """Zero-arg factory: ProtBERT-BFD with a 5-way multilabel head."""
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, problem_type="multi_label_classification"
    )
