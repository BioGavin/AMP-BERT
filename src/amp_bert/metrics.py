"""Evaluation metrics for binary AMP / non-AMP classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def compute_metrics(pred):
    """HuggingFace ``Trainer``-compatible metric callback.

    Returns accuracy, F1, precision, recall, MCC and ROC-AUC. MCC and AUC are
    standard in AMP literature and were missing from the original notebook.
    """
    labels = pred.label_ids
    logits = pred.predictions
    preds = logits.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mcc": matthews_corrcoef(labels, preds),
    }
    # AUC needs the positive-class probability and both classes present.
    if len(np.unique(labels)) == 2:
        probs = _softmax(np.asarray(logits))[:, 1]
        metrics["roc_auc"] = roc_auc_score(labels, probs)
    return metrics
