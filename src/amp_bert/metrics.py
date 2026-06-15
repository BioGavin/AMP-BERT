"""Evaluation metrics for binary AMP / non-AMP classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def binary_metrics(labels, logits) -> dict:
    """Full binary-classification metric set from labels and class logits.

    Returns sensitivity (SN, = recall), specificity (SP), F1, accuracy (ACC),
    precision, MCC, AUROC and AUPR. AUROC/AUPR use the positive-class
    probability and need both classes present.
    """
    labels = np.asarray(labels)
    logits = np.asarray(logits)
    preds = logits.argmax(-1)

    # Confusion matrix with fixed label order so it is always 2x2.
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0   # SN / recall
    specificity = tn / (tn + fp) if (tn + fp) else 0.0   # SP

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(labels, preds),   # ACC
        "sensitivity": sensitivity,                  # SN
        "specificity": specificity,                  # SP
        "precision": precision,
        "f1": f1,                                    # F1
        "mcc": matthews_corrcoef(labels, preds),
    }
    if len(np.unique(labels)) == 2:
        probs = _softmax(logits)[:, 1]
        metrics["auroc"] = roc_auc_score(labels, probs)        # AUROC
        metrics["aupr"] = average_precision_score(labels, probs)  # AUPR
    return metrics


def compute_metrics(pred):
    """HuggingFace ``Trainer``-compatible metric callback (wraps binary_metrics)."""
    return binary_metrics(pred.label_ids, pred.predictions)
