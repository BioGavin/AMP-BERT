"""AMP-BERT: fine-tuned ProtBERT-BFD for antimicrobial peptide (AMP) classification.

Public API re-exported for convenience::

    from amp_bert import AmpDataset, load_dataset, compute_metrics, model_init
"""

from .config import PROJECT_ROOT, DATA_DIR, RAW_DIR, MODELS_DIR, RESULTS_DIR
from .data import AmpDataset, load_dataset
from .metrics import binary_metrics, compute_metrics
from .model import model_init, build_model, build_training_args, MODEL_NAME
from .escape import (
    EscapeDataset,
    escape_compute_metrics,
    escape_model_init,
    escape_scores,
    load_escape,
    LABEL_COLUMNS,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "MODELS_DIR",
    "RESULTS_DIR",
    "AmpDataset",
    "load_dataset",
    "binary_metrics",
    "compute_metrics",
    "model_init",
    "build_model",
    "build_training_args",
    "MODEL_NAME",
    # ESCAPE (multilabel)
    "EscapeDataset",
    "escape_scores",
    "escape_compute_metrics",
    "escape_model_init",
    "load_escape",
    "LABEL_COLUMNS",
]
