"""AMP-BERT: fine-tuned ProtBERT-BFD for antimicrobial peptide (AMP) classification.

Public API re-exported for convenience::

    from amp_bert import AmpDataset, load_dataset, compute_metrics, model_init
"""

from .config import PROJECT_ROOT, DATA_DIR, RAW_DIR, MODELS_DIR, RESULTS_DIR
from .data import AmpDataset, load_dataset
from .metrics import compute_metrics
from .model import model_init, build_training_args, MODEL_NAME

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "MODELS_DIR",
    "RESULTS_DIR",
    "AmpDataset",
    "load_dataset",
    "compute_metrics",
    "model_init",
    "build_training_args",
    "MODEL_NAME",
]
