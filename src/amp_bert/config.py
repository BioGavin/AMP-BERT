"""Centralised, machine-independent paths.

The original notebook hard-coded ``/home/hansol/amp/...``. Here every path is
derived from the repository root so the code runs unchanged locally, on a
server, or in Colab (after cloning the repo).
"""

from pathlib import Path

# .../AMP-BERT/src/amp_bert/config.py -> repo root is three parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Convenience handles for the datasets shipped with the repo.
TRAIN_CSV = RAW_DIR / "all_veltri.csv"               # AMP-BERT training set (Veltri)
TEST_NONAMP_CSV = RAW_DIR / "non_amp_ampep_cdhit90.csv"   # external non-AMP test
TEST_AMP_CSV = RAW_DIR / "veltri_dramp_cdhit_90.csv"      # external AMP test (DRAMP)

# ESCAPE benchmark (multilabel), reconstructed CSVs.
ESCAPE_DIR = DATA_DIR / "escape"
ESCAPE_FOLD1_CSV = ESCAPE_DIR / "Fold1_reconstructed.csv"
ESCAPE_FOLD2_CSV = ESCAPE_DIR / "Fold2_reconstructed.csv"
ESCAPE_TEST_CSV = ESCAPE_DIR / "Test_reconstructed.csv"

for _d in (MODELS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
