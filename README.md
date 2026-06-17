# AMP-BERT: Prediction of Antimicrobial Peptide Function Based on a BERT Model

AMP-BERT is a deep-learning classifier that labels a peptide sequence as **AMP** (antimicrobial) or **non-AMP**. It fine-tunes the protein language model **[ProtBERT-BFD](https://huggingface.co/Rostlab/prot_bert_bfd)** ([Elnaggar et al., 2021](https://doi.org/10.1109/TPAMI.2021.3095381)) on curated AMP datasets.

This repository is organised around three things you can reproduce end-to-end, runnable on **Google Colab** or headless on a **GPU server** via the CLI scripts in [`scripts/`](scripts):

1. [Part 1 · About AMP-BERT](#part-1--about-amp-bert)
2. [Part 2 · Reproduce the original paper (train + test)](#part-2--reproduce-the-original-paper-train--test)
3. [Part 3 · ESCAPE benchmark (train + test)](#part-3--escape-benchmark-train--test)

---

## Part 1 · About AMP-BERT

### Abstract
Antimicrobial resistance is a growing health concern. Antimicrobial peptides (AMPs) disrupt harmful microorganisms by non-specific mechanisms, making it difficult for microbes to develop resistance. Accordingly, they are promising alternatives to traditional antimicrobial drugs. In this study, we developed an improved AMP classification model, called AMP-BERT. We propose a deep learning model with a fine-tuned BERT architecture designed to extract structural/functional information from input peptides and identify each input as AMP or non-AMP. We compared the performance of our proposed model and other machine learning-based and deep learning-based methods. Our model, AMP-BERT, yielded the best prediction results among all models evaluated with our curated external dataset. In addition, we utilized the attention mechanism in BERT to implement an interpretable feature analysis and determine the specific residues in known AMPs that contribute to peptide structure and antimicrobial function. The results show that AMP-BERT can capture the structural properties of peptides for model learning, enabling the prediction of AMPs or non-AMPs from input sequences. AMP-BERT is expected to contribute to the identification of candidate AMPs for functional validation and drug development.

### Model overview
![OverviewFigure](assets/Fig1_Overview_final.png)

### Repository layout
```
AMP-BERT/
├── src/amp_bert/          # reusable package (notebooks & scripts stay thin)
│   ├── config.py          # repo-relative dataset / model paths
│   ├── data.py            # AmpDataset + load_dataset (binary)
│   ├── metrics.py         # SN / SP / F1 / ACC / AUROC / AUPR + MCC (binary)
│   ├── model.py           # model_init + build_training_args
│   └── escape.py          # ESCAPE multilabel: EscapeDataset, escape_scores, model_init
├── notebooks/             # Colab, self-contained
│   ├── 01_reproduce_amp_bert.ipynb  # Part 2 — train + test
│   ├── 02_escape_benchmark.ipynb    # Part 3 — train + test
│   └── _legacy_fine-tune_with_amps.ipynb  # original notebook, kept for reference
├── scripts/               # server / CLI train + test (see scripts/README.md)
│   ├── train_amp_bert.py  test_amp_bert.py  run_amp_bert.sh   # Part 2
│   └── train_escape.py    test_escape.py    run_escape.sh     # Part 3
├── data/raw/  data/escape/   # datasets (see data/README.md)
├── config/default.yaml    # experiment hyper-parameters
├── models/  results/      # checkpoints & outputs (git-ignored)
└── requirements.txt  pyproject.toml
```

### Setup
**Colab (recommended):** the notebooks are **self-contained** — open one via its
Colab badge below and run top to bottom. The first cell installs a pinned
`transformers`, the logic is inlined step by step, parameters sit next to the
step that uses them, and datasets are read straight from this repo's raw URLs.
Each notebook covers **both train and test**, saving/loading the model via Google
Drive so you can retrain once and re-test in any later session. No cloning, no
`git pull`, no restart cycle.

**Server / CLI:** for headless runs on a GPU server, [`scripts/`](scripts) provides
command-line train/test scripts that reuse the same `src/amp_bert/` package (so
behaviour matches the notebooks). They log to file, print the device, and take
hyper-parameters as flags. Install the deps first:
```bash
pip install -r requirements.txt
pip install -e .          # optional; scripts also add src/ to sys.path
```
Then see [scripts/README.md](scripts/README.md), or the per-part commands below.

### Datasets
| file | role | label |
|------|------|-------|
| `data/raw/all_veltri.csv` | training set (Veltri) | mixed |
| `data/raw/veltri_dramp_cdhit_90.csv` | external test — AMPs (DRAMP) | AMP |
| `data/raw/non_amp_ampep_cdhit90.csv` | external test — non-AMPs (AMPEP) | non-AMP |

All CSVs share the schema `aa_seq, aa_len, AMP`. See [data/README.md](data/README.md) for details.

---

## Part 2 · Reproduce the original paper (train + test)

Reproduce AMP-BERT exactly as published: fine-tune on the Veltri training set, then evaluate on the external test set (DRAMP AMPs ∪ AMPEP non-AMPs). Part A trains and saves AMP-BERT to Drive; Part B loads it and evaluates.

| notebook | Colab |
|----------|-------|
| [`notebooks/01_reproduce_amp_bert.ipynb`](notebooks/01_reproduce_amp_bert.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BioGavin/AMP-BERT/blob/main/notebooks/01_reproduce_amp_bert.ipynb) |

Reports SN / SP / F1 / ACC / AUROC / AUPR (plus precision, MCC) on the merged external test set, writing per-sequence predictions to `results/external_test_predictions.csv`.

**On a server (no Colab):**
```bash
python3 scripts/train_amp_bert.py --epochs 15      # -> models/amp_bert
python3 scripts/test_amp_bert.py                   # metrics + predictions
# or both at once:
bash scripts/run_amp_bert.sh
```
See [scripts/README.md](scripts/README.md) for all flags (`--lr`, `--batch-size`, `--grad-accum`, `--val-frac`, `--log-file`, …).

---

## Part 3 · ESCAPE benchmark (train + test)

Reproduce **AMP-BERT on the [ESCAPE benchmark](https://github.com/BCV-Uniandes/ESCAPE)** (Ojeda et al., NeurIPS 2025) — a **multilabel** task with 5 binary labels (Antibacterial, Antifungal, Antiviral, Antiparasitic, Antimicrobial). The notebook adapts AMP-BERT with a 5-way multilabel head (`BCEWithLogitsLoss`), trains on Fold1 + Fold2 (seed 0, single model — no multi-seed/ensemble), and evaluates on the Test split. Metrics replicate ESCAPE's official `compute_metrics`: per-class AP → **mAP**, and per-class best-threshold F1 → overall **F1** (paper reports AMP-BERT: F1 64.7 / mAP 66.9).

**Data:** the Harvard Dataverse release blanks the sequences from three license-restricted sources (DFBP, dbAMP 3.0, DBAASP), keeping only their `Hash`. The reconstructed full CSVs (sequences refilled by hash) are committed under [`data/escape/`](data/escape) and the notebook reads them directly — no download or reconstruction step needed.

| notebook | Colab |
|----------|-------|
| [`notebooks/02_escape_benchmark.ipynb`](notebooks/02_escape_benchmark.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BioGavin/AMP-BERT/blob/main/notebooks/02_escape_benchmark.ipynb) |

The Colab notebook trains a single model on Fold1 + Fold2 combined. The **server scripts** follow ESCAPE's **two-fold cross-validation**: one model per fold, then ensembled on the Test split (hyper-parameters aligned with the paper's Table 5).

**On a server (no Colab):**
```bash
python3 scripts/train_escape.py        # trains models/amp_bert_escape/{fold1,fold2}
python3 scripts/test_escape.py         # ensembles both, reports per-class AP/F1 + mAP/F1
# or both at once:
bash scripts/run_escape.sh
```
Use `--only-fold 1 --epochs 1` for a quick pipeline check first. See [scripts/README.md](scripts/README.md).

