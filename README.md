# AMP-BERT: Prediction of Antimicrobial Peptide Function Based on a BERT Model

AMP-BERT is a deep-learning classifier that labels a peptide sequence as **AMP** (antimicrobial) or **non-AMP**. It fine-tunes the protein language model **[ProtBERT-BFD](https://huggingface.co/Rostlab/prot_bert_bfd)** ([Elnaggar et al., 2021](https://doi.org/10.1109/TPAMI.2021.3095381)) on curated AMP datasets.

This repository is organised around three things you can reproduce end-to-end, all runnable on **Google Colab**:

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
├── src/amp_bert/          # reusable package (notebooks stay thin)
│   ├── config.py          # repo-relative paths (no hard-coded /home/...)
│   ├── data.py            # AmpDataset + load_dataset
│   ├── metrics.py         # accuracy / F1 / precision / recall / MCC / AUC
│   └── model.py           # model_init + build_training_args
├── notebooks/
│   ├── 01_train_amp_bert.ipynb   # Part 2 — train
│   ├── 02_test_reproduce.ipynb   # Part 2 — test
│   ├── 03_escape_train.ipynb     # Part 3 — train
│   ├── 04_escape_test.ipynb      # Part 3 — test
│   └── _legacy_fine-tune_with_amps.ipynb  # original notebook, kept for reference
├── data/raw/              # datasets (see data/README.md)
├── config/default.yaml    # experiment hyper-parameters
├── models/  results/      # checkpoints & outputs (git-ignored)
└── requirements.txt  pyproject.toml
```

### Setup
**Colab (recommended):** the notebooks are **self-contained** — open one via its
Colab badge below and run top to bottom. The first cell installs a pinned
`transformers`, all logic is inlined, every tunable parameter sits in a **Config**
cell, and datasets are read straight from this repo's raw URLs. No cloning, no
`git pull`, no restart cycle.

**Local / scripting:** the same logic is packaged under `src/amp_bert/` for `import`
and reuse:
```bash
pip install -r requirements.txt
pip install -e .          # makes `import amp_bert` available
```

### Datasets
| file | role | label |
|------|------|-------|
| `data/raw/all_veltri.csv` | training set (Veltri) | mixed |
| `data/raw/veltri_dramp_cdhit_90.csv` | external test — AMPs (DRAMP) | AMP |
| `data/raw/non_amp_ampep_cdhit90.csv` | external test — non-AMPs (AMPEP) | non-AMP |

All CSVs share the schema `aa_seq, aa_len, AMP`. See [data/README.md](data/README.md) for details.

---

## Part 2 · Reproduce the original paper (train + test)

Reproduce AMP-BERT exactly as published: fine-tune on the Veltri training set, then evaluate on the external test set (DRAMP AMPs ∪ AMPEP non-AMPs).

| step | notebook | Colab |
|------|----------|-------|
| **Train** | [`notebooks/01_train_amp_bert.ipynb`](notebooks/01_train_amp_bert.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BioGavin/AMP-BERT/blob/main/notebooks/01_train_amp_bert.ipynb) |
| **Test**  | [`notebooks/02_test_reproduce.ipynb`](notebooks/02_test_reproduce.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BioGavin/AMP-BERT/blob/main/notebooks/02_test_reproduce.ipynb) |

**What happens**
1. `01` fine-tunes ProtBERT-BFD (15 epochs, effective batch size 64) and saves the model to `models/amp_bert/`.
2. `02` loads that model and reports accuracy, F1, precision, recall, MCC and ROC-AUC on the merged external test set, writing per-sequence predictions to `results/external_test_predictions.csv`.

---

## Part 3 · ESCAPE benchmark (train + test)

Train and evaluate AMP-BERT on the **ESCAPE benchmark** using the same pipeline.

| step | notebook | Colab |
|------|----------|-------|
| **Train** | [`notebooks/03_escape_train.ipynb`](notebooks/03_escape_train.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BioGavin/AMP-BERT/blob/main/notebooks/03_escape_train.ipynb) |
| **Test**  | [`notebooks/04_escape_test.ipynb`](notebooks/04_escape_test.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BioGavin/AMP-BERT/blob/main/notebooks/04_escape_test.ipynb) |

**Setup:** place the ESCAPE splits (`escape_train.csv`, `escape_test.csv`, schema `aa_seq, aa_len, AMP`) under `data/raw/escape/`, then run `03` → `04`. `03` saves the model to `models/amp_bert_escape/`; `04` evaluates it and writes
`results/escape_test_predictions.csv`.

