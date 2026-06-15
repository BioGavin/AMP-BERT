# Datasets

All CSVs share the schema produced by the AMP-BERT pipeline:

| column   | type   | description                          |
|----------|--------|--------------------------------------|
| (index)  | str    | accession id (AP…, DRAMP…, UniRef…)  |
| `aa_seq` | str    | amino-acid sequence                  |
| `aa_len` | int    | sequence length                      |
| `AMP`    | bool   | label: `True` = AMP, `False` = non-AMP |

## `raw/`

| file | role | rows | label |
|------|------|------|-------|
| `all_veltri.csv` | AMP-BERT **training** set (Veltri) | 3556 | mixed |
| `veltri_dramp_cdhit_90.csv` | external **test** — AMPs (DRAMP, CD-HIT 90%) | 2065 | AMP |
| `non_amp_ampep_cdhit90.csv` | external **test** — non-AMPs (AMPEP, CD-HIT 90%) | 1908 | non-AMP |

The external test set used in the paper is the **concatenation** of the two test
files above (AMP + non-AMP). Notebook `02_test_reproduce.ipynb` does this merge.

## ESCAPE benchmark (multilabel)

The ESCAPE dataset is **not stored here** — `notebooks/02_escape_benchmark.ipynb`
auto-downloads it from Harvard Dataverse ([doi:10.7910/DVN/C69MCD](https://doi.org/10.7910/DVN/C69MCD)):
`Fold1` + `Fold2` (training folds) and `Test`. Its schema differs from the binary
files above — `Sequence, Hash, Antibacterial, Antifungal, Antiviral, Antiparasitic,
Antimicrobial` — where the five label columns are 0/1 multi-hot (non-AMP = all zeros).
AMP-BERT uses `Sequence` + the five labels only (the `Hash` structural maps are for
the ESCAPE Baseline, not needed here).
