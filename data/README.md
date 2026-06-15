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

## `escape/` (multilabel)

The ESCAPE benchmark, **reconstructed** to fill in the license-restricted
sequences that Harvard Dataverse ([doi:10.7910/DVN/C69MCD](https://doi.org/10.7910/DVN/C69MCD))
ships blanked (sourced from DFBP, dbAMP 3.0, DBAASP; ≈3.5k of the ~16.5k test rows).
`notebooks/02_escape_benchmark.ipynb` reads these directly.

| file | role | rows |
|------|------|------|
| `escape/Fold1_reconstructed.csv` | training fold 1 | 32948 |
| `escape/Fold2_reconstructed.csv` | training fold 2 | 32922 |
| `escape/Test_reconstructed.csv`  | independent test | 16489 |

Schema: `Sequence, Hash, Antibacterial, Antifungal, Antiviral, Antiparasitic,
Antimicrobial` — the five label columns are 0/1 multi-hot (non-AMP = all zeros);
`Hash` = `sha256(sequence)[:16]`. AMP-BERT uses `Sequence` + the five labels only.

> The ESCAPE dataset is CC BY 4.0, but the three sources above carry their own
> upstream licenses — these reconstructed files are kept here for reproduction
> convenience; redistribute with that in mind.
