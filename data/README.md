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

## `raw/escape/`

Placeholder for the **ESCAPE benchmark**. Drop the train/test CSVs here (same
schema as above) and point `config/default.yaml -> escape.*` at them. Notebooks
`03_escape_train.ipynb` and `04_escape_test.ipynb` consume these.
