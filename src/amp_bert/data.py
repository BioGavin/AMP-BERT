"""Dataset utilities for AMP-BERT.

Each CSV is expected to have the columns produced by the AMP-BERT pipeline::

    <index>, aa_seq, aa_len, AMP

``aa_seq`` is the raw amino-acid string and ``AMP`` is a boolean label.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

DEFAULT_TOKENIZER = "Rostlab/prot_bert_bfd"


def load_dataset(path, shuffle: bool = False, random_state: int = 0) -> pd.DataFrame:
    """Read an AMP CSV into a DataFrame, optionally shuffling the rows."""
    df = pd.read_csv(path, index_col=0)
    if shuffle:
        df = df.sample(frac=1, random_state=random_state)
    return df


class AmpDataset(Dataset):
    """Tokenised view over an AMP DataFrame for the HuggingFace ``Trainer``.

    Fixes the original notebook bug where ``get_seqs_labels`` read a module-level
    global ``df`` instead of the instance's own frame.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer_name: str = DEFAULT_TOKENIZER,
        max_len: int = 200,
    ):
        self.df = df.reset_index(drop=True)
        # use_fast=False loads the plain WordPiece BertTokenizer directly from
        # vocab.txt. The fast tokenizer path on recent transformers tries to
        # convert via sentencepiece (not installed on Colab) and fails for this
        # older model, so we avoid it.
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, do_lower_case=False, use_fast=False
        )
        self.max_len = max_len
        self.seqs, self.labels = self._get_seqs_labels()

    def _get_seqs_labels(self) -> Tuple[List[str], List[int]]:
        seqs = list(self.df["aa_seq"])
        labels = list(self.df["AMP"].astype(int))
        assert len(seqs) == len(labels)
        return seqs, labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # ProtBERT expects space-separated residues; collapse any existing
        # whitespace first, then re-join character by character.
        seq = " ".join("".join(self.seqs[idx].split()))
        seq_ids = self.tokenizer(
            seq,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
        )
        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample["labels"] = torch.tensor(self.labels[idx])
        return sample
