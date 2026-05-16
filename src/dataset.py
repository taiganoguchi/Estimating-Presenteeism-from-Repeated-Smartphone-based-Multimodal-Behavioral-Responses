"""SequenceDataset + collate function (cell 35)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, index_df: pd.DataFrame):
        self.df = index_df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        npz = np.load(row["seq_path"], allow_pickle=True)
        seq = torch.tensor(npz["seq"], dtype=torch.float32)
        mask = torch.tensor(npz["mask"], dtype=torch.float32)
        if "turn_ids" in npz.files:
            turn_ids = torch.tensor(npz["turn_ids"], dtype=torch.long)
        else:
            turn_ids = torch.zeros(seq.shape[0], dtype=torch.long)
        label = int(npz["label"])
        return seq, mask, label, row["user_id"], turn_ids, i


def make_collate(turn_vocab: int = 128):
    def collate(batch):
        seqs, masks, labels, users, turn_ids, idxs = zip(*batch)
        max_len = max(s.shape[0] for s in seqs)
        D = seqs[0].shape[1]
        padded = torch.zeros(len(seqs), max_len, D)
        padded_mask = torch.zeros(len(seqs), max_len)
        padded_turn = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, (s, m) in enumerate(zip(seqs, masks)):
            T = s.shape[0]
            padded[i, :T] = s
            padded_mask[i, :T] = m
            padded_turn[i, :T] = torch.clamp(turn_ids[i], 0, turn_vocab - 1)
        labels = torch.tensor(labels, dtype=torch.long)
        return padded, padded_mask, labels, list(users), padded_turn, torch.tensor(idxs, dtype=torch.long)

    return collate


def build_state_weights(df: pd.DataFrame, col: str = "sleep_bin", pow_gamma: float = 1.0) -> np.ndarray:
    s = df[col].fillna("UNK")
    vc = s.value_counts()
    w = s.map(lambda k: 1.0 / max(vc.get(k, 1), 1))
    w = w / w.mean()
    return (w.values.astype(np.float32)) ** float(pow_gamma)
