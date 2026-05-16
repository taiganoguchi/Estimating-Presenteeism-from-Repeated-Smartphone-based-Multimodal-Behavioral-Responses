"""Fold-aware PCA compression for SBERT text features in v1 pipeline."""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA


def fit_sbert_pca(
    all_seqs: list[np.ndarray],
    train_idx: list[int],
    text_slice: tuple[int, int] = (44, 812),
    k: int = 64,
    seed: int = 42,
) -> PCA:
    """Fit PCA on SBERT frames from training clips only (no data leak)."""
    s, e = text_slice
    Xs = np.concatenate([all_seqs[i][:, s:e] for i in train_idx], axis=0)
    pca = PCA(n_components=min(k, Xs.shape[1]), random_state=seed, svd_solver="randomized")
    pca.fit(Xs)
    return pca


def apply_sbert_pca(
    seq: np.ndarray,
    pca: PCA,
    text_slice: tuple[int, int] = (44, 812),
) -> np.ndarray:
    """(T, 815) -> (T, 44+k+3) by replacing SBERT block with PCA-compressed version."""
    s, e = text_slice
    pre, txt, post = seq[:, :s], seq[:, s:e], seq[:, e:]
    return np.concatenate([pre, pca.transform(txt).astype(np.float32), post], axis=1)
