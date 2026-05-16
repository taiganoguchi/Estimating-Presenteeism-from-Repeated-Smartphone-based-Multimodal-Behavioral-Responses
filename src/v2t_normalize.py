"""Fold-time normalization for v2-temporal (v3) features.

Fits blended per-user/cohort shrinkage z-score on TRAIN clips only,
then applies the same transformation to ALL clips in the fold.
Test-fold users (absent from train) fall back to cohort statistics (w=0).

Usage in worker:
    normed_seqs, state = normalize_fold(
        all_seqs, all_users, all_cohorts,
        train_idx=fold_train_idx,
        freeze_col_indices=list(range(1792, 2560)),  # freeze SBERT columns
    )
    df_tr = idx_all.iloc[fold_train_idx]  # still uses original parquet index
    # seqs are now index-aligned: normed_seqs[i] corresponds to all_seqs[i]
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def fit_norm_on_train(
    seqs: list[np.ndarray],
    users: list[str],
    cohorts: list[str],
    freeze_col_indices: list[int] | None = None,
    eps: float = 1e-8,
) -> dict:
    """Compute normalization statistics from a list of training clips.

    Parameters
    ----------
    seqs : list of (T_i, D) arrays — raw feature arrays for each clip
    users : list of user_id strings, len == len(seqs)
    cohorts : list of cohort strings, len == len(seqs)
    freeze_col_indices : column indices to skip normalization (e.g. SBERT)
    eps : numerical stability constant

    Returns
    -------
    state dict with keys: D, eps, tau, mu_u, var_u, mu_c, var_c,
                          n_user, freeze_mask, col_medians
    """
    D = seqs[0].shape[1]
    freeze_mask = np.zeros(D, dtype=bool)
    if freeze_col_indices:
        for i in freeze_col_indices:
            if 0 <= i < D:
                freeze_mask[i] = True

    # Build big DataFrame from all train clips
    rows = []
    for i, X in enumerate(seqs):
        u, c, t = users[i], cohorts[i], X.shape[0]
        rows.append(pd.DataFrame(
            {"user_id": [u] * t, "cohort": [c] * t,
             **{f"f{j}": X[:, j] for j in range(D)}}
        ))
    big = pd.concat(rows, axis=0, ignore_index=True)

    # Fill NaN with per-column median before computing statistics
    col_medians: dict[str, float] = {}
    for j in range(D):
        col = f"f{j}"
        med = float(big[col].median(skipna=True))
        if np.isnan(med):
            med = 0.0
        col_medians[col] = med
        big[col] = big[col].fillna(med)

    feat_cols = [f"f{j}" for j in range(D)]
    gvar = big[feat_cols].var(axis=0, ddof=0).replace(0, np.nan).fillna(1.0)
    tau = 5.0 * np.sqrt(gvar)

    mu_u = big.groupby("user_id").mean(numeric_only=True)
    var_u = big.groupby("user_id").var(ddof=0, numeric_only=True).fillna(0.0)
    mu_c = big.groupby("cohort").mean(numeric_only=True)
    var_c = big.groupby("cohort").var(ddof=0, numeric_only=True).fillna(0.0)
    n_user = big["user_id"].value_counts()

    return {
        "D": int(D),
        "eps": float(eps),
        "tau": tau,
        "mu_u": mu_u,
        "var_u": var_u,
        "mu_c": mu_c,
        "var_c": var_c,
        "n_user": n_user,
        "freeze_mask": freeze_mask,
        "col_medians": col_medians,
    }


def apply_norm(
    seq: np.ndarray,
    user: str,
    cohort: str,
    state: dict,
) -> np.ndarray:
    """Apply fitted normalization state to a single clip.

    For test-fold users not present in training data, w=0 so only cohort
    statistics are used (graceful fallback).
    """
    D = state["D"]
    tau = state["tau"]
    mu_u, var_u = state["mu_u"], state["var_u"]
    mu_c, var_c = state["mu_c"], state["var_c"]
    n_user = state["n_user"]
    freeze_mask = state["freeze_mask"]
    eps = state["eps"]
    medians = state["col_medians"]

    X = seq.copy().astype(np.float32)

    # Fill NaN with training-set medians (M1 fix: meaningful fill value)
    for j in range(D):
        col = f"f{j}"
        nan_mask = np.isnan(X[:, j])
        if nan_mask.any():
            X[nan_mask, j] = medians[col]

    Xn = X.copy()
    for j in range(D):
        if freeze_mask[j]:
            continue
        col = f"f{j}"
        tau_j = float(tau[col])

        # Shrinkage weight: zero for unseen users → pure cohort prior
        nu = float(n_user.get(user, 0.0))
        w = nu / (nu + tau_j)

        if user in mu_u.index:
            mu_u_j = float(mu_u.loc[user, col])
            var_u_j = float(var_u.loc[user, col])
        else:
            w = 0.0
            mu_u_j = 0.0
            var_u_j = 0.0

        if cohort in mu_c.index:
            mu_c_j = float(mu_c.loc[cohort, col])
            var_c_j = float(var_c.loc[cohort, col])
        else:
            mu_c_j = 0.0
            var_c_j = 1.0

        mu_bl = w * mu_u_j + (1.0 - w) * mu_c_j
        var_bl = w * var_u_j + (1.0 - w) * var_c_j
        sd_bl = math.sqrt(max(var_bl, 0.0))
        Xn[:, j] = (X[:, j] - mu_bl) / (sd_bl + eps)

    return Xn


def normalize_fold(
    all_seqs: list[np.ndarray],
    all_users: list[str],
    all_cohorts: list[str],
    train_idx: np.ndarray,
    freeze_col_indices: list[int] | None = None,
    nan_fill: bool = True,
) -> tuple[list[np.ndarray], dict]:
    """Fit normalization on train_idx clips, apply to ALL clips.

    Parameters
    ----------
    all_seqs : raw feature arrays for every clip (full dataset)
    all_users : user_id for each clip
    all_cohorts : cohort string for each clip
    train_idx : indices (into all_seqs) of the training fold
    freeze_col_indices : columns to skip (e.g. SBERT text [1792:2560])
    nan_fill : if True, replace any remaining NaN with 0 after normalization

    Returns
    -------
    normed_seqs : list of normalized arrays (same length as all_seqs)
    state : fitted normalization state dict
    """
    train_seqs = [all_seqs[i] for i in train_idx]
    train_users = [all_users[i] for i in train_idx]
    train_cohorts = [all_cohorts[i] for i in train_idx]

    state = fit_norm_on_train(
        train_seqs, train_users, train_cohorts, freeze_col_indices
    )

    normed: list[np.ndarray] = []
    for i in range(len(all_seqs)):
        xn = apply_norm(all_seqs[i], all_users[i], all_cohorts[i], state)
        if nan_fill:
            xn = np.nan_to_num(xn, nan=0.0, posinf=0.0, neginf=0.0)
        normed.append(xn)

    return normed, state
