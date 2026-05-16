"""Per-user × cohort blended window-level normalization (cell 25)."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def blended_group_norm_windows(
    X_list: list[np.ndarray],
    user_list: list[str],
    cohort_list: list[str],
    eps: float = 1e-8,
    freeze_col_indices: list[int] | None = None,
):
    D = X_list[0].shape[1]
    freeze_mask = np.zeros(D, dtype=bool)
    if freeze_col_indices is not None:
        freeze_mask[[int(i) for i in freeze_col_indices if 0 <= int(i) < D]] = True

    rows = []
    for i, X in enumerate(X_list):
        u = user_list[i]; c = cohort_list[i]; t = X.shape[0]
        rows.append(pd.DataFrame({
            "user_id": [u] * t,
            "cohort": [c] * t,
            **{f"f{j}": X[:, j] for j in range(D)},
        }))
    big = pd.concat(rows, axis=0, ignore_index=True)

    for j in range(D):
        col = f"f{j}"
        med = big[col].median(skipna=True)
        big[col] = big[col].fillna(med)

    gvar = big[[f"f{j}" for j in range(D)]].var(axis=0, ddof=0).replace(0, np.nan).fillna(1.0)
    tau = 5.0 * np.sqrt(gvar)
    mu_u = big.groupby("user_id").mean(numeric_only=True)
    var_u = big.groupby("user_id").var(ddof=0, numeric_only=True).fillna(0.0)
    mu_c = big.groupby("cohort").mean(numeric_only=True)
    var_c = big.groupby("cohort").var(ddof=0, numeric_only=True).fillna(0.0)
    n_user = big["user_id"].value_counts()

    Xn_list = []
    st = 0
    for i, X in enumerate(X_list):
        u = user_list[i]; c = cohort_list[i]; t = X.shape[0]
        blk = big.iloc[st: st + t].copy()
        st += t
        blk_orig = blk[[f"f{j}" for j in range(D)]].copy()
        for j in range(D):
            col = f"f{j}"
            if freeze_mask[j]:
                blk[col] = blk_orig[col].values
                continue
            nu = float(n_user.get(u, 1.0))
            w = nu / (nu + float(tau[col]))
            mu_bl = w * float(mu_u.loc[u, col]) + (1 - w) * float(mu_c.loc[c, col])
            var_bl = w * float(var_u.loc[u, col]) + (1 - w) * float(var_c.loc[c, col])
            sd_bl = math.sqrt(max(var_bl, 0.0))
            blk[col] = (blk[col].values - mu_bl) / (sd_bl + eps)
        Xn = blk[[f"f{j}" for j in range(D)]].to_numpy(dtype=np.float32)
        Xn_list.append(Xn)

    norm_meta = {
        "tau": {f"f{j}": float(tau[f"f{j}"]) for j in range(D)},
        "eps": float(eps),
        "D": int(D),
        "freeze_col_indices": [int(i) for i in np.where(freeze_mask)[0].tolist()],
    }
    return Xn_list, norm_meta
