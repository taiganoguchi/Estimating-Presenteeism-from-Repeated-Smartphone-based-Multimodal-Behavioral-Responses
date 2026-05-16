"""Stage 4: build CV / holdout splits saved alongside seq_index.

Ported from pipeline.ipynb cells 27-28, 31, 33.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, List, Tuple
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold, KFold, StratifiedKFold, LeaveOneGroupOut,
    train_test_split, GroupShuffleSplit, StratifiedShuffleSplit,
)

from .utils import json_dump


@dataclass
class SplitPlan:
    mode: str
    n_splits: int
    seed: int
    test_size: float
    group_col: str
    stratify_col: str
    save_to: Path


def build_split_plan(cfg: dict, seq_dir: Path) -> SplitPlan:
    sp = cfg.get("split", {})
    save_rel = sp.get("save_to", "split.json")
    save_abs = seq_dir / save_rel
    return SplitPlan(
        mode=str(sp.get("mode", "group_kfold")),
        n_splits=int(sp.get("n_splits", 5)),
        seed=int(sp.get("seed", 42)),
        test_size=float(sp.get("test_size", 0.2)),
        group_col=str(sp.get("group_col", "user_id")),
        stratify_col=str(sp.get("stratify_col", "label")),
        save_to=save_abs,
    )


def iter_cv_indices(df: pd.DataFrame, plan: SplitPlan) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(plan.seed)
    y = df[plan.stratify_col].values if plan.stratify_col in df.columns else None
    g = df[plan.group_col].values if plan.group_col in df.columns else None
    n = len(df)
    mode = plan.mode.lower()
    if mode == "group_kfold":
        for tr, va in GroupKFold(n_splits=plan.n_splits).split(np.arange(n), y, groups=g):
            yield tr, va
    elif mode == "kfold":
        for tr, va in KFold(n_splits=plan.n_splits, shuffle=True, random_state=plan.seed).split(np.arange(n)):
            yield tr, va
    elif mode == "stratified_kfold":
        assert y is not None
        for tr, va in StratifiedKFold(n_splits=plan.n_splits, shuffle=True, random_state=plan.seed).split(np.arange(n), y):
            yield tr, va
    elif mode == "leave_one_group_out":
        for tr, va in LeaveOneGroupOut().split(np.arange(n), y, groups=g):
            yield tr, va
    elif mode == "holdout":
        if g is None:
            tr, va = train_test_split(
                np.arange(n), test_size=plan.test_size, random_state=plan.seed, stratify=y
            )
        else:
            uniq = np.unique(g); rng.shuffle(uniq)
            cut = int(len(uniq) * plan.test_size)
            te_groups = set(uniq[:cut])
            va = np.where(np.isin(g, list(te_groups)))[0]
            tr = np.setdiff1d(np.arange(n), va)
        yield tr, va
    else:
        raise ValueError(f"Unknown split.mode: {plan.mode}")


def save_split_indices(plan: SplitPlan, folds: List[Tuple[np.ndarray, np.ndarray]], n_samples: int) -> None:
    meta = {
        "mode": plan.mode,
        "n_splits": plan.n_splits,
        "seed": plan.seed,
        "group_col": plan.group_col,
        "stratify_col": plan.stratify_col,
        "folds": [{"train_idx": tr.tolist(), "val_idx": va.tolist()} for tr, va in folds],
        "n_samples": n_samples,
    }
    plan.save_to.parent.mkdir(parents=True, exist_ok=True)
    json_dump(meta, plan.save_to)
    print(f"[splits] saved -> {plan.save_to}")


def make_holdout_split(idx_df: pd.DataFrame, seq_dir: Path, seed: int = 42, test_size: float = 0.2) -> Path:
    """User-grouped holdout split with stratified fallback (cell 33)."""
    groups = idx_df["user_id"].astype(str)
    y = idx_df["label"].astype(int)
    gs = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gs.split(idx_df, y, groups))
    if set(np.unique(y.iloc[test_idx])) != set(np.unique(y)):
        print("[split] fallback to StratifiedShuffleSplit")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(sss.split(np.zeros(len(y)), y))
    split = {
        "seed": seed,
        "test_size": test_size,
        "train_idx": sorted(int(i) for i in train_idx),
        "test_idx": sorted(int(i) for i in test_idx),
        "label_dist_all": idx_df["label"].value_counts().sort_index().to_dict(),
        "label_dist_train": idx_df.iloc[train_idx]["label"].value_counts().sort_index().to_dict(),
        "label_dist_test": idx_df.iloc[test_idx]["label"].value_counts().sort_index().to_dict(),
    }
    path = seq_dir / "split.holdout.json"
    json_dump(split, path)
    print(f"[splits] holdout -> {path}")
    return path


def run_splits(cfg: dict) -> dict:
    root = Path(cfg["paths"]["root"])
    seq_dir = root / cfg["paths"]["outputs_dir"] / "sequences"
    idx_path = seq_dir / "seq_index.parquet"
    assert idx_path.exists(), f"seq_index.parquet not found: {idx_path}"
    seq_df = pd.read_parquet(idx_path).copy()

    plan = build_split_plan(cfg, seq_dir)
    folds = []
    for tr_idx, va_idx in iter_cv_indices(seq_df, plan):
        assert len(set(tr_idx).intersection(va_idx)) == 0
        folds.append((tr_idx, va_idx))
    save_split_indices(plan, folds, len(seq_df))

    holdout_path = make_holdout_split(
        seq_df, seq_dir,
        seed=int(cfg.get("runtime", {}).get("seed", 42)),
        test_size=float(cfg.get("split", {}).get("test_size", 0.2)),
    )
    return {"status": "ok", "split": str(plan.save_to), "holdout": str(holdout_path)}
