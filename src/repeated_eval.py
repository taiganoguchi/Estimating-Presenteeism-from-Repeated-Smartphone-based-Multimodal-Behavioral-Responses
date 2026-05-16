"""Repeated External CV runner.

Ports notebook cells 31 (CellE++ — repeated split generation) and 43
(CellG++ — Optuna inner CV + outer evaluation per fold) into a single
reusable module.

Heavy: each outer fold runs Optuna inside, so this is GPU-bound. Designed
to be invoked from a notebook or via ``python -m src.repeated_eval``.
"""
from __future__ import annotations
from pathlib import Path
import json, gc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import f1_score

from .dataset import SequenceDataset, make_collate
from .model_define import build_model
from .model_train_eval import EarlyStopper
from .utils import json_dump, seed_everything
from .logging_setup import get_logger

log = get_logger(__name__)


# ---- Split generation (CellE++) ------------------------------------------
def build_splits_repeats(cfg: dict) -> Path:
    root = Path(cfg["paths"]["root"])
    seq_dir = root / cfg["paths"]["outputs_dir"] / "sequences"
    idx = pd.read_parquet(seq_dir / "seq_index.parquet").reset_index(drop=True)
    y = idx["label"].to_numpy()
    groups = idx["user_id"].astype(str).to_numpy()

    outer = cfg.get("split", {}).get("outer", {})
    R = int(outer.get("repeats", 3))
    scheme = outer.get("scheme", "kfold")
    mode = "groupkfold" if scheme == "kfold" else "group_holdout"
    test_size = float(outer.get("holdout_test_size", 0.2))
    K_outer = int(outer.get("k_outer", 10))
    seed0 = int(outer.get("seed_base", cfg.get("runtime", {}).get("seed", 42)))

    repeats = []
    for r in range(R):
        folds: list[dict] = []
        if mode == "group_holdout":
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed0 + r)
            (tr_idx, te_idx), = gss.split(np.zeros(len(y)), y, groups)
            folds.append({"train_idx": tr_idx.tolist(), "val_idx": te_idx.tolist()})
        elif mode == "groupkfold":
            gkf = GroupKFold(n_splits=K_outer)
            for tr_idx, te_idx in gkf.split(np.zeros(len(y)), y, groups):
                folds.append({"train_idx": tr_idx.tolist(), "val_idx": te_idx.tolist()})
        else:
            raise ValueError(f"unknown outer.scheme: {scheme}")
        repeats.append({"folds": folds, "mode": mode, "seed": seed0 + r})

    obj = {
        "repeats": repeats,
        "meta": {"R": R, "mode": mode, "test_size": test_size, "k_outer": K_outer, "seed0": seed0},
    }
    out_path = seq_dir / "splits_repeats.json"
    out_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    log.info("wrote %s (R=%d, mode=%s)", out_path, R, mode)
    return out_path


# ---- Helpers --------------------------------------------------------------
def _parse_lr_space(v, lo: float = 3e-5, hi: float = 3e-3) -> tuple[float, float]:
    if isinstance(v, str) and v.startswith("loguniform:"):
        a, b = v.split(":", 1)[1].split(",", 1)
        return float(a), float(b)
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return float(v[0]), float(v[1])
    try:
        x = float(v)
        return x / 3.0, x * 3.0
    except Exception:
        return lo, hi


def _maybe_empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _make_outer_val_split(df_tr: pd.DataFrame, n_splits_cfg: int, seed: int = 42):
    groups = df_tr["user_id"].astype(str).to_numpy()
    n_unique = int(len(np.unique(groups)))
    n_splits = int(max(2, min(n_splits_cfg, n_unique)))
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df_tr))
    rng.shuffle(idx)
    df_shuf = df_tr.iloc[idx].reset_index(drop=True)
    groups_shuf = df_shuf["user_id"].astype(str).to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    tr_i, va_i = next(gkf.split(np.zeros(len(df_shuf)), df_shuf["label"], groups_shuf))
    return df_shuf.iloc[tr_i].reset_index(drop=True), df_shuf.iloc[va_i].reset_index(drop=True)


def _train_loop(model, dl_tr, dl_va, lr, epochs, device, patience=5, min_delta=1e-3):
    opt = optim.AdamW(model.parameters(), lr=float(lr))
    stopper = EarlyStopper(patience=patience, min_delta=min_delta, mode="max")
    best_va = -1.0
    best_state = None
    for _ in range(epochs):
        model.train()
        for batch in dl_tr:
            X, M, y, _, T, _ = batch
            X, M, y, T = X.to(device), M.to(device), y.to(device), T.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(X, M, T), y)
            loss.backward()
            opt.step()
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for batch in dl_va:
                X, M, y, _, T, _ = batch
                X, M, T = X.to(device), M.to(device), T.to(device)
                p = model(X, M, T).argmax(-1).cpu().numpy()
                ys.append(y.numpy()); ps.append(p)
        f1_va = f1_score(np.concatenate(ys), np.concatenate(ps), average="macro")
        if f1_va > best_va:
            best_va = f1_va
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if stopper.step(best_va):
            break
    return best_va, best_state


def run_one_outer_fold(
    cfg: dict,
    idx_all: pd.DataFrame,
    train_idx: list[int],
    test_idx: list[int],
    optuna_trials: int | None = None,
) -> dict:
    """Optuna inside train_idx, then a single evaluation on test_idx."""
    import optuna
    from optuna.samplers import TPESampler

    seed = int(cfg.get("runtime", {}).get("seed", 42))
    seed_everything(seed)

    df_tr = idx_all.iloc[train_idx].reset_index(drop=True)
    df_te = idx_all.iloc[test_idx].reset_index(drop=True)
    groups_tr = df_tr["user_id"].astype(str).to_numpy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = int(idx_all["label"].nunique())
    SS = cfg.get("optuna", {}).get("search_space", {})
    SS_D = list(SS.get("d_model", [128, 192, 256, 384]))
    SS_H = list(SS.get("n_heads", [2, 4, 8]))
    SS_L = list(SS.get("n_layers", [1, 2, 3, 4]))
    SS_DR = list(SS.get("dropout", [0.1, 0.2, 0.3]))
    SS_BS = list(SS.get("batch_size", [16, 32, 64]))
    LR_LO, LR_HI = _parse_lr_space(SS.get("lr", "loguniform:3e-5,3e-3"))

    cv_n = int(cfg.get("cv", {}).get("n_splits", 5))
    epochs_inner = int(cfg.get("optuna", {}).get("objective_epochs",
                       cfg.get("training", {}).get("epochs_inner", 5)))
    epochs_outer = int(cfg.get("train", {}).get("epochs", 10))
    eval_bs = int(cfg.get("dataset", {}).get("batch_size", 64))
    turn_vocab = int(cfg.get("turn_id", {}).get("num_embeddings", 128))
    collate = make_collate(turn_vocab=turn_vocab)
    num_workers = int(cfg.get("dataset", {}).get("num_workers", 0))
    pin = bool(cfg.get("dataset", {}).get("pin_memory", torch.cuda.is_available()))

    def _loader(df, bs, shuffle):
        return DataLoader(
            SequenceDataset(df), batch_size=int(bs), shuffle=shuffle,
            collate_fn=collate, num_workers=num_workers, pin_memory=pin,
        )

    def objective(trial):
        seed_everything(seed + trial.number)
        d_model = trial.suggest_categorical("d_model", SS_D)
        n_heads = trial.suggest_categorical("n_heads", SS_H)
        if d_model % n_heads != 0:
            raise optuna.exceptions.TrialPruned()
        n_layers = trial.suggest_categorical("n_layers", SS_L)
        dropout = trial.suggest_categorical("dropout", SS_DR)
        lr = trial.suggest_float("lr", LR_LO, LR_HI, log=True)
        bs = trial.suggest_categorical("batch_size", SS_BS)

        n_unique = int(len(np.unique(groups_tr)))
        n_splits = max(2, min(cv_n, n_unique))
        gkf = GroupKFold(n_splits=n_splits)
        scores = []
        for fold_i, (tr_i, va_i) in enumerate(gkf.split(np.zeros(len(df_tr)), df_tr["label"], groups_tr)):
            _tr = df_tr.iloc[tr_i].reset_index(drop=True)
            _va = df_tr.iloc[va_i].reset_index(drop=True)
            try:
                model = build_model(
                    cfg, input_dim=int(idx_all["dim"].iloc[0]),
                    d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                    dropout=dropout, n_classes=n_classes, device=device,
                )
                best_va, _ = _train_loop(
                    model, _loader(_tr, bs, True), _loader(_va, eval_bs, False),
                    lr=lr, epochs=epochs_inner, device=device,
                )
                scores.append(best_va)
                trial.report(best_va, step=fold_i)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            finally:
                _maybe_empty_cache()
        return float(np.mean(scores)) if scores else 0.0

    n_trials = int(optuna_trials if optuna_trials is not None
                   else cfg.get("optuna", {}).get("n_trials", 20))
    warmup = int(cfg.get("optuna", {}).get("pruner", {}).get("warmup_epochs", 8))
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=warmup),
    )
    study.optimize(objective, n_trials=n_trials,
                   timeout=cfg.get("optuna", {}).get("timeout"),
                   show_progress_bar=False)
    best = study.best_params

    # Outer final training with held-out val for early stopping.
    seed_everything(seed)
    df_tr_in, df_va_out = _make_outer_val_split(df_tr, n_splits_cfg=cv_n, seed=seed)
    model = build_model(
        cfg, input_dim=int(idx_all["dim"].iloc[0]),
        d_model=int(best["d_model"]), n_heads=int(best["n_heads"]),
        n_layers=int(best["n_layers"]), dropout=float(best["dropout"]),
        n_classes=n_classes, device=device,
    )
    bs_outer = int(best["batch_size"])
    _, best_state = _train_loop(
        model,
        _loader(df_tr_in, bs_outer, True),
        _loader(df_va_out, eval_bs, False),
        lr=float(best["lr"]), epochs=epochs_outer, device=device,
    )
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for batch in _loader(df_te, eval_bs, False):
            X, M, y, _, T, _ = batch
            X, M, T = X.to(device), M.to(device), T.to(device)
            p = model(X, M, T).argmax(-1).cpu().numpy()
            ys.append(y.numpy()); ps.append(p)
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    result = {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro")),
        "best_params": best,
    }
    _maybe_empty_cache()
    return result


def run_repeated_eval(cfg: dict) -> dict:
    """End-to-end repeated external evaluation. Writes CSV + JSON summary."""
    root = Path(cfg["paths"]["root"])
    seq_dir = root / cfg["paths"]["outputs_dir"] / "sequences"
    rep_path = seq_dir / "splits_repeats.json"
    if not rep_path.exists():
        build_splits_repeats(cfg)
    rep_obj = json.loads(rep_path.read_text(encoding="utf-8"))
    idx_all = pd.read_parquet(seq_dir / "seq_index.parquet").reset_index(drop=True)

    rows = []
    for r, rep in enumerate(rep_obj["repeats"], 1):
        for f, fld in enumerate(rep["folds"], 1):
            log.info("repeat %d/%d  fold %d/%d", r, len(rep_obj["repeats"]), f, len(rep["folds"]))
            sc = run_one_outer_fold(cfg, idx_all, fld["train_idx"], fld["val_idx"])
            sc.update({"repeat": r, "fold": f})
            rows.append(sc)

    df = pd.DataFrame(rows)
    summary = {
        "macro_f1_mean": float(df["f1_macro"].mean()),
        "macro_f1_std": float(df["f1_macro"].std(ddof=1)) if len(df) > 1 else 0.0,
        "micro_f1_mean": float(df["f1_micro"].mean()),
        "micro_f1_std": float(df["f1_micro"].std(ddof=1)) if len(df) > 1 else 0.0,
        "n_rows": int(len(df)),
    }
    json_dump(summary, seq_dir / "repeats_external_eval.summary.json")
    csv_out = seq_dir / "repeats_external_eval.csv"
    df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    log.info("saved %s and summary", csv_out)
    return {"status": "ok", "summary": summary, "csv": str(csv_out)}


if __name__ == "__main__":
    import argparse
    from .config import load_and_prepare
    from .logging_setup import setup_logging
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    setup_logging("INFO")
    cfg, _, _ = load_and_prepare(args.config)
    run_repeated_eval(cfg)
