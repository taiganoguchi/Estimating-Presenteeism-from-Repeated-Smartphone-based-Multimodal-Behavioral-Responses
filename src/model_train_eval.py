"""Optuna search + final retrain (cells 35-37, 40).

Replaces the previous stub. Imports the Transformer model from model_define.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

from .dataset import SequenceDataset, make_collate, build_state_weights
from .model_define import TransformerClassifier, build_model
from .utils import json_dump, seed_everything


# ---- EarlyStopper (cell 40) ----
class EarlyStopper:
    def __init__(self, patience: int = 5, min_delta: float = 1e-3, mode: str = "max"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = None
        self.count = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False
        improved = (
            (value > self.best + self.min_delta) if self.mode == "max"
            else (value < self.best - self.min_delta)
        )
        if improved:
            self.best = value
            self.count = 0
        else:
            self.count += 1
        return self.count >= self.patience


def _device(cfg: dict) -> str:
    dev = cfg.get("runtime", {}).get("device", "auto")
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return dev


def _train_one_epoch(model, loader, optimizer, device, state_w=None):
    model.train()
    crit = nn.CrossEntropyLoss(reduction="none")
    total = 0.0
    for batch in loader:
        seq, mask, label, _, turn_ids, idxs = batch
        seq, mask, label = seq.to(device), mask.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(seq, mask, turn_ids.to(device))
        loss_vec = crit(out, label)
        if state_w is not None and idxs is not None:
            wb = torch.tensor(state_w[idxs.cpu().numpy()], device=device, dtype=torch.float32)
            loss = (loss_vec * wb).mean()
        else:
            loss = loss_vec.mean()
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(len(loader), 1)


def _eval_macro_f1(model, loader, device) -> float:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            seq, mask, label, _, turn_ids, _ = batch
            seq, mask, turn_ids = seq.to(device), mask.to(device), turn_ids.to(device)
            pred = model(seq, mask, turn_ids).argmax(1).cpu().numpy()
            y_true.extend(label.numpy())
            y_pred.extend(pred)
    return float(f1_score(y_true, y_pred, average="macro"))


def _resolve_train_pool(split_obj: dict, n_all: int):
    if "folds" in split_obj:
        tr_lists = [np.array(f["train_idx"], dtype=int) for f in split_obj["folds"]]
        if not tr_lists:
            raise ValueError("split.folds is empty")
        tr_all = np.unique(np.concatenate(tr_lists)).tolist()
        return tr_all
    return split_obj["train_idx"]


def run_optuna(cfg: dict) -> dict:
    import optuna
    seed_everything(int(cfg.get("runtime", {}).get("seed", 42)))
    root = Path(cfg["paths"]["root"])
    seq_dir = root / cfg["paths"]["outputs_dir"] / "sequences"
    idx_all = pd.read_parquet(seq_dir / "seq_index.parquet").reset_index(drop=True)

    sp_holdout = seq_dir / "split.holdout.json"
    sp_default = seq_dir / "split.json"
    sp_path = sp_holdout if sp_holdout.exists() else sp_default
    assert sp_path.exists(), f"split file not found"
    split = json.loads(sp_path.read_text(encoding="utf-8"))
    train_pool = _resolve_train_pool(split, len(idx_all))
    idx_df = idx_all.iloc[train_pool].reset_index(drop=True)

    device = _device(cfg)
    turn_vocab = int(cfg.get("turn_id", {}).get("num_embeddings", 128))
    collate = make_collate(turn_vocab=turn_vocab)
    SS = cfg.get("optuna", {}).get("search_space", {})
    epochs_inner = int(cfg.get("optuna", {}).get("objective_epochs", 5))
    cv_n = int(cfg.get("cv", {}).get("n_splits", 5))
    n_unique = idx_df["user_id"].astype(str).nunique()

    def objective(trial):
        d_model = trial.suggest_categorical("d_model", SS.get("d_model", [128, 192, 256]))
        n_heads = trial.suggest_categorical("n_heads", SS.get("n_heads", [2, 4, 8]))
        n_layers = trial.suggest_categorical("n_layers", SS.get("n_layers", [2, 3, 4]))
        dropout = trial.suggest_categorical("dropout", SS.get("dropout", [0.1, 0.2, 0.3]))
        # lr range parsed from "loguniform:lo,hi" or list
        lr_spec = SS.get("lr", ["loguniform:1e-4,5e-4"])
        if isinstance(lr_spec, list) and lr_spec and isinstance(lr_spec[0], str) and lr_spec[0].startswith("loguniform:"):
            lo, hi = lr_spec[0].split(":", 1)[1].split(",", 1)
            lr = trial.suggest_float("lr", float(lo), float(hi), log=True)
        else:
            lr = trial.suggest_categorical("lr", lr_spec)
        bs = trial.suggest_categorical("batch_size", SS.get("batch_size", [8, 16, 32]))

        if d_model % n_heads != 0:
            raise optuna.exceptions.TrialPruned()

        gkf = GroupKFold(n_splits=max(2, min(cv_n, n_unique)))
        scores = []
        for tr_i, va_i in gkf.split(idx_df, idx_df["label"], groups=idx_df["user_id"]):
            _tr = idx_df.iloc[tr_i].reset_index(drop=True)
            _va = idx_df.iloc[va_i].reset_index(drop=True)
            sw = build_state_weights(_tr, col="sleep_bin", pow_gamma=float(cfg.get("weights", {}).get("gamma", 1.0)))
            dl_tr = DataLoader(SequenceDataset(_tr), batch_size=int(bs), shuffle=True, collate_fn=collate)
            dl_va = DataLoader(SequenceDataset(_va), batch_size=int(bs), shuffle=False, collate_fn=collate)
            model = build_model(
                cfg, input_dim=int(idx_df["dim"].iloc[0]),
                d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                dropout=dropout, n_classes=int(idx_df["label"].nunique()), device=device,
            )
            opt = optim.AdamW(model.parameters(), lr=float(lr))
            for _ in range(epochs_inner):
                _train_one_epoch(model, dl_tr, opt, device, state_w=sw)
            scores.append(_eval_macro_f1(model, dl_va, device))
            del model, opt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return float(np.mean(scores)) if scores else 0.0

    seed = int(cfg.get("runtime", {}).get("seed", 42))
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True),
    )
    opt_cfg = cfg.get("optuna", {})
    study.optimize(
        objective,
        n_trials=int(opt_cfg.get("n_trials", 20)),
        timeout=opt_cfg.get("timeout"),
        show_progress_bar=bool(opt_cfg.get("show_progress_bar", False)),
    )
    print("Best params:", study.best_params)
    out_dir = seq_dir
    json_dump(study.best_params, out_dir / "best_params.json")
    return {"status": "ok", "best_params": study.best_params, "best_params_path": str(out_dir / "best_params.json")}


def run_train(cfg: dict, best_params: dict | None = None) -> dict:
    seed_everything(int(cfg.get("runtime", {}).get("seed", 42)))
    root = Path(cfg["paths"]["root"])
    seq_dir = root / cfg["paths"]["outputs_dir"] / "sequences"
    idx_all = pd.read_parquet(seq_dir / "seq_index.parquet").reset_index(drop=True)

    if best_params is None:
        bp_path = seq_dir / "best_params.json"
        if bp_path.exists():
            best_params = json.loads(bp_path.read_text(encoding="utf-8"))
        else:
            tr = cfg.get("train", {})
            best_params = {
                "d_model": 256, "n_heads": 4, "n_layers": 2,
                "dropout": 0.2, "lr": float(tr.get("lr", 1.5e-4)),
                "batch_size": int(tr.get("batch_size", 32)),
            }

    device = _device(cfg)
    turn_vocab = int(cfg.get("turn_id", {}).get("num_embeddings", 128))
    collate = make_collate(turn_vocab=turn_vocab)

    sp_holdout = seq_dir / "split.holdout.json"
    sp_default = seq_dir / "split.json"
    sp_path = sp_holdout if sp_holdout.exists() else sp_default
    split = json.loads(sp_path.read_text(encoding="utf-8"))
    train_pool = _resolve_train_pool(split, len(idx_all))
    idx_df = idx_all.iloc[train_pool].reset_index(drop=True)

    state_w_full = build_state_weights(idx_df, col="sleep_bin", pow_gamma=float(cfg.get("weights", {}).get("gamma", 1.0)))
    full_loader = DataLoader(
        SequenceDataset(idx_df),
        batch_size=int(best_params["batch_size"]),
        shuffle=True,
        collate_fn=collate,
    )
    model = build_model(
        cfg,
        input_dim=int(idx_df["dim"].iloc[0]),
        d_model=int(best_params["d_model"]),
        n_heads=int(best_params["n_heads"]),
        n_layers=int(best_params["n_layers"]),
        dropout=float(best_params["dropout"]),
        n_classes=int(idx_df["label"].nunique()),
        device=device,
    )
    opt = optim.AdamW(model.parameters(), lr=float(best_params["lr"]))
    epochs = int(cfg.get("train", {}).get("epochs", 10))
    for ep in range(epochs):
        loss = _train_one_epoch(model, full_loader, opt, device, state_w=state_w_full)
        print(f"[train] epoch {ep + 1}/{epochs} loss={loss:.4f}")

    ckpt_path = seq_dir / "best_model.pt"
    hparams = {
        "input_dim": int(idx_df["dim"].iloc[0]),
        "n_classes": int(idx_df["label"].nunique()),
        "d_model": int(best_params["d_model"]),
        "n_heads": int(best_params["n_heads"]),
        "n_layers": int(best_params["n_layers"]),
        "dropout": float(best_params["dropout"]),
        "lr": float(best_params["lr"]),
        "batch_size": int(best_params["batch_size"]),
        "turn_emb_dim": int(cfg.get("turn_id", {}).get("embed_dim", 8)),
        "turn_vocab": turn_vocab,
    }
    torch.save({"state_dict": model.state_dict(), "hparams": hparams}, ckpt_path)
    json_dump(hparams, seq_dir / "best_model.meta.json")
    print(f"[train] saved -> {ckpt_path}")
    return {"status": "ok", "ckpt": str(ckpt_path), "hparams": hparams}
