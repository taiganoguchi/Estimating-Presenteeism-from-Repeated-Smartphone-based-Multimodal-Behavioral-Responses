"""Evaluation: load best model, predict on test split, write metrics report.

Ported from pipeline.ipynb cells 39, 41 (core only — extended ablation/analysis
cells remain in the notebook).
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from .dataset import SequenceDataset, make_collate
from .model_define import TransformerClassifier
from .utils import json_dump


def load_best_model(seq_dir: Path, device: str | None = None, strict: bool = False):
    seq_dir = Path(seq_dir)
    ckpt = seq_dir / "best_model.pt"
    meta = seq_dir / "best_model.meta.json"
    assert ckpt.exists() and meta.exists()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    hparams = json.loads(meta.read_text(encoding="utf-8"))
    model = TransformerClassifier(
        input_dim=int(hparams["input_dim"]),
        d_model=int(hparams["d_model"]),
        n_heads=int(hparams["n_heads"]),
        n_layers=int(hparams["n_layers"]),
        dropout=float(hparams["dropout"]),
        n_classes=int(hparams.get("n_classes", 3)),
        turn_emb_dim=int(hparams.get("turn_emb_dim", 8)),
        turn_vocab=int(hparams.get("turn_vocab", 128)),
        train_turn_emb=False,
    ).to(device)
    obj = torch.load(ckpt, map_location=device)
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    model.load_state_dict(state, strict=strict)
    model.eval()
    return model, hparams


def predict_for_index(df_subset: pd.DataFrame, model, batch_size: int = 64, turn_vocab: int = 128) -> np.ndarray:
    collate = make_collate(turn_vocab=turn_vocab)
    dl = DataLoader(SequenceDataset(df_subset), batch_size=batch_size, shuffle=False, collate_fn=collate)
    device = next(model.parameters()).device
    preds = []
    with torch.no_grad():
        for batch in dl:
            seq, mask, label, _, turn_ids, _ = batch
            logits = model(seq.to(device), mask.to(device), turn_ids.to(device))
            preds.extend(logits.argmax(1).cpu().numpy())
    return np.array(preds)


def summarize_user_dependency(idx_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    aux = pd.DataFrame({
        "u": idx_df["user_id"].astype(str).to_numpy(),
        "y": y_true,
        "p": y_pred,
    })
    return (
        aux.groupby("u")
        .apply(lambda g: pd.Series({
            "support": len(g),
            "f1_macro": f1_score(g["y"], g["p"], average="macro", zero_division=0),
            "f1_weighted": f1_score(g["y"], g["p"], average="weighted", zero_division=0),
            "acc": float((g["y"] == g["p"]).mean()),
        }))
        .reset_index()
        .sort_values("support", ascending=False)
    )


def run_evaluate(cfg: dict) -> dict:
    root = Path(cfg["paths"]["root"])
    out_dir = root / cfg["paths"]["outputs_dir"]
    seq_dir = out_dir / "sequences"
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    idx_all = pd.read_parquet(seq_dir / "seq_index.parquet").reset_index(drop=True)
    sp_holdout = seq_dir / "split.holdout.json"
    if not sp_holdout.exists():
        raise FileNotFoundError(f"holdout split missing: {sp_holdout}")
    split = json.loads(sp_holdout.read_text(encoding="utf-8"))
    test_idx = split["test_idx"]
    df_te = idx_all.iloc[test_idx].reset_index(drop=True)

    model, hparams = load_best_model(seq_dir)
    turn_vocab = int(hparams.get("turn_vocab", 128))
    y_pred = predict_for_index(df_te, model, batch_size=64, turn_vocab=turn_vocab)
    y_true = df_te["label"].astype(int).to_numpy()

    f1m = float(f1_score(y_true, y_pred, average="macro"))
    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    (reports_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    json_dump(
        {"f1_macro": f1m, "confusion_matrix": cm.tolist(), "n_test": int(len(y_true))},
        reports_dir / "metrics.json",
    )

    per_user = summarize_user_dependency(df_te, y_true, y_pred)
    per_user.to_csv(reports_dir / "per_user_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"[evaluate] f1_macro={f1m:.4f} -> {reports_dir}")
    return {"status": "ok", "f1_macro": f1m, "reports_dir": str(reports_dir)}
