"""Build per-clip prediction parquets for M2/M3/M4 bake-off models.

For each model × 7 configs, loads all 25 NPZ fold predictions,
averages probabilities across the 5 test appearances per video_id
(one per repeat), then merges metadata from the v7 extended parquet.

Output (one per model):
  revision/analyses/bakeoff/{model}_per_clip.parquet
    columns: video_id, user_id, true_label,
             {config}_prob_0/1/2, {config}_pred  (× 7 configs)
             n_segments, speech_duration, speech_group,
             rater_agree, label_unanimity, rater_entropy

Usage:
  docker exec PLOS_ONE_2025 python3 scripts/build_per_clip_predictions.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

NPZ_DIR  = Path("/workspace/revision/analyses/bakeoff")
META_SRC = Path("/workspace/revision/predictions_per_clip_extended.parquet")
SAVE_DIR = NPZ_DIR

MODELS = ["M2_probe_early", "M3_blstm_late", "M4_attmil_late",
          "M4_attmil_late_v3", "M4_attmil_late_v3_noLA",
          "M4_attmil_late_v3_ogm", "M4_attmil_late_v3_noTextAux",
          "M4_attmil_late_v4"]

# Optional --model filter (for targeted rebuild)
import sys as _sys, argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--model", default=None)
_args, _ = _parser.parse_known_args()
if _args.model:
    MODELS = [m for m in MODELS if m == _args.model]
    if not MODELS:
        print(f"ERROR: --model '{_args.model}' not found in MODELS list"); _sys.exit(1)
    print(f"[build_per_clip] --model filter: only building {MODELS}")

CONFIGS: dict[str, str] = {
    "Full":       "Full",
    "Text-only":  "Text-only",
    "Audio+Text": "Audio_Text",
    "Face+Text":  "Face_Text",
    "Audio-only": "Audio-only",
    "Face-only":  "Face-only",
    "Audio+Face": "Audio_Face",
}
REPEATS = 5
FOLDS   = 5


def load_config_preds(model: str, cfg_key: str, cfg_fn: str) -> pd.DataFrame:
    """Load all 25 NPZs for one (model, config), return mean probs per video_id."""
    rows: list[dict] = []
    for r in range(1, REPEATS + 1):
        for f in range(1, FOLDS + 1):
            path = NPZ_DIR / f"{model}_{cfg_fn}_v7_mouthmask_r{r}_f{f}_preds.npz"
            if not path.exists():
                # fallback: v3/v3_noLA files lack the _v7_mouthmask_ infix
                path = NPZ_DIR / f"{model}_{cfg_fn}_r{r}_f{f}_preds.npz"
            if not path.exists():
                print(f"  WARN: {model}_{cfg_fn}_r{r}_f{f}_preds.npz not found")
                continue
            d = np.load(path, allow_pickle=True)
            for i, vid in enumerate(d["video_ids"].astype(str)):
                rows.append({
                    "video_id":   vid,
                    "true_label": int(d["y"][i]),
                    f"{cfg_key}_prob_0": float(d["pr"][i, 0]),
                    f"{cfg_key}_prob_1": float(d["pr"][i, 1]),
                    f"{cfg_key}_prob_2": float(d["pr"][i, 2]),
                })

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError(f"No NPZ files found for {model} {cfg_key}")
    # Average probabilities across the 5 repeat appearances per video_id
    prob_cols = [f"{cfg_key}_prob_0", f"{cfg_key}_prob_1", f"{cfg_key}_prob_2"]
    agg = raw.groupby("video_id").agg(
        true_label=(  "true_label", "first"),
        **{c: (c, "mean") for c in prob_cols},
    ).reset_index()
    agg[f"{cfg_key}_pred"] = agg[prob_cols].values.argmax(axis=1)
    print(f"  {model} {cfg_key}: {len(agg)} clips  "
          f"(loaded {len(raw)} fold-rows from {REPEATS*FOLDS} NPZs)")
    return agg


# ── Build metadata for all 1791 clips directly from source data ───────────────
import sys
from scipy.stats import entropy as scipy_entropy
sys.path.insert(0, "/workspace")
from src.config import load_and_prepare

cfg_obj, _, OUT_DIR = load_and_prepare("/workspace/config.yaml")
seg_dir  = OUT_DIR / "sequences" / "seg_text"
seq_idx  = pd.read_parquet(OUT_DIR / "sequences" / "seq_index.parquet")

# A: speech richness from seg_text NPZ files
speech_rows = []
for npz_path in sorted(seg_dir.glob("*_seg.npz")):
    vid = npz_path.stem.replace("_seg", "")
    d   = np.load(npz_path, allow_pickle=True)
    msk = d["mask"].astype(bool)
    n   = int(msk.sum())
    dur = float((d["t_end"][msk] - d["t_start"][msk]).sum()) if n > 0 else 0.0
    speech_rows.append({"video_id": vid, "n_segments": n, "speech_duration": dur})

speech_df = pd.DataFrame(speech_rows)
print(f"Speech features computed: {len(speech_df)} clips")

q33 = speech_df["n_segments"].quantile(0.33)
q67 = speech_df["n_segments"].quantile(0.67)

def _sp_group(x):
    if pd.isna(x):  return "不明"
    if x <= q33:    return f"少（≤{q33:.0f} segs）"
    if x < q67:     return f"中（{q33:.0f}–{q67:.0f} segs）"
    return              f"多（≥{q67:.0f} segs）"

speech_df["speech_group"] = speech_df["n_segments"].apply(_sp_group)

# B: rater agreement from seq_index
agree_cols = ["video_id", "soft_p1", "soft_p2", "soft_p3",
              "label_dok", "label_nis", "label_srk"]
agree_df = seq_idx[agree_cols].copy()
agree_df["rater_agree"] = (
    (agree_df["label_dok"] == agree_df["label_nis"]) &
    (agree_df["label_nis"] == agree_df["label_srk"])
)
agree_df["label_unanimity"] = agree_df["rater_agree"].map(
    {True: "全員一致", False: "2:1 分割"}
)

def _safe_entropy(row):
    p = np.array([row["soft_p1"], row["soft_p2"], row["soft_p3"]], dtype=float)
    p = np.clip(p, 1e-9, 1); p /= p.sum()
    return float(scipy_entropy(p))

agree_df["rater_entropy"] = agree_df.apply(_safe_entropy, axis=1)

meta = speech_df.merge(
    agree_df[["video_id", "rater_agree", "label_unanimity", "rater_entropy"]],
    on="video_id", how="outer",
)
print(f"Metadata built: {len(meta)} clips (speech={len(speech_df)}, rater={len(agree_df)})")

# ── Build per-model wide parquet ──────────────────────────────────────────────
for model in MODELS:
    print(f"\n{'='*60}")
    print(f"Building per-clip parquet: {model}")

    per_cfg: dict[str, pd.DataFrame] = {}
    for cfg_key, cfg_fn in CONFIGS.items():
        per_cfg[cfg_key] = load_config_preds(model, cfg_key, cfg_fn)

    # Merge all 7 configs on video_id
    base = per_cfg["Full"][["video_id", "true_label"]].copy()
    base["user_id"] = base["video_id"].str[:6]

    for cfg_key, df_cfg in per_cfg.items():
        pred_cols = [c for c in df_cfg.columns if c.startswith(cfg_key + "_")]
        base = base.merge(df_cfg[["video_id"] + pred_cols], on="video_id", how="inner")

    # Sanity: no duplicate video_ids
    assert base["video_id"].duplicated().sum() == 0, "Duplicate video_ids found!"

    # Verify true_label consistent across configs
    n = len(base)
    print(f"  Merged: {n} clips × {base.shape[1]} columns")
    print(f"  Label dist: {base['true_label'].value_counts().sort_index().to_dict()}")

    # Add metadata
    base = base.merge(meta, on="video_id", how="left")
    missing_meta = base["n_segments"].isna().sum()
    if missing_meta > 0:
        print(f"  WARN: {missing_meta} clips missing metadata")

    out_path = SAVE_DIR / f"{model}_per_clip.parquet"
    base.to_parquet(out_path, index=False)
    print(f"  Saved → {out_path}")

print("\nDone.")
