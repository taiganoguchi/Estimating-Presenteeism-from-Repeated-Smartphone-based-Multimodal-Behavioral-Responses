"""Stage 2: build sample_manifest.parquet (DOK/NIS/SRK -> consensus label).

Ported from pipeline.ipynb cell 11.
"""
from __future__ import annotations
from pathlib import Path
import re, math
import numpy as np
import pandas as pd

REQUIRED_COLS = ["user_id", "video_id", "face_csv", "voice_csv", "whisper_csv"]


def _extract_video_id_from_filename(s: str, vid_pat: re.Pattern) -> str | None:
    if not isinstance(s, str):
        return None
    s = s.strip()
    s = re.sub(r"^cap_", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\.mp4$", "", s, flags=re.IGNORECASE)
    m = vid_pat.search(s)
    return m.group(0) if m else None


def _normalize_tri_label(v):
    if pd.isna(v):
        return np.nan
    try:
        iv = int(float(v))
    except Exception:
        m = re.search(r"\b([0-9]+)\b", str(v))
        if not m:
            return np.nan
        iv = int(m.group(1))
    return iv if iv in (1, 2, 3) else np.nan


def load_doctor_labels_multi(cfg: dict) -> pd.DataFrame:
    """Excel 1 枚目から video_id と DOK/NIS/SRK を抽出。"""
    root = Path(cfg["paths"]["root"])
    xls_path = root / cfg["paths"]["doctor_xlsx"]
    vid_pat = re.compile(cfg["scan"]["video_id_regex"])
    if not xls_path.exists():
        print(f"[warn] doctor_xlsx not found: {xls_path}")
        return pd.DataFrame(columns=["video_id", "label_dok", "label_nis", "label_srk"])

    df = pd.read_excel(xls_path, sheet_name=0)
    col_file = next(
        (c for c in ["動画ファイル", "ファイル名", "Video", "video", "filename", "file"] if c in df.columns),
        None,
    )
    if col_file is None:
        print(f"[warn] video filename column not found in {xls_path.name}")
        return pd.DataFrame(columns=["video_id", "label_dok", "label_nis", "label_srk"])

    col_err = next((c for c in ["記録エラー", "エラー", "error", "Error"] if c in df.columns), None)
    if col_err is not None:
        df = df[df[col_err].isna()].copy()

    def pick_col(cands):
        return next((c for c in cands if c in df.columns), None)

    c_dok = pick_col(["DOK", "評価DOK", "dok"])
    c_nis = pick_col(["NIS", "評価NIS", "nis"])
    c_srk = pick_col(["SRK", "評価SRK", "srk"])

    df["video_id"] = df[col_file].astype(str).map(lambda x: _extract_video_id_from_filename(x, vid_pat))
    df["label_dok"] = df[c_dok].map(_normalize_tri_label).astype("Int64") if c_dok else pd.Series(dtype="Int64")
    df["label_nis"] = df[c_nis].map(_normalize_tri_label).astype("Int64") if c_nis else pd.Series(dtype="Int64")
    df["label_srk"] = df[c_srk].map(_normalize_tri_label).astype("Int64") if c_srk else pd.Series(dtype="Int64")

    return (
        df.dropna(subset=["video_id"])
          .drop_duplicates(subset=["video_id"], keep="last")
          .loc[:, ["video_id", "label_dok", "label_nis", "label_srk"]]
    )


def _round_half_away_from_zero(x: float) -> int:
    return int(np.sign(x) * math.floor(abs(x) + 0.5))


def consensus_label(row, mode: str = "majority", tie: str = "nearest"):
    vals = [v for v in [row.get("label_dok"), row.get("label_nis"), row.get("label_srk")] if pd.notna(v)]
    vals = [int(v) for v in vals if v in (1, 2, 3)]
    if len(vals) == 0:
        return np.nan
    if mode == "dok_only":
        return row.get("label_dok", np.nan)
    if mode == "nis_only":
        return row.get("label_nis", np.nan)
    if mode == "srk_only":
        return row.get("label_srk", np.nan)
    if mode == "average_round":
        m = float(np.mean(vals))
        if tie == "nearest":
            return int(np.clip(_round_half_away_from_zero(m), 1, 3))
        if tie == "up":
            return int(np.clip(math.ceil(m), 1, 3))
        return int(np.clip(math.floor(m), 1, 3))

    vc = pd.Series(vals).value_counts()
    top = vc.max()
    # 3 名すべて別ラベル -> 教師なし
    if mode == "majority" and len(vals) == 3 and top == 1:
        return np.nan
    cands = sorted(vc[vc == top].index.tolist())
    if len(cands) == 1:
        return int(cands[0])
    if tie == "up":
        return int(max(cands))
    if tie == "down":
        return int(min(cands))
    m = float(np.mean(vals))
    cands.sort(key=lambda k: (abs(k - m), -k))
    return int(cands[0])


def soft_target(row):
    xs = [row.get("label_dok"), row.get("label_nis"), row.get("label_srk")]
    xs = [int(v) for v in xs if pd.notna(v) and v in (1, 2, 3)]
    if not xs:
        return None
    p = np.zeros(3, dtype=np.float32)
    for v in xs:
        p[v - 1] += 1.0
    p /= p.sum()
    return p


def run_preprocess(cfg: dict, index_metadata_path: str | Path) -> dict:
    root = Path(cfg["paths"]["root"])
    out_dir = root / cfg["paths"]["outputs_dir"]
    manifest_path = out_dir / cfg["paths"].get("manifest_filename", "sample_manifest.parquet")

    idx = pd.read_parquet(index_metadata_path).copy()
    for c in REQUIRED_COLS:
        if c not in idx.columns:
            idx[c] = pd.NA

    lab = load_doctor_labels_multi(cfg)
    df = idx.merge(lab, how="left", on="video_id")

    LCFG = cfg.get("labels", {})
    mode = LCFG.get("consensus", "majority")
    tie = LCFG.get("tie_break", "nearest")
    emit_soft = bool(LCFG.get("emit_soft_target", True))

    df["label_consensus"] = df.apply(lambda r: consensus_label(r, mode, tie), axis=1).astype("Int64")
    if emit_soft:
        soft = df.apply(soft_target, axis=1)
        df["soft_p1"] = soft.map(lambda v: None if v is None else float(v[0]))
        df["soft_p2"] = soft.map(lambda v: None if v is None else float(v[1]))
        df["soft_p3"] = soft.map(lambda v: None if v is None else float(v[2]))
        df["n_raters_used"] = df[["label_dok", "label_nis", "label_srk"]].notna().sum(axis=1)

    df["label"] = df["label_consensus"].astype("Int64")
    df = df[df["label"].notna()].reset_index(drop=True)

    keep = df["face_csv"].notna() & df["voice_csv"].notna() & df["whisper_csv"].notna()
    df = df.loc[keep].reset_index(drop=True)

    col_order = [
        "user_id", "video_id",
        "label", "label_consensus", "label_dok", "label_nis", "label_srk",
        "soft_p1", "soft_p2", "soft_p3", "n_raters_used",
        "face_csv", "voice_csv", "whisper_csv",
    ]
    cols = [c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]
    df = df[cols]

    df.to_parquet(manifest_path)
    print(f"[preprocess] saved: {manifest_path} | rows={len(df)}")
    return {"status": "ok", "manifest": str(manifest_path)}
