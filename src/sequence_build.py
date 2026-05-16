"""Per-video sequence construction (OpenFace + Parselmouth + text -> windows).

Ported from pipeline.ipynb cell 24.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .text_features import (
    parse_whisper_segments,
    extract_latency_and_text,
    make_turn_id_series_from_whisper,
    build_text_and_flags_on_timeline,
    sbert_encode,
)


def _interp_to_base(t_src, x_src, t_base):
    if len(t_src) == 0 or x_src.shape[0] == 0:
        return np.full((len(t_base), x_src.shape[1]), np.nan, dtype=np.float32)
    out = np.empty((len(t_base), x_src.shape[1]), dtype=np.float32)
    for j in range(x_src.shape[1]):
        y = x_src[:, j]
        mask = np.isfinite(y)
        if mask.sum() < 2:
            if mask.sum() == 1:
                out[:, j] = np.interp(t_base, t_src[mask], y[mask])
            else:
                out[:, j] = np.nan
        else:
            out[:, j] = np.interp(t_base, t_src[mask], y[mask])
    return out


def _uniform_downsample_idx(n: int, k: int) -> np.ndarray:
    if n <= k:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, num=k, dtype=np.int64)


def build_sequence_for_video(
    cfg: dict,
    face_csv: str | Path,
    voice_csv: str | Path,
    whisper_csv: str | Path,
    au_cols: list[str],
    angle_cols: list[str],
    prosody_cols: list[str],
    mfcc_dim: int,
):
    base_hz = int(cfg["sequence"]["base_hz"])
    dt = 1.0 / base_hz
    win = float(cfg["sequence"]["window_ms"]) / 1000.0
    hop = float(cfg["sequence"]["hop_ms"]) / 1000.0
    max_len = int(cfg["sequence"]["max_seq_len"])
    trunc_mode = cfg["sequence"].get("truncate_mode", "uniform")

    segs = parse_whisper_segments(whisper_csv)
    lat_text = extract_latency_and_text(whisper_csv)
    latency_q2a = float(lat_text.get("latency_q2a", np.nan))
    sbert_dim_v = int(cfg.get("features", {}).get("sbert", {}).get("dim", 768))

    face_df = pd.read_csv(face_csv)
    if "timestamp" in face_df.columns:
        t_face = pd.to_numeric(face_df["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        fps = float(cfg["openface"]["default_fps"])
        fr = face_df["frame"] if "frame" in face_df.columns else pd.Series(np.arange(len(face_df)))
        t_face = pd.to_numeric(fr, errors="coerce").to_numpy(dtype=np.float64) / max(fps, 1e-6)
    face_cols = [c for c in au_cols + angle_cols if c in face_df.columns]
    Xf = face_df[face_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)

    v_df = pd.read_csv(voice_csv)
    aliases = {c.lower().strip(): c for c in v_df.columns}
    time_col = aliases.get("time") or "Time"
    if time_col not in v_df.columns:
        raise ValueError("Parselmouth csv must have 'Time' column (case-insensitive).")
    t_voice = pd.to_numeric(v_df[time_col], errors="coerce").to_numpy(dtype=np.float64)
    prosody_fixed = [c for c in prosody_cols if c in v_df.columns]
    mfcc_cols = [f"MFCC{i}" for i in range(1, mfcc_dim + 1) if f"MFCC{i}" in v_df.columns]
    v_cols = prosody_fixed + mfcc_cols
    Xv = v_df[v_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)

    t_start = 0.0
    t_end = float(np.nanmax([np.nanmax(t_face), np.nanmax(t_voice)]))
    if not np.isfinite(t_end) or t_end <= 0:
        t_end = 1.0
    t_base = np.arange(t_start, t_end + 1e-9, dt, dtype=np.float64)

    turn_ids = make_turn_id_series_from_whisper(whisper_csv, t_base)
    Xf_base = _interp_to_base(t_face, Xf, t_base) if len(face_cols) > 0 else np.empty((len(t_base), 0), np.float32)
    Xv_base = _interp_to_base(t_voice, Xv, t_base) if len(v_cols) > 0 else np.empty((len(t_base), 0), np.float32)
    Xt, Xflags = build_text_and_flags_on_timeline(
        t_base=t_base, segs=segs, sbert_encode_fn=sbert_encode, sbert_dim_v=sbert_dim_v
    )
    X_all = np.concatenate([Xf_base, Xv_base, Xt, Xflags], axis=1)

    step = max(1, int(round(hop / dt)))
    win_n = max(1, int(round(win / dt)))
    feats, t0s = [], []
    for s in range(0, max(1, len(t_base) - win_n + 1), step):
        seg = X_all[s: s + win_n]
        feats.append(np.nanmean(seg, axis=0))
        t0s.append(t_base[s])
    feats = np.stack(feats, axis=0) if feats else np.zeros((1, X_all.shape[1]), np.float32)
    mask = (~np.isnan(feats)).any(axis=1).astype(np.float32)

    if len(feats) > max_len:
        if trunc_mode == "head":
            idx = np.arange(max_len, dtype=np.int64)
        elif trunc_mode == "tail":
            idx = np.arange(len(feats) - max_len, len(feats), dtype=np.int64)
        elif trunc_mode == "center":
            s0 = (len(feats) - max_len) // 2
            idx = np.arange(s0, s0 + max_len, dtype=np.int64)
        else:
            idx = _uniform_downsample_idx(len(feats), max_len)
        feats = feats[idx]
        mask = mask[idx]

    meta = {
        "latency_q2a": latency_q2a,
        "resp_text": str(lat_text.get("resp_text", "")),
        "n_steps": int(feats.shape[0]),
        "dims": {
            "face": len(face_cols),
            "voice": len(v_cols),
            "text": int(Xt.shape[1]),
            "flags": int(Xflags.shape[1]),
            "turn_id": "separate",
        },
        "t0_list": t0s[: len(feats)],
        "win_ms": cfg["sequence"]["window_ms"],
        "hop_ms": cfg["sequence"]["hop_ms"],
    }
    turn_win = []
    for s in range(0, max(1, len(t_base) - win_n + 1), step):
        turn_win.append(int(turn_ids[s]))
    turn_win = np.array(turn_win, dtype=np.int32)
    if len(feats) < len(turn_win):
        turn_win = turn_win[: len(feats)]
    return feats.astype(np.float32), mask.astype(np.float32), meta, turn_win
