"""Shared feature loading for bake-off classifiers.

Loads the same feature sources as the v7 3-stream Cross-Modal Transformer:
  - face:  seq[:, 0:22]  -- OpenFace AU handcrafted (22d per frame, 20Hz)
  - voice: seq[:, 22:44] -- Parselmouth LLD handcrafted (22d per frame, 20Hz)
  - text:  {seg_dir}/{video_id}_seg.npz["X_text"] -- SBERT (S, 768) per segment
  - meta:  11d context (latency, sleep, hour, place, state) -- same as v6/v7

All classifiers in the bake-off use these same features so that only the
classifier architecture varies relative to the v7 3-stream Transformer.

Tabular feature dimensions:
  stat_pool: face(22x4=88) + voice(22x4=88) + text(768x2=1536) + meta(11) = 1723
  mean_pool: face(22)      + voice(22)       + text(768)         + meta(11) = 823
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace")
from src.static_features import build_static_features

META_DIM  = 11
FACE_DIM  = 22
VOICE_DIM = 22
TEXT_DIM  = 768
HZ        = 20   # face/voice frame rate

MOUTH_AU_COLS = list(range(7, 16))  # AU10/12/14/15/17/20/23/25/26 (9 mouth/jaw AUs)

TABULAR_STAT_DIM = FACE_DIM * 4 + VOICE_DIM * 4 + TEXT_DIM * 2 + META_DIM  # 1723
TABULAR_MEAN_DIM = FACE_DIM     + VOICE_DIM      + TEXT_DIM      + META_DIM  # 823


def build_speaking_mask(t_start_arr, t_end_arr, T: int, hz: int = HZ) -> np.ndarray:
    """Rasterise A-segment speaking intervals onto a boolean frame mask (identical to v7).

    seg_text npz contains A-only segments (Q excluded at build time), so this
    naturally produces a mask covering only respondent speaking intervals.
    """
    mask = np.zeros(T, dtype=bool)
    for ts, te in zip(t_start_arr, t_end_arr):
        f0 = max(0, min(int(round(float(ts) * hz)), T))
        f1 = max(0, min(int(round(float(te) * hz)), T))
        if f1 > f0:
            mask[f0:f1] = True
    return mask


def build_meta_lookup(idx_df: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    """Build video_id → (META_DIM,) array.  Identical logic to v6/v7 worker."""
    sf = build_static_features(cfg)
    merged = idx_df[["video_id"]].drop_duplicates().merge(sf, on="video_id", how="left")
    if "latency_q2a" in idx_df.columns:
        merged = merged.merge(
            idx_df[["video_id", "latency_q2a"]].drop_duplicates(),
            on="video_id", how="left",
        )
    else:
        merged = merged.assign(latency_q2a=0.0)

    sleep_mean = merged["survey_sleep_hours"].mean()
    sleep_std  = max(float(merged["survey_sleep_hours"].std()), 1e-6)

    lookup: dict[str, np.ndarray] = {}
    for _, row in merged.iterrows():
        vid = row["video_id"]
        lat   = float(row.get("latency_q2a", 0.0) or 0.0) / 10.0
        sleep = (float(row.get("survey_sleep_hours", sleep_mean) or sleep_mean)
                 - sleep_mean) / sleep_std
        try:
            h = int(str(vid).split("_")[1][8:10])
        except Exception:
            h = 12
        sin_h = math.sin(2 * math.pi * h / 24)
        cos_h = math.cos(2 * math.pi * h / 24)
        place = str(row.get("survey_place", "")).strip()
        p_oh  = [0.0, 0.0, 0.0]
        if place in ("1", "2", "3"):
            p_oh[int(place) - 1] = 1.0
        state = str(row.get("survey_state", "")).strip()
        s_oh  = [0.0, 0.0, 0.0, 0.0]
        if state in ("1", "2", "3", "4"):
            s_oh[int(state) - 1] = 1.0
        lookup[vid] = np.array(
            [lat, sleep, sin_h, cos_h] + p_oh + s_oh, dtype=np.float32
        )
    return lookup


def load_clip_seq(row, seg_dir: Path, meta_lookup: dict,
                  return_speech_intervals: bool = False):
    """Load one clip's raw sequences (valid frames/segments only).

    Returns:
      face  (T, 22)   -- valid frames
      voice (T, 22)   -- valid frames
      text  (S, 768)  -- valid segments
      meta  (11,)
      label int
      (t_start, t_end)  -- only when return_speech_intervals=True
    """
    npz   = np.load(row["seq_path"], allow_pickle=True)
    seq   = npz["seq"].astype(np.float32)   # (T_full, 815)
    mask  = npz["mask"].astype(np.float32)  # (T_full,)
    label = int(npz["label"])
    vid   = row["video_id"]

    valid_t = mask > 0
    face    = seq[valid_t, 0:22]    # (T, 22)
    voice   = seq[valid_t, 22:44]   # (T, 22)

    seg_path = seg_dir / f"{vid}_seg.npz"
    t_start = np.array([], dtype=np.float64)
    t_end   = np.array([], dtype=np.float64)
    if seg_path.exists():
        sg      = np.load(str(seg_path), allow_pickle=True)
        text    = sg["X_text"].astype(np.float32)          # (S_full, 768)
        seg_msk = (sg["mask"].astype(np.float32) > 0
                   if "mask" in sg else np.ones(len(text), dtype=bool))
        text    = text[seg_msk]                             # (S, 768)
        if return_speech_intervals:
            ts_full = sg["t_start"].astype(np.float64)
            te_full = sg.get("t_end", ts_full + 3.0).astype(np.float64)
            t_start = ts_full[seg_msk]
            t_end   = te_full[seg_msk]
    else:
        # Fallback: time-aligned SBERT from main seq (already at each frame)
        text = seq[valid_t, 44:44 + TEXT_DIM]              # (T, 768)

    if len(face)  == 0:
        face  = np.zeros((1, FACE_DIM),  dtype=np.float32)
        voice = np.zeros((1, VOICE_DIM), dtype=np.float32)
    if len(text)  == 0:
        text  = np.zeros((1, TEXT_DIM),  dtype=np.float32)

    meta = meta_lookup.get(vid, np.zeros(META_DIM, dtype=np.float32))
    if return_speech_intervals:
        return face, voice, text, meta, label, (t_start, t_end)
    return face, voice, text, meta, label


def tabular_stat(face, voice, text, meta) -> np.ndarray:
    """Stat-pool to 1723d flat vector.

    face/voice: [mean, std, max, min] per dim
    text:       [mean, std]           per dim  (S is small, max/min less robust)
    meta:       identity
    """
    def s4(x):
        return np.concatenate([x.mean(0), x.std(0), x.max(0), x.min(0)])

    def s2(x):
        return np.concatenate([x.mean(0), x.std(0)])

    return np.concatenate([
        s4(face).astype(np.float32),
        s4(voice).astype(np.float32),
        s2(text).astype(np.float32),
        meta.astype(np.float32),
    ])


def tabular_mean(face, voice, text, meta) -> np.ndarray:
    """Mean-pool to 823d flat vector."""
    return np.concatenate([
        face.mean(0).astype(np.float32),
        voice.mean(0).astype(np.float32),
        text.mean(0).astype(np.float32),
        meta.astype(np.float32),
    ])


def preload_sequences(
    idx_df: pd.DataFrame,
    seg_dir: Path,
    meta_lookup: dict,
    verbose: bool = True,
) -> list[tuple]:
    """Pre-load all clips into RAM as (face, voice, text, meta, label, (t_start, t_end)) tuples.

    Speech intervals are always cached alongside features so callers can apply
    mouth-mask on-the-fly without re-reading disk.
    Memory: ~1791 clips × ~200 frames × (22+22+768) × 4B ≈ 1.3 GB.
    """
    cache = []
    for i in range(len(idx_df)):
        row = idx_df.iloc[i]
        try:
            face, voice, text, meta, label, (t_start, t_end) = load_clip_seq(
                row, seg_dir, meta_lookup, return_speech_intervals=True)
        except Exception as e:
            if verbose:
                print(f"  WARN {row['video_id']}: {e}", flush=True)
            face    = np.zeros((1, FACE_DIM),  dtype=np.float32)
            voice   = np.zeros((1, VOICE_DIM), dtype=np.float32)
            text    = np.zeros((1, TEXT_DIM),  dtype=np.float32)
            meta    = np.zeros(META_DIM, dtype=np.float32)
            label   = 0
            t_start = np.array([], dtype=np.float64)
            t_end   = np.array([], dtype=np.float64)
        cache.append((face, voice, text, meta, label, (t_start, t_end)))
        if verbose and i % 300 == 0:
            print(f"  Cached {i}/{len(idx_df)} clips...", flush=True)
    return cache


# ── Dynamics feature helpers (for M4_attmil_dynamics) ────────────────────────

def _dyn_frames(t_start, t_end, T_full):
    f0 = max(0, int(round(float(t_start) * HZ)))
    f1 = min(int(round(float(t_end)   * HZ)), T_full)
    return f0, f1


def _seg_dyn_fv(sub: np.ndarray, au_dur_thr: float = 0.5) -> np.ndarray:
    """Compute 7 per-segment dynamics features, returns (7*D,) array.

    Order: [mean, wseg_std, range, rising_slope, falling_slope, velocity, au_dur]
    """
    sub = sub.astype(np.float32)
    D   = sub.shape[1]
    n   = len(sub)
    z   = np.zeros(D, dtype=np.float32)

    mean_fv = sub.mean(0)

    if n < 2:
        return np.concatenate([mean_fv, z, z, z, z, z, z])

    std_fv      = sub.std(0)
    range_fv    = sub.max(0) - sub.min(0)
    velocity_fv = np.abs(np.diff(sub, axis=0)).mean(0)
    au_dur_fv   = (np.abs(sub) > au_dur_thr).mean(0).astype(np.float32)

    d = np.diff(sub, axis=0)  # (n-1, D)

    if n < 3:
        return np.concatenate([mean_fv, std_fv, range_fv, z, z, velocity_fv, au_dur_fv])

    rising_fv  = np.zeros(D, dtype=np.float32)
    falling_fv = np.zeros(D, dtype=np.float32)
    for dim in range(D):
        for sign, out in [(1, rising_fv), (-1, falling_fv)]:
            idx = np.where(d[:, dim] * sign > 0)[0]
            if len(idx) < 2:
                continue
            t = idx.astype(np.float32)
            t -= t.mean()
            y = sub[idx + 1, dim]
            denom = float((t * t).sum())
            if denom > 1e-8:
                out[dim] = float((t * y).sum()) / denom

    return np.concatenate([
        mean_fv, std_fv, range_fv, rising_fv, falling_fv, velocity_fv, au_dur_fv
    ])


def _clip_speaking_contrast(face_full: np.ndarray, mask: np.ndarray,
                             t_start_arr, t_end_arr) -> np.ndarray:
    """face mean(speaking frames) - face mean(silent frames), mask-aware."""
    T   = face_full.shape[0]
    D   = face_full.shape[1]
    sp  = np.zeros(T, dtype=bool)
    for ts, te in zip(t_start_arr, t_end_arr):
        f0, f1 = _dyn_frames(ts, te, T)
        sp[f0:f1] = True
    vld = mask > 0
    spk = sp & vld
    sil = ~sp & vld
    if spk.sum() >= 5 and sil.sum() >= 5:
        return (face_full[spk].mean(0) - face_full[sil].mean(0)).astype(np.float32)
    return np.zeros(D, dtype=np.float32)


def _clip_inter_seg_std(seq_full: np.ndarray, t_start_arr, t_end_arr,
                         valid_seg_mask) -> np.ndarray:
    """std of per-segment means across segments (clip-level variation)."""
    T = seq_full.shape[0]
    D = seq_full.shape[1]
    seg_means = []
    for s, (ts, te) in enumerate(zip(t_start_arr, t_end_arr)):
        if valid_seg_mask[s] <= 0:
            continue
        f0, f1 = _dyn_frames(ts, te, T)
        if f1 > f0:
            seg_means.append(seq_full[f0:f1].mean(0))
    if len(seg_means) >= 2:
        return np.stack(seg_means).std(0).astype(np.float32)
    return np.zeros(D, dtype=np.float32)


def load_all_tabular(
    idx_df: pd.DataFrame,
    seg_dir: Path,
    meta_lookup: dict,
    mode: str = "stat",
    apply_mouth_mask: bool = False,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all clips → (X, y, groups).

    mode: "stat" → 1723d, "mean" → 823d
    apply_mouth_mask: zero mouth AU cols (7:16) during speaking frames before pooling.
    """
    pool_fn = tabular_stat if mode == "stat" else tabular_mean
    dim     = TABULAR_STAT_DIM if mode == "stat" else TABULAR_MEAN_DIM
    X_list, y_list = [], []
    for i in range(len(idx_df)):
        row = idx_df.iloc[i]
        try:
            face, voice, text, meta, label, (t_start, t_end) = load_clip_seq(
                row, seg_dir, meta_lookup, return_speech_intervals=True)
            if apply_mouth_mask and len(t_start) > 0:
                sp = build_speaking_mask(t_start, t_end, face.shape[0])
                if sp.any():
                    face = face.copy()
                    face[np.ix_(sp, MOUTH_AU_COLS)] = 0.0
            feat = pool_fn(face, voice, text, meta)
        except Exception as e:
            if verbose:
                print(f"  WARN {row['video_id']}: {e}", flush=True)
            feat  = np.zeros(dim, dtype=np.float32)
            label = 0
        X_list.append(feat)
        y_list.append(label)
        if verbose and i % 300 == 0:
            print(f"  Loaded {i}/{len(idx_df)} clips...", flush=True)
    X      = np.stack(X_list).astype(np.float32)
    y      = np.array(y_list, dtype=np.int64)
    groups = idx_df["user_id"].astype(str).to_numpy()
    return X, y, groups
