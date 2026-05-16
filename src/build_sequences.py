"""Stage 3b: build per-video sequence files (.npz) and seq_index.parquet.

Iterates over manifest rows, runs sequence_build for each video, applies
blended group normalization, then saves outputs to OUT/sequences/.

Ported from the loop following cell 26 in pipeline.ipynb.
"""
from __future__ import annotations
from pathlib import Path
import os, json
import numpy as np
import pandas as pd

from .sequence_build import build_sequence_for_video
from .normalize import blended_group_norm_windows
from .static_features import build_static_features
from .cohort import build_cohort
from .text_features import configure_sbert
from .utils import json_dump


def run_build_sequences(cfg: dict) -> dict:
    configure_sbert(cfg)
    root = Path(cfg["paths"]["root"])
    out_dir = root / cfg["paths"]["outputs_dir"]
    seq_dir = out_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    man_path = out_dir / cfg["paths"]["manifest_filename"]
    manifest = pd.read_parquet(man_path)

    static_features = build_static_features(cfg)
    df_cohort, lookups = build_cohort(static_features, cfg)
    cohort_by_video = lookups["cohort_by_video"]
    sleep_bin_by_video = lookups["sleep_bin_by_video"]
    workh_bin_by_video = lookups["workh_bin_by_video"]
    place_bin_by_video = lookups["place_bin_by_video"]
    state_bin_by_video = lookups["state_bin_by_video"]

    fallback_with_user = bool(cfg.get("cohort", {}).get("fallback_with_user", True))
    au_cols = cfg["dataset"]["face_features"]["au_r"]
    angle_cols = cfg["dataset"]["face_features"]["angles"]
    prosody_cols = cfg["parselmouth"]["prosody_cols"]
    mfcc_dim = int(cfg["parselmouth"]["mfcc_dim"])

    X_list, user_list, cohort_list = [], [], []
    meta_list, vid_list, lab_list = [], [], []
    mask_list, turn_list = [], []
    freeze_cols_global: list[int] = []

    for _, r in manifest.iterrows():
        user_id = r["user_id"]; vid = r["video_id"]
        face_csv, voice_csv, whisper_csv = r["face_csv"], r["voice_csv"], r["whisper_csv"]
        if not (isinstance(face_csv, str) and os.path.exists(face_csv)):
            continue
        if not (isinstance(voice_csv, str) and os.path.exists(voice_csv)):
            continue
        if not (isinstance(whisper_csv, str) and os.path.exists(whisper_csv)):
            continue
        if pd.isna(r.get("label")):
            continue
        lab = int(r["label"])
        X, mask, meta, turn_win = build_sequence_for_video(
            cfg, face_csv, voice_csv, whisper_csv,
            au_cols, angle_cols, prosody_cols, mfcc_dim,
        )
        X_list.append(X); mask_list.append(mask); turn_list.append(turn_win)
        user_list.append(user_id)
        fallback = (f"UNK|user:{user_id}" if fallback_with_user else "UNK")
        cohort_list.append(cohort_by_video.get(vid, fallback))
        meta_list.append(meta); vid_list.append(vid); lab_list.append(lab)

        d_face = int(meta["dims"].get("face", 0))
        d_voice = int(meta["dims"].get("voice", 0))
        d_text = int(meta["dims"].get("text", 0))
        if d_text > 0:
            start_text = d_face + d_voice
            end_text = start_text + d_text
            freeze_cols_global.extend(range(start_text, end_text))

    print(f"[build_sequences] videos processed: {len(X_list)}")
    assert len(X_list) > 0, "no samples built"
    assert len({len(X_list), len(mask_list), len(meta_list), len(vid_list), len(lab_list), len(user_list)}) == 1

    sample_dims = meta_list[0]["dims"]
    f = int(sample_dims.get("face", 0)); v = int(sample_dims.get("voice", 0))
    t = int(sample_dims.get("text", 0)); g = int(sample_dims.get("flags", 0))
    freeze_cols_global = sorted(set(freeze_cols_global))

    Xn_list, norm_meta = blended_group_norm_windows(
        X_list, user_list, cohort_list,
        eps=float(cfg["normalization"]["eps"]),
        freeze_col_indices=freeze_cols_global,
    )
    norm_meta["freeze_col_indices"] = freeze_cols_global
    norm_meta["dims_ranges"] = {
        "face": [0, f], "voice": [f, f + v],
        "text": [f + v, f + v + t], "flags": [f + v + t, f + v + t + g],
    }

    index_rows = []
    for i in range(len(Xn_list)):
        out_path = seq_dir / f"{vid_list[i]}.npz"
        _lab0 = int(lab_list[i]) - 1
        if _lab0 not in (0, 1, 2):
            _lab0 = 0
        row_in_man = manifest.loc[manifest["video_id"] == vid_list[i]]
        def _get(col, default=np.nan):
            return row_in_man.iloc[0][col] if (col in manifest.columns and len(row_in_man)) else default
        _ld = float(_get("label_dok", np.nan))
        _ln = float(_get("label_nis", np.nan))
        _ls = float(_get("label_srk", np.nan))
        _sp1 = float(_get("soft_p1", np.nan))
        _sp2 = float(_get("soft_p2", np.nan))
        _sp3 = float(_get("soft_p3", np.nan))
        _nrat = int(_get("n_raters_used", 0) or 0)

        np.savez_compressed(
            out_path,
            seq=Xn_list[i],
            mask=mask_list[i],
            label=np.int64(_lab0),
            latency_q2a=np.float32(meta_list[i]["latency_q2a"]),
            dims=json.dumps(meta_list[i]["dims"]),
            resp_text=meta_list[i]["resp_text"],
            win_ms=np.float32(meta_list[i]["win_ms"]),
            hop_ms=np.float32(meta_list[i]["hop_ms"]),
            turn_ids=turn_list[i].astype(np.int32),
            label_dok=np.array(_ld, dtype="float32"),
            label_nis=np.array(_ln, dtype="float32"),
            label_srk=np.array(_ls, dtype="float32"),
            soft_p=np.array([_sp1, _sp2, _sp3], dtype="float32"),
            n_raters=np.int64(_nrat),
        )
        index_rows.append({
            "user_id": user_list[i],
            "video_id": vid_list[i],
            "label": int(_lab0),
            "label_consensus_123": int(lab_list[i]),
            "seq_path": str(out_path),
            "n_steps": int(Xn_list[i].shape[0]),
            "dim": int(Xn_list[i].shape[1]),
            "latency_q2a": float(meta_list[i]["latency_q2a"]),
            "cohort": cohort_list[i],
            "label_dok": _ld if np.isfinite(_ld) else np.nan,
            "label_nis": _ln if np.isfinite(_ln) else np.nan,
            "label_srk": _ls if np.isfinite(_ls) else np.nan,
            "soft_p1": _sp1 if np.isfinite(_sp1) else np.nan,
            "soft_p2": _sp2 if np.isfinite(_sp2) else np.nan,
            "soft_p3": _sp3 if np.isfinite(_sp3) else np.nan,
            "n_raters_used": int(_nrat),
            "sleep_bin": sleep_bin_by_video.get(vid_list[i], "sleep:UNK"),
            "workh_bin": workh_bin_by_video.get(vid_list[i], "workh:UNK"),
            "place_bin": place_bin_by_video.get(vid_list[i], "place:UNK"),
            "state_bin": state_bin_by_video.get(vid_list[i], "state:UNK"),
        })

    idx_df = pd.DataFrame(index_rows)
    idx_path = seq_dir / "seq_index.parquet"
    idx_df.to_parquet(idx_path)
    json_dump(norm_meta, seq_dir / "norm_meta.json")
    print(f"[build_sequences] index: {idx_path} rows={len(idx_df)}")
    return {"status": "ok", "seq_index": str(idx_path), "seq_dir": str(seq_dir)}
