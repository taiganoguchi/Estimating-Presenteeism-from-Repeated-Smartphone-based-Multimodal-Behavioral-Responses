"""Stage 1: scan results/ to build index_metadata.parquet.

Ported from pipeline.ipynb cell 9.
"""
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd


def run_data_import(cfg: dict) -> dict:
    root = Path(cfg["paths"]["root"])
    res_dir = root / cfg["paths"]["results_dir"]
    out_dir = root / cfg["paths"]["outputs_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_path = out_dir / "index_metadata.parquet"

    user_pat = re.compile(cfg["scan"]["user_dir_regex"])
    vid_pat = re.compile(cfg["scan"]["video_id_regex"])
    suf = cfg["scan"]["expected_suffixes"]

    rows: list[dict] = []
    if not res_dir.exists():
        print(f"[warn] results_dir not found: {res_dir}")
        pd.DataFrame(
            columns=["user_id", "video_id", "face_csv", "voice_csv", "whisper_csv"]
        ).to_parquet(idx_path)
        return {"status": "ok", "index_metadata": str(idx_path)}

    for user_dir in sorted(p for p in res_dir.iterdir() if p.is_dir() and user_pat.match(p.name)):
        user_id = user_dir.name
        for video_dir in sorted(p for p in user_dir.iterdir() if p.is_dir()):
            vid = video_dir.name
            if not vid_pat.match(vid):
                continue
            base = video_dir / vid

            def exists_with_suffix(key: str) -> str | None:
                path = Path(str(base) + suf.get(key, ""))
                return str(path) if path.exists() else None

            rows.append({
                "user_id": user_id,
                "video_id": vid,
                "face_csv":    exists_with_suffix("openface_results"),
                "voice_csv":   exists_with_suffix("parselmouth"),
                "whisper_csv": exists_with_suffix("whisper"),
            })

    index_df = pd.DataFrame(rows)
    index_df.to_parquet(idx_path)
    print(f"[data_import] saved: {idx_path} | rows={len(index_df)}")
    return {"status": "ok", "index_metadata": str(idx_path)}
