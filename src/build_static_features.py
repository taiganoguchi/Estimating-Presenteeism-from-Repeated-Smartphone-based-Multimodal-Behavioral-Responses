"""Backwards-compatible thin wrapper.

Stage 3a in the original notebook saved a `static_features.parquet` file
under outputs/features/. The reorganized pipeline computes static features
on-the-fly inside build_sequences. This module is kept so existing imports
keep working and to optionally persist the dataframe for inspection.
"""
from __future__ import annotations
from pathlib import Path

from .static_features import build_static_features


def run_build_static(cfg: dict, manifest_path: str | Path | None = None) -> dict:
    root = Path(cfg["paths"]["root"])
    out_dir = root / cfg["paths"]["outputs_dir"] / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_static_features(cfg)
    static_path = out_dir / "static_features.parquet"
    df.to_parquet(static_path)
    print(f"[build_static] saved: {static_path} | rows={len(df)}")
    return {"status": "ok", "static_features": str(static_path)}
