"""Configuration loading + snapshotting.

Mirrors pipeline.ipynb cells 2-3: load YAML, ensure paths/outputs are valid,
synchronize tokenize<-sequence, and save snapshot files alongside outputs.
"""
from __future__ import annotations
from pathlib import Path
import os, json, copy, yaml

from .utils import resolve_paths


def load_cfg(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def setup_runtime_env() -> None:
    """Apply env vars used by the notebook (PYTHONHASHSEED etc.)."""
    os.environ.setdefault("PYTHONHASHSEED", "42")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def finalize_cfg(cfg: dict) -> dict:
    """Sync tokenize <- sequence keys, normalize outputs_dir, etc."""
    if "tokenize" in cfg and "sequence" in cfg:
        for k in ("base_hz", "window_ms", "hop_ms", "max_seq_len"):
            if k in cfg["sequence"]:
                cfg["tokenize"][k] = cfg["sequence"][k]
    od = str(cfg["paths"]["outputs_dir"])
    if od.startswith("/"):
        cfg["paths"]["outputs_dir"] = od.lstrip("/")
    return cfg


def snapshot_cfg(cfg: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.snapshot.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (out_dir / "config.snapshot.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_and_prepare(path: str | Path) -> tuple[dict, Path, Path]:
    """One-shot helper used by pipeline.py: load YAML, finalize, snapshot."""
    setup_runtime_env()
    cfg = load_cfg(path)
    cfg = finalize_cfg(cfg)
    root, out = resolve_paths(cfg)
    snapshot_cfg(cfg, out)
    return cfg, root, out
