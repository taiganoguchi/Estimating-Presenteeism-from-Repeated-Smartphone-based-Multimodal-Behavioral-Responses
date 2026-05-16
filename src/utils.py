"""Common utilities: seeding, device, JSON dump, run metadata."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import os, json, random, socket, hashlib, platform
import numpy as np


def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        try:
            import torch.backends.cudnn as cudnn
            cudnn.deterministic = True
            cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        pass


def device_auto() -> str:
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def json_dump(obj, path) -> None:
    """numpy 対応の安全な JSON ダンプ。"""
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return str(o)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_default)


def write_run_meta(out_dir: Path, cfg: dict, extra: dict | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_json = out_dir / "config.snapshot.json"
    try:
        h = hashlib.sha256(cfg_json.read_bytes()).hexdigest()[:12]
    except Exception:
        h = None
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        cuda_ver = torch.version.cuda if hasattr(torch, "version") else None
        cudnn_ok = bool(getattr(torch.backends, "cudnn", None) and torch.backends.cudnn.enabled)
        gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda_ok else []
        torch_ver = torch.__version__
    except Exception:
        cuda_ok, cuda_ver, cudnn_ok, gpus, torch_ver = None, None, None, [], None
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "platform": platform.platform(),
        "device": device_auto(),
        "seed": int(cfg.get("runtime", {}).get("seed", 42)),
        "cfg_hash": h,
        "torch": torch_ver,
        "cuda_available": cuda_ok,
        "cuda_version": cuda_ver,
        "cudnn_enabled": cudnn_ok,
        "gpus": gpus,
    }
    if extra:
        meta.update(extra)
    path = out_dir / "run_meta.json"
    json_dump(meta, path)
    return path


def resolve_paths(cfg: dict) -> tuple[Path, Path]:
    """Return (ROOT, OUT) — ROOT is data root, OUT is the outputs dir."""
    root = Path(cfg["paths"]["root"]).resolve()
    od = str(cfg["paths"]["outputs_dir"]).lstrip("/")
    cfg["paths"]["outputs_dir"] = od
    out = root / od
    out.mkdir(parents=True, exist_ok=True)
    return root, out
