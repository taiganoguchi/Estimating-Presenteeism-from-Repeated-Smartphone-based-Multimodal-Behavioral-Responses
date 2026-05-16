"""Environment-driven path resolution for the PLOS_ONE_2025 pipeline.

Many scripts in this repository historically used hard-coded absolute paths
(``/workspace/...``) that match the Docker container's working directory.
This module centralises path resolution so the same code works inside the
container *and* on a host environment where the workspace lives at a
different root.

Usage
-----
```python
from src.paths import WORKSPACE, BAKEOFF_DIR, DATA_ROOT

per_clip_parquet = BAKEOFF_DIR / "M4_attmil_late_v4_per_clip.parquet"
```

Environment variables
---------------------
``WORKSPACE_ROOT``
    Absolute path to the repository root. Defaults to ``/workspace`` (the
    bind-mount inside the Docker container) but may be overridden for
    host-side execution, e.g. ``export WORKSPACE_ROOT=/path/to/repo``.

``DATA_ROOT``
    Absolute path to the dataset root, equivalent to
    ``${WORKSPACE_ROOT}/data/2024/cohort`` by default.

``BAKEOFF_DIR``
    Absolute path for the bake-off analysis directory, default
    ``${WORKSPACE_ROOT}/revision/analyses/bakeoff``.

All paths returned are ``pathlib.Path`` instances and are not created on
disk; callers are responsible for ensuring they exist.
"""
from __future__ import annotations

import os
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE_ROOT", "/workspace")).resolve()
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(WORKSPACE / "data" / "2024" / "cohort"))).resolve()
BAKEOFF_DIR = Path(os.environ.get("BAKEOFF_DIR", str(WORKSPACE / "revision" / "analyses" / "bakeoff"))).resolve()

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", str(WORKSPACE / "config.yaml"))).resolve()

SEQUENCES_DIR = DATA_ROOT / "outputs_clean_v7" / "sequences"
SEQ_INDEX_PARQUET = SEQUENCES_DIR / "seq_index.parquet"

IMAGE_OUTPUT_DIR = WORKSPACE / "revision" / "figures"

__all__ = [
    "WORKSPACE",
    "DATA_ROOT",
    "BAKEOFF_DIR",
    "CONFIG_PATH",
    "SEQUENCES_DIR",
    "SEQ_INDEX_PARQUET",
    "IMAGE_OUTPUT_DIR",
]
