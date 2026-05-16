"""pr_curves.py — Generate per-class and macro PR curves to replace
mis-named fig_pr_ovr.png (which previously held ROC content)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score

PARQUET = Path("/workspace/revision/analyses/bakeoff/M4_attmil_late_v4_per_clip.parquet")
FIG_DIR = Path("/workspace/revision/analyses/bakeoff/multi_perspective/M4_attmil_late_v4/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ["Full", "Text-only", "Audio+Text", "Face+Text",
           "Audio-only", "Face-only", "Audio+Face"]
CLASSES = [0, 1, 2]
CLASS_NAMES = ["Class 0 (healthy)", "Class 1 (moderate)", "Class 2 (unwell)"]

STYLE_MAP = {
    "Full":       ("-",  "#E53935", 2.5),
    "Text-only":  ("-",  "#1E88E5", 2.0),
    "Audio+Text": ("-",  "#43A047", 2.0),
    "Face+Text":  ("-",  "#FB8C00", 1.8),
    "Audio-only": ("--", "#8E24AA", 1.5),
    "Face-only":  ("--", "#00ACC1", 1.5),
    "Audio+Face": ("--", "#6D4C41", 1.5),
}

df = pd.read_parquet(PARQUET)
y  = df["true_label"].values
print(f"Loaded: {len(df)} clips")

probs_per_cfg = {cfg: df[[f"{cfg}_prob_0", f"{cfg}_prob_1", f"{cfg}_prob_2"]].values for cfg in CONFIGS}

# ── Per-class PR (large, 3 panels, key configs only) ──────────────────────────
focus_cfgs = ["Full", "Text-only", "Audio+Text", "Audio+Face"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # larger panels
for ax, cls in zip(axes, CLASSES):
    y_bin = (y == cls).astype(int)
    baseline = y_bin.mean()
    ax.axhline(baseline, color="k", lw=1, ls=":", alpha=0.5,
               label=f"chance = {baseline:.2f}")
    for cfg in focus_cfgs:
        ls, color, lw = STYLE_MAP[cfg]
        prob = probs_per_cfg[cfg]
        precision, recall, _ = precision_recall_curve(y_bin, prob[:, cls])
        ap = average_precision_score(y_bin, prob[:, cls])
        ax.plot(recall, precision, ls=ls, color=color, lw=lw,
                label=f"{cfg} (AP={ap:.3f})")
    ax.set_title(CLASS_NAMES[cls], fontsize=12)
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision" if cls == 0 else "", fontsize=10)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.grid(True, alpha=0.3)

fig.suptitle("Per-class Precision-Recall curves (one-vs-rest)", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "pr_E1_perclass.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved pr_E1_perclass.png")
