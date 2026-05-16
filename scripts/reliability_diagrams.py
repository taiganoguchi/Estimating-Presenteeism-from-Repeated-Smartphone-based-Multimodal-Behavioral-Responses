"""Verification C: reliability diagrams and per-class ECE (reviewer R2-6).

Computes:
  - Per-class reliability diagrams (10 bins)
  - Per-class ECE (Expected Calibration Error)
  - Brier score per class
  - Comparison: Full vs Text-only vs Audio+Text

Input:  revision/predictions_per_clip_extended.parquet
Output: revision/figures/calib_C_*.png
        revision/calibration_results.csv
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PARQUET  = Path("/workspace/revision/predictions_per_clip_extended.parquet")
FIG_DIR  = Path("/workspace/revision/figures")
FIG_DIR.mkdir(exist_ok=True)

CONFIGS = ["Full", "Text-only", "Audio+Text", "Face+Text",
           "Audio-only", "Face-only", "Audio+Face"]
CLASSES  = [0, 1, 2]
N_BINS   = 10

df = pd.read_parquet(PARQUET)
y  = df["true_label"].values
print(f"Loaded: {len(df)} clips  label dist: {dict(zip(*np.unique(y, return_counts=True)))}")


def compute_ece(y_true, prob, n_bins=10):
    """Per-class ECE (one-vs-rest)."""
    ece_per_class = []
    for cls in CLASSES:
        y_bin  = (y_true == cls).astype(float)
        p_bin  = prob[:, cls]
        bins   = np.linspace(0, 1, n_bins + 1)
        ece    = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p_bin >= lo) & (p_bin < hi)
            if mask.sum() == 0: continue
            acc  = y_bin[mask].mean()
            conf = p_bin[mask].mean()
            ece += abs(acc - conf) * mask.sum() / len(y_true)
        ece_per_class.append(ece)
    return ece_per_class  # [ECE_cls0, ECE_cls1, ECE_cls2]


def compute_brier(y_true, prob):
    """Per-class Brier score (one-vs-rest)."""
    brier = []
    for cls in CLASSES:
        y_bin = (y_true == cls).astype(float)
        brier.append(float(np.mean((prob[:, cls] - y_bin) ** 2)))
    return brier


def reliability_data(y_true, prob, cls, n_bins=10):
    """Returns (mean_conf, frac_pos, bin_count) for reliability diagram."""
    y_bin = (y_true == cls).astype(float)
    p_bin = prob[:, cls]
    bins  = np.linspace(0, 1, n_bins + 1)
    confs, fracs, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_bin >= lo) & (p_bin < hi)
        if mask.sum() == 0:
            confs.append((lo + hi) / 2)
            fracs.append(np.nan)
            counts.append(0)
        else:
            confs.append(p_bin[mask].mean())
            fracs.append(y_bin[mask].mean())
            counts.append(mask.sum())
    return np.array(confs), np.array(fracs), np.array(counts)


# ── compute all metrics ───────────────────────────────────────────────────────
records = []
for cfg in CONFIGS:
    prob = df[[f"{cfg}_prob_0", f"{cfg}_prob_1", f"{cfg}_prob_2"]].values
    ece  = compute_ece(y, prob)
    brier = compute_brier(y, prob)
    macro_ece = float(np.mean(ece))
    macro_brier = float(np.mean(brier))
    records.append({
        "config": cfg,
        "ECE_class0": ece[0], "ECE_class1": ece[1], "ECE_class2": ece[2],
        "ECE_macro": macro_ece,
        "Brier_class0": brier[0], "Brier_class1": brier[1], "Brier_class2": brier[2],
        "Brier_macro": macro_brier,
    })
    print(f"  {cfg:15s}: ECE class0={ece[0]:.3f} class1={ece[1]:.3f} class2={ece[2]:.3f}  macro={macro_ece:.3f}  Brier_macro={macro_brier:.3f}")

calib_df = pd.DataFrame(records)
calib_df.to_csv("/workspace/revision/calibration_results.csv", index=False)
print("Saved calibration_results.csv")


# ── Figure C1: reliability diagrams (3 key configs × 3 classes) ──────────────
focus_cfgs = ["Full", "Text-only", "Audio+Text"]
class_names = ["Class 0\n(healthy)", "Class 1\n(moderate)", "Class 2\n(unwell)"]
colors_cfg  = {"Full": "#E53935", "Text-only": "#1E88E5", "Audio+Text": "#43A047"}

fig, axes = plt.subplots(len(CLASSES), len(focus_cfgs), figsize=(12, 11),
                          sharex=True, sharey=True)
for col, cfg in enumerate(focus_cfgs):
    prob = df[[f"{cfg}_prob_0", f"{cfg}_prob_1", f"{cfg}_prob_2"]].values
    ece  = compute_ece(y, prob)
    for row, cls in enumerate(CLASSES):
        ax = axes[row, col]
        confs, fracs, counts = reliability_data(y, prob, cls)
        valid = ~np.isnan(fracs)
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect")
        ax.bar(confs[valid], fracs[valid], width=0.08, alpha=0.55,
               color=colors_cfg[cfg], align="center", label="_nolegend_")
        ax.plot(confs[valid], fracs[valid], "o-", color=colors_cfg[cfg], lw=2,
                markersize=4, label=cfg)
        ax.fill_between(confs[valid], fracs[valid], confs[valid],
                        alpha=0.12, color="red" if fracs[valid].mean() > confs[valid].mean() else "blue")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability" if row == 2 else "", fontsize=8)
        ax.set_ylabel(f"{class_names[cls]}\nFraction of positives" if col == 0 else "", fontsize=8)
        if row == 0:
            ax.set_title(f"{cfg}\n(ECE={ece[cls]:.3f})", fontsize=10)
        else:
            ax.set_title(f"ECE={ece[cls]:.3f}", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.92, f"ECE={ece[cls]:.3f}", transform=ax.transAxes,
                fontsize=8, color=colors_cfg[cfg], fontweight="bold")

fig.suptitle("Reliability diagrams (Full / Text-only / Audio+Text x 3 classes)\nDashed diagonal = perfect calibration", fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "calib_C1_reliability_diagrams.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved calib_C1_reliability_diagrams.png")


# ── Figure C2: ECE bar chart (all 7 configs × 3 classes) ─────────────────────
x = np.arange(len(CONFIGS))
width = 0.25
colors3 = ["#42A5F5", "#FF7043", "#66BB6A"]

fig, ax = plt.subplots(figsize=(11, 5))
for i, cls in enumerate(CLASSES):
    vals = [calib_df.loc[calib_df["config"] == cfg, f"ECE_class{cls}"].values[0]
            for cfg in CONFIGS]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=f"Class {cls}", color=colors3[i], alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(CONFIGS, rotation=30, ha="right", fontsize=10)
ax.set_ylabel("Expected Calibration Error (lower is better)", fontsize=10)
ax.set_title("Per-class ECE by configuration (lower is better)", fontsize=12)
ax.legend(title="Class", fontsize=9)
ax.axhline(0.05, color="gray", ls=":", lw=1, alpha=0.7)
ax.text(len(CONFIGS) - 0.5, 0.052, "ECE=0.05", fontsize=8, color="gray")
ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
plt.tight_layout()
plt.savefig(FIG_DIR / "calib_C2_ece_bars.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved calib_C2_ece_bars.png")


# ── Figure C3: Brier score bar chart ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
for i, cls in enumerate(CLASSES):
    vals = [calib_df.loc[calib_df["config"] == cfg, f"Brier_class{cls}"].values[0]
            for cfg in CONFIGS]
    ax.bar(x + (i - 1) * width, vals, width, label=f"Class {cls}", color=colors3[i], alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(CONFIGS, rotation=30, ha="right", fontsize=10)
ax.set_ylabel("Brier Score (lower is better)", fontsize=10)
ax.set_title("Per-class Brier score by configuration (lower is better)", fontsize=12)
ax.legend(title="Class", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "calib_C3_brier_bars.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved calib_C3_brier_bars.png")


# ── Figure C4: ECE heatmap (config × class) ──────────────────────────────────
ece_mat = np.array([
    [calib_df.loc[calib_df["config"] == cfg, f"ECE_class{cls}"].values[0]
     for cls in CLASSES]
    for cfg in CONFIGS
])

fig, ax = plt.subplots(figsize=(5, 7))
im = ax.imshow(ece_mat, cmap="Reds", aspect="auto", vmin=0, vmax=ece_mat.max()*1.1)
ax.set_xticks(range(3))
ax.set_xticklabels(["Class 0\n(healthy)", "Class 1\n(moderate)", "Class 2\n(unwell)"], fontsize=9)
ax.set_yticks(range(len(CONFIGS)))
ax.set_yticklabels(CONFIGS, fontsize=10)
for i, cfg in enumerate(CONFIGS):
    for j, cls in enumerate(CLASSES):
        v = ece_mat[i, j]
        worst = "*" if v == ece_mat[:, j].max() else ""
        ax.text(j, i, f"{v:.3f}{worst}", ha="center", va="center", fontsize=9,
                color="white" if v > ece_mat.max()*0.6 else "black")
fig.colorbar(im, ax=ax, label="ECE (lower is better)")
ax.set_title("Per-class ECE heatmap\n(* = max ECE within class)", fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / "calib_C4_ece_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved calib_C4_ece_heatmap.png")

print("\nDone.")
