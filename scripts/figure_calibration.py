"""Generate 4-config × 3-class reliability diagram figure."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PARQUET = Path("/workspace/revision/analyses/bakeoff/M4_attmil_late_v4_per_clip.parquet")
OUT_PATH = Path("/workspace/revision/manuscript/2025 PLOS ONE/PLOS_ONE_2025/image_v1/fig_calibration.png")

CONFIGS = ["Text-only", "Face+Text", "Audio+Text", "Full"]
CLASSES = [0, 1, 2]
CLASS_NAMES = ["healthy", "moderate", "unwell"]
N_BINS = 10
COLOR = {"Text-only": "#1E88E5", "Face+Text": "#FB8C00",
         "Audio+Text": "#43A047", "Full": "#E53935"}

df = pd.read_parquet(PARQUET)
y = df["true_label"].values

def compute_ece(y_true, prob, cls, n_bins=10):
    y_bin = (y_true == cls).astype(float)
    p_bin = prob[:, cls]
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p_bin >= lo) & (p_bin < hi)
        if m.sum() == 0: continue
        ece += abs(y_bin[m].mean() - p_bin[m].mean()) * m.sum() / len(y_true)
    return ece

def reliability(y_true, prob, cls, n_bins=10):
    y_bin = (y_true == cls).astype(float)
    p_bin = prob[:, cls]
    bins = np.linspace(0, 1, n_bins + 1)
    confs, fracs, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p_bin >= lo) & (p_bin < hi)
        if m.sum() == 0:
            confs.append((lo + hi) / 2); fracs.append(np.nan); counts.append(0)
        else:
            confs.append(p_bin[m].mean()); fracs.append(y_bin[m].mean()); counts.append(m.sum())
    return np.array(confs), np.array(fracs), np.array(counts)

# 3 rows (classes) × 4 cols (configs)
fig, axes = plt.subplots(3, 4, figsize=(14, 11), sharex=True, sharey=True)

for col, cfg in enumerate(CONFIGS):
    prob = df[[f"{cfg}_prob_0", f"{cfg}_prob_1", f"{cfg}_prob_2"]].values
    for row, cls in enumerate(CLASSES):
        ax = axes[row, col]
        confs, fracs, counts = reliability(y, prob, cls)
        ece = compute_ece(y, prob, cls)
        valid = ~np.isnan(fracs)
        # diagonal
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        # bars (light)
        ax.bar(confs[valid], fracs[valid], width=0.08, alpha=0.5,
               color=COLOR[cfg], align="center")
        # line connector
        ax.plot(confs[valid], fracs[valid], "o-", color=COLOR[cfg], lw=2,
                markersize=4)
        ax.fill_between(confs[valid], fracs[valid], confs[valid],
                        alpha=0.15, color=COLOR[cfg])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        # ECE annotation
        ax.text(0.05, 0.92, f"ECE = {ece:.3f}", transform=ax.transAxes,
                fontsize=10, fontweight="bold", color=COLOR[cfg])
        if row == 0:
            ax.set_title(cfg, fontsize=12, fontweight="bold")
        if row == 2:
            ax.set_xlabel("Mean predicted probability", fontsize=10)
        if col == 0:
            ax.set_ylabel(f"Class {cls}\n({CLASS_NAMES[cls]})\nFraction of positives",
                          fontsize=10)

fig.suptitle("Per-class reliability diagrams: 4 text-containing configurations × 3 severity classes\n"
             "(OOF clip-level, $n_{\\mathrm{oof\\_clips}}=1{,}768$)",
             fontsize=12, y=0.998)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")

# Print summary
print("\nPer-class ECE × 4 configs:")
print(f"{'Config':<15} {'healthy':>10} {'moderate':>10} {'unwell':>10}")
for cfg in CONFIGS:
    prob = df[[f"{cfg}_prob_0", f"{cfg}_prob_1", f"{cfg}_prob_2"]].values
    eces = [compute_ece(y, prob, c) for c in CLASSES]
    print(f"{cfg:<15} {eces[0]:>10.3f} {eces[1]:>10.3f} {eces[2]:>10.3f}")
