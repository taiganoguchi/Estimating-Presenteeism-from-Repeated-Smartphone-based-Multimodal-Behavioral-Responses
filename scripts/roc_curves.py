"""Verification D: ROC curves for reviewer R2-5.

Generates:
  - Macro OVR ROC curves for all 7 configs (overlaid)
  - Per-class OVR ROC curves (3 panels)
  - DeLong test for AUROC pairwise comparisons

Input:  revision/predictions_per_clip_extended.parquet
Output: revision/figures/roc_D_*.png
        revision/roc_results.csv
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import stats

PARQUET  = Path("/workspace/revision/predictions_per_clip_extended.parquet")
FIG_DIR  = Path("/workspace/revision/figures")
FIG_DIR.mkdir(exist_ok=True)

CONFIGS = ["Full", "Text-only", "Audio+Text", "Face+Text",
           "Audio-only", "Face-only", "Audio+Face"]
CLASSES  = [0, 1, 2]
CLASS_NAMES = ["Class 0 (healthy)", "Class 1 (moderate)", "Class 2 (unwell)"]

# Style map: text-based configs are solid, audio/face-only are dashed
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

# ── DeLong test implementation ────────────────────────────────────────────────
# Reference: DeLong et al. 1988 "Comparing the areas under two or more correlated ROC curves"
# Approximate via bootstrap for multi-class case

def bootstrap_auroc_diff(y_true, prob_a, prob_b, n_boot=1000, seed=42):
    """Bootstrap paired AUROC difference test (two-sided)."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            a = roc_auc_score(y_true[idx], prob_a[idx], multi_class="ovr", average="macro")
            b = roc_auc_score(y_true[idx], prob_b[idx], multi_class="ovr", average="macro")
        except Exception:
            continue
        diffs.append(a - b)
    diffs = np.array(diffs)
    obs_diff = (roc_auc_score(y_true, prob_a, multi_class="ovr", average="macro") -
                roc_auc_score(y_true, prob_b, multi_class="ovr", average="macro"))
    # two-sided p-value from bootstrap distribution centered at 0
    shifted = diffs - diffs.mean()
    p_val = float(np.mean(np.abs(shifted) >= abs(obs_diff)))
    ci_lo, ci_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return obs_diff, ci_lo, ci_hi, p_val


# ── compute macro AUROC per config ────────────────────────────────────────────
auroc_per_cfg = {}
probs_per_cfg = {}
for cfg in CONFIGS:
    prob = df[[f"{cfg}_prob_0", f"{cfg}_prob_1", f"{cfg}_prob_2"]].values
    probs_per_cfg[cfg] = prob
    try:
        auroc_per_cfg[cfg] = roc_auc_score(y, prob, multi_class="ovr", average="macro")
    except Exception:
        auroc_per_cfg[cfg] = float("nan")
    print(f"  {cfg:15s}: AUROC={auroc_per_cfg[cfg]:.4f}")

# ── DeLong (bootstrap) Full vs others ─────────────────────────────────────────
print("\n=== Bootstrap AUROC test: Full vs others ===")
delong_records = []
prob_full = probs_per_cfg["Full"]
for cfg in CONFIGS:
    if cfg == "Full": continue
    diff, lo, hi, pval = bootstrap_auroc_diff(y, prob_full, probs_per_cfg[cfg])
    sig = "**" if pval < 0.01 else ("*" if pval < 0.05 else "n.s.")
    print(f"  Full vs {cfg:15s}: diff={diff:+.4f}  CI=[{lo:+.4f},{hi:+.4f}]  p={pval:.4f} {sig}")
    delong_records.append({"comparison": f"Full vs {cfg}", "auroc_diff": diff,
                            "ci_lo": lo, "ci_hi": hi, "p_bootstrap": pval})

pd.DataFrame(delong_records).to_csv("/workspace/revision/roc_results.csv", index=False)
print("Saved roc_results.csv")


# ── Figure D1: macro OVR ROC overlaid ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

for cfg in CONFIGS:
    prob = probs_per_cfg[cfg]
    ls, color, lw = STYLE_MAP[cfg]
    # macro OVR: average per-class ROC
    fpr_all, tpr_all = [], []
    for cls in CLASSES:
        y_bin = (y == cls).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, prob[:, cls])
        fpr_all.append(fpr); tpr_all.append(tpr)
    # interpolate to common FPR grid
    mean_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.mean([np.interp(mean_fpr, f, t) for f, t in zip(fpr_all, tpr_all)], axis=0)
    auroc = auroc_per_cfg[cfg]
    ax.plot(mean_fpr, mean_tpr, ls=ls, color=color, lw=lw,
            label=f"{cfg} (AUC={auroc:.4f})")

ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("Macro one-vs-rest ROC curves (7 configurations)\nSolid = text-containing; dashed = no-text", fontsize=12)
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "roc_D1_macro_ovr.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved roc_D1_macro_ovr.png")


# ── Figure D2: per-class ROC (3 panels, key configs) ─────────────────────────
focus_cfgs = ["Full", "Text-only", "Audio+Text", "Audio+Face"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
for ax, cls in zip(axes, CLASSES):
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    y_bin = (y == cls).astype(int)
    for cfg in focus_cfgs:
        ls, color, lw = STYLE_MAP[cfg]
        prob = probs_per_cfg[cfg]
        fpr, tpr, _ = roc_curve(y_bin, prob[:, cls])
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, ls=ls, color=color, lw=lw,
                label=f"{cfg} (AUC={auc_val:.3f})")
    ax.set_title(CLASS_NAMES[cls], fontsize=11)
    ax.set_xlabel("FPR", fontsize=9)
    ax.set_ylabel("TPR" if cls == 0 else "", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.grid(True, alpha=0.3)

fig.suptitle("Per-class ROC (one-vs-rest): Full / Text-only / Audio+Text / Audio+Face", fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "roc_D2_perclass.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved roc_D2_perclass.png")


# ── Figure D3: AUROC bar chart with bootstrap CI ─────────────────────────────
# compute bootstrap CI for each config
ci_records = []
for cfg in CONFIGS:
    prob = probs_per_cfg[cfg]
    rng  = np.random.default_rng(42)
    boot = []
    for _ in range(500):
        idx = rng.integers(0, len(y), len(y))
        try:
            boot.append(roc_auc_score(y[idx], prob[idx], multi_class="ovr", average="macro"))
        except Exception:
            pass
    boot = np.array(boot)
    ci_records.append({"config": cfg, "auroc": auroc_per_cfg[cfg],
                        "ci_lo": np.percentile(boot, 2.5),
                        "ci_hi": np.percentile(boot, 97.5)})

ci_df = pd.DataFrame(ci_records)

fig, ax = plt.subplots(figsize=(10, 5))
colors_bar = [STYLE_MAP[cfg][1] for cfg in CONFIGS]
bars = ax.bar(range(len(CONFIGS)), ci_df["auroc"], color=colors_bar, alpha=0.85, edgecolor="none")
bars[0].set_edgecolor("black"); bars[0].set_linewidth(2)  # highlight Full
yerr_lo = (ci_df["auroc"] - ci_df["ci_lo"]).values
yerr_hi = (ci_df["ci_hi"] - ci_df["auroc"]).values
ax.errorbar(range(len(CONFIGS)), ci_df["auroc"],
            yerr=[yerr_lo, yerr_hi], fmt="none", color="black", capsize=5, lw=1.5)
ax.set_xticks(range(len(CONFIGS)))
ax.set_xticklabels(CONFIGS, rotation=30, ha="right", fontsize=10)
ax.set_ylabel("AUROC (macro OVR)", fontsize=11)
ax.set_title("AUROC across 7 configurations with 95\\% bootstrap CI\n(Full highlighted with black outline)", fontsize=12)
ax.set_ylim(0.5, 1.0)
ax.axhline(0.5, color="gray", ls="--", lw=0.8)
plt.tight_layout()
plt.savefig(FIG_DIR / "roc_D3_auroc_bars.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved roc_D3_auroc_bars.png")

print("\nDone.")
