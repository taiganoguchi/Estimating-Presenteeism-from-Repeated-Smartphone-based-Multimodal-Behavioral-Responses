"""Generate subgroup heatmap figure: 4-config × (speech_tertile × Healthy/non-Healthy).

Cutoffs:
  Short = speech_duration < 0.89s  (matches published n=77 in non-Healthy)
  Mid   = 0.89 ≤ speech_duration < 2.67s
  Long  = speech_duration ≥ 2.67s
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score

PARQUET = Path("/workspace/revision/analyses/bakeoff/M4_attmil_late_v4_per_clip.parquet")
OUT_DIR = Path("/workspace/revision/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ["Text-only", "Face+Text", "Audio+Text", "Full"]
df = pd.read_parquet(PARQUET)

# Tertile-like split (anchored at 0.89s for Short)
def make_tertile(x):
    if x < 0.89: return "Short"
    elif x < 2.67: return "Mid"
    else: return "Long"
df["speech_tertile"] = df["speech_duration"].apply(make_tertile)
df["label_group"] = (df["true_label"] != 0).map({True: "non-Healthy", False: "Healthy"})

# Compute macro-F1 for each cell
print("Subgroup macro-F1 matrix:")
rows = []
for lbl in ["Healthy", "non-Healthy"]:
    for tert in ["Short", "Mid", "Long"]:
        sub = df[(df["label_group"] == lbl) & (df["speech_tertile"] == tert)]
        n = len(sub)
        # For Healthy: only class 0 — macro-F1 not informative, use accuracy instead
        # For non-Healthy: classes 1 and 2 — macro-F1 over these 2 classes
        if lbl == "Healthy":
            metric_label = "Accuracy"
            metric_values = {cfg: (sub[f"{cfg}_pred"] == 0).mean() for cfg in CONFIGS}
        else:
            metric_label = "macro-F1"
            metric_values = {cfg: f1_score(sub["true_label"], sub[f"{cfg}_pred"],
                                            labels=[1, 2], average="macro", zero_division=0)
                              for cfg in CONFIGS}
        print(f"  {lbl:12s} × {tert:5s} (n={n:4d}, {metric_label}): "
              + "  ".join(f"{c}={metric_values[c]:.3f}" for c in CONFIGS))
        for cfg in CONFIGS:
            rows.append({"speech_tertile": tert, "label_group": lbl, "config": cfg,
                         "n_clips": n, "metric": metric_values[cfg],
                         "metric_label": metric_label})

sub_df = pd.DataFrame(rows)
sub_df.to_csv("/workspace/revision/v4_subgroup_4config_heatmap_v2.csv", index=False)

# ── Heatmap figure ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"wspace": 0.45})

groups = [("Healthy", "Accuracy"), ("non-Healthy", "macro-F1")]
tertiles = ["Short", "Mid", "Long"]

for ax, (lbl, metric_label) in zip(axes, groups):
    mat = np.zeros((len(CONFIGS), len(tertiles)))
    n_cells = np.zeros((len(CONFIGS), len(tertiles)), dtype=int)
    for i, cfg in enumerate(CONFIGS):
        for j, tert in enumerate(tertiles):
            row = sub_df[(sub_df["config"] == cfg) & (sub_df["label_group"] == lbl) & (sub_df["speech_tertile"] == tert)]
            mat[i, j] = row["metric"].values[0]
            n_cells[i, j] = row["n_clips"].values[0]

    vmin = mat.min() - 0.02
    vmax = mat.max() + 0.02
    im = ax.imshow(mat, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(tertiles)))
    ax.set_xticklabels([f"{t}\n(n={n_cells[0, j]})" for j, t in enumerate(tertiles)], fontsize=10)
    ax.set_yticks(range(len(CONFIGS)))
    ax.set_yticklabels(CONFIGS, fontsize=10)
    ax.set_title(f"{lbl} ({metric_label})", fontsize=12, fontweight="bold")

    # Annotate cells
    for i in range(len(CONFIGS)):
        for j in range(len(tertiles)):
            val = mat[i, j]
            txt_color = "white" if val < (vmin + vmax) / 2 else "black"
            # Highlight Short × non-Healthy Audio+Text cell
            is_critical = (lbl == "non-Healthy" and tertiles[j] == "Short" and CONFIGS[i] == "Audio+Text")
            if is_critical:
                ax.add_patch(plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                                            fill=False, edgecolor="red", lw=2.5))
                ax.text(j, i, f"{val:.3f}*", ha="center", va="center",
                        fontsize=11, fontweight="bold", color="red")
            else:
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, color=txt_color)

    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label=metric_label)
    ax.set_xlabel("Speech duration", fontsize=10)

fig.suptitle("Performance by speech duration × clinical state (4 text-containing configurations, OOF clip-level)\n"
             "Red box: Audio+Text outperforms Text-only in the diagnostically hardest subgroup "
             "(Short × non-Healthy, n=77; Δ=+0.030 fold-level, p=0.032 uncorrected, exploratory)",
             fontsize=11, y=1.06)
plt.tight_layout()

out_path = OUT_DIR / "fig_subgroup_heatmap.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")

# Also copy to image_v1
import shutil
image_v1 = Path("/workspace/revision/manuscript/2025 PLOS ONE/PLOS_ONE_2025/image_v1/fig_subgroup_heatmap.png")
shutil.copy(out_path, image_v1)
print(f"Copied to: {image_v1}")
