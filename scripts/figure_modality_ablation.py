"""Modality ablation figure for multimodal-shift revision.

Design philosophy:
  - The "proposed" hypothesis is "multimodal fusion helps", not "Full is best"
  - 4 text-containing configs treated as a coherent multimodal family
  - 3 non-text configs serve as control (text-is-necessary)
  - Reference for Δ: Text-only (the natural baseline within text-containing family)
  - No unique "Full = proposed" highlight
"""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def holm_correct(p_list):
    p = np.array(p_list, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.zeros(n)
    for i, idx in enumerate(order):
        adj[idx] = min(1.0, (n - i) * p[idx])
    # enforce monotonicity in sorted order
    for i in range(1, n):
        if adj[order[i]] < adj[order[i-1]]:
            adj[order[i]] = adj[order[i-1]]
    return adj

BAKEOFF = "/workspace/revision/analyses/bakeoff/"
OUTFILE = "/workspace/revision/manuscript/2025 PLOS ONE/PLOS_ONE_2025/image_v1/fig_modality_ablation.png"

# Order: text-containing family first, then non-text control
configs_order = ["Text-only", "Face_Text", "Audio_Text", "Full",
                 "Audio_Face", "Audio-only", "Face-only"]
labels = ["Text-only", "Face\n+Text", "Audio\n+Text", "Full",
          "Audio\n+Face", "Audio\nonly", "Face\nonly"]
file_map = {
    "Text-only":  "M4_attmil_late_v4_Text-only_ckpt.csv",
    "Face_Text":  "M4_attmil_late_v4_Face_Text_ckpt.csv",
    "Audio_Text": "M4_attmil_late_v4_Audio_Text_ckpt.csv",
    "Full":       "M4_attmil_late_v4_Full_ckpt.csv",
    "Audio_Face": "M4_attmil_late_v4_Audio_Face_ckpt.csv",
    "Audio-only": "M4_attmil_late_v4_Audio-only_ckpt.csv",
    "Face-only":  "M4_attmil_late_v4_Face-only_ckpt.csv",
}

# Color: gradient within text-containing family + flat gray for non-text
TXT_PALETTE = {
    "Text-only":  "#1E88E5",   # blue
    "Face_Text":  "#FB8C00",   # orange
    "Audio_Text": "#43A047",   # green
    "Full":       "#E53935",   # red
}
NON_TXT = "#9E9E9E"  # gray (control)
colors = {c: TXT_PALETTE.get(c, NON_TXT) for c in configs_order}

# ── Load data ──────────────────────────────────────────────────────────────
f1_data = {}
for cfg in configs_order:
    df = pd.read_csv(BAKEOFF + file_map[cfg])
    assert len(df) == 25, f"Expected 25 folds for {cfg}, got {len(df)}"
    f1_data[cfg] = df["f1_macro"].values

# Reference = Text-only
ref = f1_data["Text-only"]
ref_mean = ref.mean()

# ── Statistics: Δ vs Text-only, Holm-corrected ─────────────────────────────
means, sds, deltas, raw_p = [], [], [], []
for cfg in configs_order:
    v = f1_data[cfg]
    means.append(v.mean())
    sds.append(v.std())
    deltas.append(v.mean() - ref_mean)
    if cfg == "Text-only":
        raw_p.append(None)
    else:
        _, p = stats.ttest_rel(v, ref)
        raw_p.append(p)

# Holm-Bonferroni over the 6 non-reference contrasts
non_ref_p = [p for p in raw_p if p is not None]
p_adj = holm_correct(non_ref_p)
holm = {cfg: p_adj[i] for i, cfg in enumerate([c for c in configs_order if c != "Text-only"])}
holm["Text-only"] = None

def stars(pa):
    if pa is None: return ""
    if pa < 0.001: return "***"
    if pa < 0.01: return "**"
    if pa < 0.05: return "*"
    return "n.s."

print("=== Modality ablation (Text-only baseline, fold-level paired t, Holm-corrected) ===")
print(f"{'Config':<12} {'mean':>8} {'sd':>8} {'Δ vs Text':>10} {'p_raw':>10} {'p_Holm':>10} {'sig'}")
for cfg, m, s, d, p in zip(configs_order, means, sds, deltas, raw_p):
    pa = holm[cfg]
    p_str = f"{p:.4f}" if p is not None else "ref"
    pa_str = f"{pa:.4f}" if pa is not None else "ref"
    print(f"{cfg:<12} {m:>8.4f} {s:>8.4f} {d:>+10.4f} {p_str:>10} {pa_str:>10} {stars(pa)}")

# ── Plot ───────────────────────────────────────────────────────────────────
bar_colors = [colors[c] for c in configs_order]
x = np.arange(len(configs_order))

fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(8.5, 6.0),
                                  gridspec_kw={"hspace": 0.5})

# Vertical separator between text-containing (first 4) and non-text (last 3)
SEPARATOR_X = 3.5

# ── Panel A: absolute Macro-F1 ─────────────────────────────────────────────
bars_a = ax_a.bar(x, means, width=0.6, color=bar_colors,
                  edgecolor="black", linewidth=0.6, zorder=3)
ax_a.errorbar(x, means, yerr=sds, fmt="none", ecolor="black",
              elinewidth=1.0, capsize=4, zorder=4)
ax_a.axhline(ref_mean, color="#1E88E5", linestyle="--",
             linewidth=1.0, alpha=0.6, zorder=2,
             label=f"Text-only mean = {ref_mean:.3f}")
ax_a.axvline(SEPARATOR_X, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
# Group annotations
ax_a.text(1.5, 0.93, "Text-containing (multimodal family)",
          ha="center", fontsize=9, fontweight="bold", color="#0072B2",
          transform=ax_a.get_xaxis_transform())
ax_a.text(5.0, 0.93, "Text-free (control)",
          ha="center", fontsize=9, fontweight="bold", color="#9E9E9E",
          transform=ax_a.get_xaxis_transform())

ax_a.set_ylim(0.0, 1.0)
ax_a.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax_a.set_ylabel("Macro-F1 (mean ± SD, 25 folds)", fontsize=10)
ax_a.set_xticks(x)
ax_a.set_xticklabels(labels, fontsize=9)
ax_a.legend(fontsize=8, loc="lower right")
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)
ax_a.yaxis.grid(True, alpha=0.3, zorder=0)
ax_a.set_axisbelow(True)
ax_a.text(-0.06, 1.04, "(A)", transform=ax_a.transAxes,
          fontsize=11, fontweight="bold", va="top")

# ── Panel B: Δ vs Text-only ────────────────────────────────────────────────
delta_sds = [np.std(f1_data[cfg] - ref) for cfg in configs_order]

bars_b = ax_b.bar(x, deltas, width=0.6, color=bar_colors,
                  edgecolor="black", linewidth=0.6, zorder=3)
ax_b.errorbar(x, deltas, yerr=delta_sds, fmt="none", ecolor="black",
              elinewidth=1.0, capsize=4, zorder=4)
ax_b.axhline(0, color="black", linewidth=1.0, zorder=2)
ax_b.axvline(SEPARATOR_X, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
ax_b.set_ylim(-0.45, 0.10)
ax_b.set_yticks([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1])
ax_b.set_ylabel(r"$\Delta$ Macro-F1 (vs Text-only)", fontsize=10)
ax_b.set_xticks(x)
ax_b.set_xticklabels(labels, fontsize=9)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
ax_b.yaxis.grid(True, alpha=0.3, zorder=0)
ax_b.set_axisbelow(True)
ax_b.text(-0.06, 1.04, "(B)", transform=ax_b.transAxes,
          fontsize=11, fontweight="bold", va="top")

# Significance markers
for i, cfg in enumerate(configs_order):
    pa = holm[cfg]
    if pa is None: continue
    s = stars(pa)
    if not s or s == "n.s.": continue
    if deltas[i] >= 0:
        y_pos = deltas[i] + delta_sds[i] + 0.01
    else:
        y_pos = deltas[i] - delta_sds[i] - 0.025
    ax_b.text(x[i], y_pos, s, ha="center", va="bottom",
              fontsize=10, fontweight="bold")

plt.savefig(OUTFILE, dpi=200, bbox_inches="tight")
plt.close()
print(f"\nFigure saved: {OUTFILE}")
