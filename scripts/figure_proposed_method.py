"""Simplified flow diagram: focus on overall pipeline, no equations or details."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings; warnings.filterwarnings("ignore")

W, H = 16.5, 5.5
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
fig.patch.set_facecolor("white")

BK = "black"; GR = "#777777"; DARK = "#1f1f1f"
COL_FACE  = "#FB8C00"
COL_VOICE = "#43A047"
COL_TEXT  = "#1E88E5"
COL_FUSE  = "#E53935"

def rect(ax, x, y, w, h, label, fs=11, lw=1.4, ls="-", fc="white", ec=BK, txt=BK):
    box = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle="square,pad=0", facecolor=fc, edgecolor=ec,
        linewidth=lw, linestyle=ls, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=txt, zorder=4)

def parallelogram(ax, x, y, w, h, label, fs=11, lw=1.4, fc="white", ec=BK, txt=BK):
    skew = 0.18
    pts = [(x-w/2+skew, y-h/2), (x+w/2, y-h/2),
            (x+w/2-skew, y+h/2), (x-w/2, y+h/2)]
    pg = mpatches.Polygon(pts, closed=True, facecolor=fc, edgecolor=ec,
                          linewidth=lw, zorder=3)
    ax.add_patch(pg)
    ax.text(x, y, label, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=txt, zorder=4)

def arrow(ax, x1, y1, x2, y2, col=BK, lw=1.5, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=col, lw=lw, linestyle=ls),
        zorder=2)

def line(ax, x1, y1, x2, y2, col=BK, lw=1.5, ls="-"):
    ax.plot([x1, x2], [y1, y2], color=col, lw=lw, linestyle=ls, zorder=2)

# Stream y-positions
YF, YV, YT = 4.2, 3.0, 1.8
YM = 3.0

# ── Title ────────────────────────────────────────────────────
ax.text(W/2, H-0.3,
    "Proposed multimodal model: 3-stream Attention-MIL with late fusion",
    ha="center", fontsize=12.5, fontweight="bold", color=DARK)

# ── 1. Input: Video clip ────────────────────────────────────
X0 = 1.2
parallelogram(ax, X0, YM, 1.6, 0.65, "Video clip", fs=11)
arrow(ax, X0+0.85, YM, X0+1.55, YM)

# ── 2. Segmentation ─────────────────────────────────────────
X1 = 3.0
rect(ax, X1, YM, 1.7, 0.65, "Segmentation", fs=11)

# branch to 3 streams
arrow(ax, X1+0.85, YM, X1+1.45, YM)
X_BUS1 = 4.65
line(ax, X_BUS1, YT, X_BUS1, YF)
ax.plot(X_BUS1, YM, "ko", markersize=4, zorder=5)

# ── 3. Feature extraction (3 streams) ────────────────────────
X2 = 6.3
streams = [
    (YF, "Face features",  COL_FACE),
    (YV, "Voice features", COL_VOICE),
    (YT, "Text features",  COL_TEXT),
]
for yi, lbl, col in streams:
    line(ax, X_BUS1, yi, X2-0.95, yi)
    arrow(ax, X2-0.95, yi, X2-0.8, yi)
    parallelogram(ax, X2, yi, 1.7, 0.55, lbl, fs=10, ec=col, txt=col)

# ── 4. Per-stream Attention-MIL encoders ─────────────────────
X3 = 8.6
for yi, lbl, col in [
    (YF, "Face Attention-MIL",  COL_FACE),
    (YV, "Voice Attention-MIL", COL_VOICE),
    (YT, "Text Attention-MIL",  COL_TEXT),
]:
    arrow(ax, X2+0.85, yi, X3-1.0, yi)
    rect(ax, X3, yi, 2.0, 0.55, lbl, fs=10, ec=col, txt=col)

# Modality dropout annotation (light, no formula) — placed BELOW the dashed box
dp = mpatches.FancyBboxPatch((X3-1.05, YV-0.32), 2.1, YF-YV+0.64,
    boxstyle="square,pad=0", facecolor="none",
    edgecolor=GR, linewidth=0.9, linestyle="--", zorder=2)
ax.add_patch(dp)
ax.text(X3, YV-0.55, "Modality dropout (train only)",
        ha="center", va="center", fontsize=8.5, color=GR, style="italic")

# ── 5. Late fusion + LA ──────────────────────────────────────
X4 = 11.5
X_BUS2 = 10.2
line(ax, X_BUS2, YT, X_BUS2, YF)
ax.plot(X_BUS2, YM, "ko", markersize=4, zorder=5)
for yi in [YF, YV, YT]:
    arrow(ax, X3+1.0, yi, X_BUS2, yi)
arrow(ax, X_BUS2, YM, X4-1.0, YM)
rect(ax, X4, YM, 2.0, 0.65, "Late fusion", fs=11, ec=COL_FUSE, txt=COL_FUSE)

# ── 6. CORN ordinal decode ───────────────────────────────────
X5 = 13.7
arrow(ax, X4+1.0, YM, X5-0.95, YM)
rect(ax, X5, YM, 1.9, 0.65, "CORN decode", fs=11, ec=COL_FUSE, txt=COL_FUSE)

# ── 7. Output ────────────────────────────────────────────────
X6 = 15.5
arrow(ax, X5+0.95, YM, X6-0.72, YM)
parallelogram(ax, X6, YM, 1.5, 0.65, "Output", fs=11, ec=COL_FUSE, txt=COL_FUSE)

plt.tight_layout(pad=0.3)
out = "/workspace/revision/manuscript/2025 PLOS ONE/PLOS_ONE_2025/image_v1/fig_proposed_methods.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("saved:", out)
