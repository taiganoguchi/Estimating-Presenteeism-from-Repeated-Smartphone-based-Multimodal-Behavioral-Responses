"""Compute subgroup metrics (speech tertile × Healthy/non-Healthy) × 4 configs."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from scipy import stats

PARQUET = Path("/workspace/revision/analyses/bakeoff/M4_attmil_late_v4_per_clip.parquet")
OUT_DIR = Path("/workspace/revision")

CONFIGS = ["Full", "Audio+Text", "Face+Text", "Text-only"]
df = pd.read_parquet(PARQUET)
print(f"Loaded: {len(df)} clips")

# ── Speech-duration tertiles ────────────────────────────────────────────────
print(f"\nspeech_duration distribution: {df['speech_duration'].describe()}")
q1, q2 = df["speech_duration"].quantile([1/3, 2/3]).values
print(f"\nTertile cutoffs: short<={q1:.3f}, mid={q1:.3f}-{q2:.3f}, long>{q2:.3f}")

def tertile(x):
    if x <= q1: return "Short"
    elif x <= q2: return "Mid"
    else: return "Long"

df["speech_tertile"] = df["speech_duration"].apply(tertile)
df["label_group"] = df["true_label"].apply(lambda y: "Healthy" if y == 0 else "non-Healthy")

# ── Subgroup × 4-config matrix ──────────────────────────────────────────────
print("\n=== Subgroup macro-F1 matrix (speech_tertile × label_group × config) ===")
records = []
for tert in ["Short", "Mid", "Long"]:
    for lbl in ["Healthy", "non-Healthy"]:
        sub = df[(df["speech_tertile"] == tert) & (df["label_group"] == lbl)]
        n = len(sub)
        if n < 5:
            print(f"  [{tert} × {lbl}] n={n} too small, skipped")
            continue
        for cfg in CONFIGS:
            f1 = f1_score(sub["true_label"], sub[f"{cfg}_pred"],
                          average="macro", zero_division=0)
            records.append({"speech_tertile": tert, "label_group": lbl,
                            "config": cfg, "n_clips": n, "macro_F1": f1})

sub_df = pd.DataFrame(records)
print(sub_df.to_string())

# Pivot for heatmap input
pivot = sub_df.pivot_table(index="config", columns=["label_group", "speech_tertile"],
                            values="macro_F1")
print("\nPivot (heatmap-ready):")
print(pivot.round(3))
sub_df.to_csv(OUT_DIR / "v4_subgroup_4config_heatmap.csv", index=False)
pivot.to_csv(OUT_DIR / "v4_subgroup_4config_pivot.csv")

# ── Critical subgroup: Short × non-Healthy ────────────────────────────────
print("\n=== CRITICAL: Short × non-Healthy ===")
crit = df[(df["speech_tertile"] == "Short") & (df["label_group"] == "non-Healthy")]
print(f"  n_clips = {len(crit)}")
print(f"  Label distribution: {dict(zip(*np.unique(crit['true_label'], return_counts=True)))}")
print(f"  User count: {crit['user_id'].nunique()}")

# Fold-level paired test for Audio+Text vs Text-only on this subgroup
# (we don't have fold labels in parquet, so use clip-level cluster-bootstrap diff)
print("\n  Per-config macro-F1 on critical subgroup:")
crit_records = []
for cfg in CONFIGS:
    f1 = f1_score(crit["true_label"], crit[f"{cfg}_pred"],
                  average="macro", zero_division=0)
    print(f"    {cfg:15s}: macro-F1 = {f1:.3f}")
    crit_records.append({"config": cfg, "n": len(crit), "macro_F1": f1})

# Cluster-bootstrap Δ (Audio+Text vs Text-only) on critical subgroup
rng = np.random.default_rng(42)
users = crit["user_id"].unique()
user_to_clips = {u: crit[crit["user_id"] == u].index.values for u in users}
deltas_at, deltas_ft, deltas_full = [], [], []
for _ in range(2000):
    sampled = rng.choice(users, size=len(users), replace=True)
    idx = np.concatenate([user_to_clips[u] for u in sampled])
    sub = crit.loc[idx]
    if len(np.unique(sub["true_label"])) < 2: continue
    try:
        f1_at = f1_score(sub["true_label"], sub["Audio+Text_pred"], average="macro", zero_division=0)
        f1_ft = f1_score(sub["true_label"], sub["Face+Text_pred"], average="macro", zero_division=0)
        f1_full = f1_score(sub["true_label"], sub["Full_pred"], average="macro", zero_division=0)
        f1_to = f1_score(sub["true_label"], sub["Text-only_pred"], average="macro", zero_division=0)
        deltas_at.append(f1_at - f1_to)
        deltas_ft.append(f1_ft - f1_to)
        deltas_full.append(f1_full - f1_to)
    except Exception:
        continue

for name, d in [("Audio+Text vs Text-only", deltas_at),
                ("Face+Text vs Text-only", deltas_ft),
                ("Full vs Text-only", deltas_full)]:
    d = np.array(d)
    obs = d.mean()
    lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
    # p-value: fraction of bootstrap distribution at or below 0
    p = float((d <= 0).mean() * 2)  # two-sided
    p = min(p, 1.0)
    print(f"    {name}: mean Δ={obs:+.3f} CI[{lo:+.3f},{hi:+.3f}] p_boot={p:.3f}")

# ── Also: Healthy macro-F1 vs other groups for narrative ────────────────────
print("\n=== Macro-F1 by subgroup (all 4 configs side-by-side) ===")
for tert in ["Short", "Mid", "Long"]:
    for lbl in ["Healthy", "non-Healthy"]:
        sub = df[(df["speech_tertile"] == tert) & (df["label_group"] == lbl)]
        n = len(sub)
        if n < 5: continue
        line = f"  {tert:5s} × {lbl:12s} (n={n:4d}):"
        for cfg in CONFIGS:
            f1 = f1_score(sub["true_label"], sub[f"{cfg}_pred"],
                          average="macro", zero_division=0)
            line += f"  {cfg.split('+')[0][:4] if '+' in cfg else cfg[:4]}={f1:.3f}"
        print(line)

print("\nDone.")
