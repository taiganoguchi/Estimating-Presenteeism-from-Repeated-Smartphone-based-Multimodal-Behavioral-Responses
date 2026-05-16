"""Conditional capture OOF simulation.

Tests deployment policies that combine Text-only (default) with multimodal
(triggered) and reports overall macro-F1 + activation rate + bootstrap CI.

Policies:
  P1 (speech-gated): if speech_duration < D, use Audio+Text; else Text-only.
  P2 (confidence-gated): if max Text-only prob < C, use Audio+Text; else Text-only.
  P3 (joint-gated): if speech_duration < D OR max Text-only prob < C, use Audio+Text.

Baselines: Text-only-everywhere, Full-everywhere (multimodal-always).
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

PARQUET = Path("/workspace/revision/analyses/bakeoff/M4_attmil_late_v4_per_clip.parquet")
df = pd.read_parquet(PARQUET)
N = len(df)
y = df["true_label"].values
print(f"Loaded: {N} clips, {df['user_id'].nunique()} users")

# Text-only confidence = max softmax probability
text_prob = df[["Text-only_prob_0", "Text-only_prob_1", "Text-only_prob_2"]].values
text_conf = text_prob.max(axis=1)
text_pred = df["Text-only_pred"].values
at_pred   = df["Audio+Text_pred"].values
full_pred = df["Full_pred"].values

def macro_f1(yt, yp):
    return f1_score(yt, yp, average="macro", zero_division=0)

# ── Cluster bootstrap helper ────────────────────────────────────────────────
def cluster_bootstrap(yt, yp, user_ids, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    users = np.unique(user_ids)
    u_to_idx = {u: np.where(user_ids == u)[0] for u in users}
    boot = []
    for _ in range(n_boot):
        s = rng.choice(users, size=len(users), replace=True)
        idx = np.concatenate([u_to_idx[u] for u in s])
        try:
            boot.append(macro_f1(yt[idx], yp[idx]))
        except Exception:
            continue
    boot = np.array(boot)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

users = df["user_id"].values

# ── Baselines ──────────────────────────────────────────────────────────────
print("\n=== Baselines ===")
print(f"{'Policy':<40} {'macro-F1':>10} {'95% CI':>20} {'Multimodal %':>14}")
for name, pred in [("Text-only everywhere", text_pred),
                    ("Audio+Text everywhere", at_pred),
                    ("Full everywhere", full_pred)]:
    f1 = macro_f1(y, pred)
    lo, hi = cluster_bootstrap(y, pred, users)
    print(f"{name:<40} {f1:>10.4f} [{lo:.4f},{hi:.4f}]   {'100.0%' if 'everywhere' in name and 'Text-only' not in name else '0.0%' if 'Text-only' in name else '100.0%':>14}")

# ── P1: speech-gated ───────────────────────────────────────────────────────
print("\n=== P1: speech-gated (use AT if speech_duration < D, else Text-only) ===")
print(f"{'D (s)':>8} {'Trigger %':>12} {'macro-F1':>10} {'95% CI':>20}")
for D in [0.5, 0.7, 0.89, 1.0, 1.5, 2.0, 3.0]:
    mask = df["speech_duration"].values < D
    pred = np.where(mask, at_pred, text_pred)
    f1 = macro_f1(y, pred)
    lo, hi = cluster_bootstrap(y, pred, users)
    print(f"{D:>8.2f} {100*mask.mean():>11.1f}% {f1:>10.4f} [{lo:.4f},{hi:.4f}]")

# ── P2: confidence-gated ───────────────────────────────────────────────────
print("\n=== P2: confidence-gated (use AT if max Text-only prob < C, else Text-only) ===")
print(f"{'C':>8} {'Trigger %':>12} {'macro-F1':>10} {'95% CI':>20}")
for C in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
    mask = text_conf < C
    pred = np.where(mask, at_pred, text_pred)
    f1 = macro_f1(y, pred)
    lo, hi = cluster_bootstrap(y, pred, users)
    print(f"{C:>8.2f} {100*mask.mean():>11.1f}% {f1:>10.4f} [{lo:.4f},{hi:.4f}]")

# ── P3: joint-gated ────────────────────────────────────────────────────────
print("\n=== P3: joint-gated (use AT if speech_duration < D OR max Text-only prob < C) ===")
print(f"{'D':>6} {'C':>6} {'Trigger %':>12} {'macro-F1':>10} {'95% CI':>20}")
best = (None, -1, None)
for D in [0.89, 1.0]:
    for C in [0.60, 0.70, 0.80]:
        mask = (df["speech_duration"].values < D) | (text_conf < C)
        pred = np.where(mask, at_pred, text_pred)
        f1 = macro_f1(y, pred)
        lo, hi = cluster_bootstrap(y, pred, users)
        print(f"{D:>6.2f} {C:>6.2f} {100*mask.mean():>11.1f}% {f1:>10.4f} [{lo:.4f},{hi:.4f}]")
        if f1 > best[1]:
            best = ((D, C), f1, (lo, hi))

print(f"\nBest joint policy: D={best[0][0]}, C={best[0][1]}, F1={best[1]:.4f} CI={best[2]}")

# ── Comparison vs baselines: focus on D=0.89 (matches subgroup analysis) ───
print("\n=== Headline comparison: speech-gated policy at D=0.89 vs baselines ===")
print(f"{'Policy':<45} {'macro-F1':>10} {'95% CI':>22} {'Cost (multimodal %)':>22}")
# Text-only
f1 = macro_f1(y, text_pred); lo, hi = cluster_bootstrap(y, text_pred, users)
print(f"{'Text-only everywhere (cheapest)':<45} {f1:>10.4f} [{lo:.4f}, {hi:.4f}]   {'0.0%':>22}")
# Conditional D=0.89
D = 0.89
mask = df["speech_duration"].values < D
pred = np.where(mask, at_pred, text_pred)
f1 = macro_f1(y, pred); lo, hi = cluster_bootstrap(y, pred, users)
print(f"{'Conditional capture (D=0.89s → A+T, else Text)':<45} {f1:>10.4f} [{lo:.4f}, {hi:.4f}]   {100*mask.mean():>21.1f}%")
# Audio+Text always
f1 = macro_f1(y, at_pred); lo, hi = cluster_bootstrap(y, at_pred, users)
print(f"{'Audio+Text everywhere':<45} {f1:>10.4f} [{lo:.4f}, {hi:.4f}]   {'100.0%':>22}")
# Full always
f1 = macro_f1(y, full_pred); lo, hi = cluster_bootstrap(y, full_pred, users)
print(f"{'Full everywhere (most expensive)':<45} {f1:>10.4f} [{lo:.4f}, {hi:.4f}]   {'100.0%':>22}")

# ── Per-class breakdown for conditional vs text-only ───────────────────────
print("\n=== Per-class F1: Conditional (D=0.89) vs Text-only ===")
mask = df["speech_duration"].values < 0.89
pred_cond = np.where(mask, at_pred, text_pred)
for c, name in [(0, "healthy"), (1, "moderate"), (2, "unwell")]:
    f1_text = f1_score((y == c).astype(int), (text_pred == c).astype(int), zero_division=0)
    f1_cond = f1_score((y == c).astype(int), (pred_cond == c).astype(int), zero_division=0)
    print(f"  Class {c} ({name:8s}): Text-only={f1_text:.4f}  Conditional={f1_cond:.4f}  Δ={f1_cond - f1_text:+.4f}")

# ── Focus on subgroup performance under conditional policy ─────────────────
print("\n=== Subgroup (short × non-Healthy, n=77) F1 under different policies ===")
sub_mask = (df["speech_duration"].values < 0.89) & (y != 0)
n_sub = sub_mask.sum()
print(f"  Subgroup n = {n_sub}")
for name, pred in [("Text-only", text_pred), ("Audio+Text", at_pred),
                    ("Conditional (D=0.89)", np.where(df["speech_duration"].values < 0.89, at_pred, text_pred))]:
    f1 = f1_score(y[sub_mask], pred[sub_mask], labels=[1, 2], average="macro", zero_division=0)
    print(f"  {name:<25}: macro-F1 = {f1:.4f}")

print("\nDone.")
