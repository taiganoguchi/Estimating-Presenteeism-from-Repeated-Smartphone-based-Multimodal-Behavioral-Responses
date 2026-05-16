"""Compute OOF aggregate stats for M4_attmil_late_v4 and save to JSON.

v4 = v3 + early-dropout exclusion (recording span ≤14 days, 5 users removed)
1,781 → 1,768 clips, 33 → 29 users.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    f1_score, confusion_matrix, roc_auc_score,
    average_precision_score, precision_score, recall_score
)

PARQUET = Path('/workspace/revision/analyses/bakeoff/M4_attmil_late_v4_per_clip.parquet')
OUTJSON = Path('/workspace/revision/analyses/bakeoff/v4_final_numbers.json')

df = pd.read_parquet(PARQUET)
y_true = df['true_label'].values
y_pred = df['Full_pred'].values
prob_cols = ['Full_prob_0', 'Full_prob_1', 'Full_prob_2']
y_prob = df[prob_cols].values
n = len(df)
print(f"n_clips = {n}")
print(f"Label dist: {dict(zip(*np.unique(y_true, return_counts=True)))}")

nums = {}
nums['n_clips'] = n

nums['macro_f1']   = float(f1_score(y_true, y_pred, average='macro'))
nums['micro_f1']   = float(f1_score(y_true, y_pred, average='micro'))
nums['prec_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
nums['rec_macro']  = float(recall_score(y_true, y_pred, average='macro', zero_division=0))

print(f"\n=== OOF Aggregate Classification (v4) ===")
print(f"Macro-F1:        {nums['macro_f1']:.4f}")
print(f"Micro-F1 (Acc):  {nums['micro_f1']:.4f}")
print(f"Macro Precision: {nums['prec_macro']:.4f}")
print(f"Macro Recall:    {nums['rec_macro']:.4f}")

nums['per_class'] = {}
for c in range(3):
    f1c = float(f1_score(y_true, y_pred, labels=[c], average=None, zero_division=0)[0])
    pc  = float(precision_score(y_true, y_pred, labels=[c], average=None, zero_division=0)[0])
    rc  = float(recall_score(y_true, y_pred, labels=[c], average=None, zero_division=0)[0])
    sup = int((y_true == c).sum())
    nums['per_class'][str(c)] = {'f1': f1c, 'prec': pc, 'rec': rc, 'support': sup}
    print(f"  Class {c}: P={pc:.4f}  R={rc:.4f}  F1={f1c:.4f}  n={sup}")

cm = confusion_matrix(y_true, y_pred)
print(f"\nConfusion matrix (rows=true, cols=pred):\n{cm}")
nums['confusion_matrix'] = cm.tolist()

nums['auroc_macro'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
print(f"\nMacro-AUROC (OOF): {nums['auroc_macro']:.4f}")

nums['auroc_per_class'] = {}
for c in range(3):
    y_bin = (y_true == c).astype(int)
    auc_c = float(roc_auc_score(y_bin, y_prob[:, c]))
    nums['auroc_per_class'][str(c)] = auc_c
    print(f"  AUROC class {c}: {auc_c:.4f}")

auprc_vals = []
nums['auprc_per_class'] = {}
for c in range(3):
    y_bin = (y_true == c).astype(int)
    ap = float(average_precision_score(y_bin, y_prob[:, c]))
    auprc_vals.append(ap)
    nums['auprc_per_class'][str(c)] = ap
    print(f"  AUPRC class {c}: {ap:.4f}")
nums['auprc_macro'] = float(np.mean(auprc_vals))
print(f"Macro-AUPRC (OOF): {nums['auprc_macro']:.4f}")

nums['mae_ordinal'] = float(np.mean(np.abs(y_pred - y_true)))
print(f"\nMAE (ordinal): {nums['mae_ordinal']:.4f}")

def ece(y_true_bin, y_prob_1d, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob_1d >= lo) & (y_prob_1d < hi)
        if mask.sum() == 0: continue
        acc  = y_true_bin[mask].mean()
        conf = y_prob_1d[mask].mean()
        ece_val += mask.sum() * abs(acc - conf)
    return float(ece_val / len(y_true_bin))

conf_max = y_prob.max(axis=1)
correct   = (y_pred == y_true)
nums['ece_overall'] = ece(correct.astype(int), conf_max)
print(f"\nECE (overall): {nums['ece_overall']:.4f}")

nums['ece_per_class'] = {}
nums['brier_per_class'] = {}
for c in range(3):
    y_bin  = (y_true == c).astype(int)
    ec = ece(y_bin, y_prob[:, c])
    br = float(np.mean((y_prob[:, c] - y_bin) ** 2))
    nums['ece_per_class'][str(c)] = ec
    nums['brier_per_class'][str(c)] = br
    print(f"  Class {c}: ECE={ec:.4f}  Brier={br:.4f}")

if 'Text-only_pred' in df.columns:
    text_pred = df['Text-only_pred'].values
    agree    = int((y_pred == text_pred).sum())
    disagree = int((y_pred != text_pred).sum())
    nums['text_agree']        = agree
    nums['text_disagree']     = disagree
    nums['text_agree_pct']    = round(agree / n * 100, 1)
    nums['text_disagree_pct'] = round(disagree / n * 100, 1)
    print(f"\nText-only vs Full agree: {agree} ({nums['text_agree_pct']}%)")
    print(f"Text-only vs Full disagree: {disagree} ({nums['text_disagree_pct']}%)")

print("\n=== Bootstrap CIs (B=2000, participant cluster) ===")
rng = np.random.default_rng(42)
participants = df['user_id'].unique()
B = 2000

boot = {k: [] for k in ['f1', 'micro', 'auroc', 'prec', 'rec', 'auprc']}
boot_pc = {str(c): [] for c in range(3)}

for _ in range(B):
    sampled = rng.choice(participants, size=len(participants), replace=True)
    mask = df['user_id'].isin(sampled)
    yt   = y_true[mask]; yp = y_pred[mask]; ypr = y_prob[mask]
    if len(np.unique(yt)) < 3:
        continue
    boot['f1'].append(f1_score(yt, yp, average='macro', zero_division=0))
    boot['micro'].append(f1_score(yt, yp, average='micro', zero_division=0))
    boot['auroc'].append(roc_auc_score(yt, ypr, multi_class='ovr', average='macro'))
    boot['prec'].append(precision_score(yt, yp, average='macro', zero_division=0))
    boot['rec'].append(recall_score(yt, yp, average='macro', zero_division=0))
    ap_vals = []
    for c in range(3):
        y_bin = (yt == c).astype(int)
        if y_bin.sum() > 0:
            ap_vals.append(average_precision_score(y_bin, ypr[:, c]))
    if ap_vals:
        boot['auprc'].append(float(np.mean(ap_vals)))
    for c in range(3):
        f1c = f1_score(yt, yp, labels=[c], average=None, zero_division=0)
        boot_pc[str(c)].append(float(f1c[0]))

def ci95(arr):
    a = np.array(arr)
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

nums['ci'] = {}
for k, arr in boot.items():
    lo, hi = ci95(arr)
    nums['ci'][k] = [lo, hi]
    print(f"  {k:8s} 95% CI: [{lo:.3f}, {hi:.3f}]")

nums['ci_per_class_f1'] = {}
for c in range(3):
    lo, hi = ci95(boot_pc[str(c)])
    nums['ci_per_class_f1'][str(c)] = [lo, hi]
    print(f"  Class {c} F1 95% CI: [{lo:.3f}, {hi:.3f}]")

OUTJSON.write_text(json.dumps(nums, indent=2))
print(f"\nSaved → {OUTJSON}")
