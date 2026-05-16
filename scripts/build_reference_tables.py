"""Compute 4-config metrics with cluster bootstrap 95% CI for multimodal-shift revision."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

PARQUET = Path("/workspace/revision/analyses/bakeoff/M4_attmil_late_v4_per_clip.parquet")
OUT_DIR = Path("/workspace/revision")
OUT_DIR.mkdir(exist_ok=True)

CONFIGS_TEXT = ["Full", "Audio+Text", "Face+Text", "Text-only"]
CONFIGS_ALL = ["Full", "Audio+Text", "Face+Text", "Text-only",
               "Audio+Face", "Audio-only", "Face-only"]
CLASSES = [0, 1, 2]
CLASS_NAMES = ["healthy", "moderate", "unwell"]

df = pd.read_parquet(PARQUET)
print(f"Loaded: {len(df)} clips from {df['user_id'].nunique()} users")
print(f"Class dist: {dict(zip(*np.unique(df['true_label'], return_counts=True)))}")

# ── ECE ──────────────────────────────────────────────────────────────────────
def compute_ece(y_true, prob_max, pred, n_bins=10):
    """Overall ECE on winning-class probability."""
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob_max >= lo) & (prob_max < hi)
        if mask.sum() == 0: continue
        acc = correct[mask].mean()
        conf = prob_max[mask].mean()
        ece += abs(acc - conf) * mask.sum() / len(y_true)
    return ece

# ── cluster bootstrap helper ────────────────────────────────────────────────
def cluster_bootstrap_ci(df_sub, cfg, metric_func, n_boot=2000, seed=42):
    """Resample participants (clusters); recompute metric on resampled clips."""
    rng = np.random.default_rng(seed)
    users = df_sub["user_id"].unique()
    boot = []
    user_to_clips = {u: df_sub[df_sub["user_id"] == u].index.values for u in users}
    for _ in range(n_boot):
        sampled_users = rng.choice(users, size=len(users), replace=True)
        clip_idx = np.concatenate([user_to_clips[u] for u in sampled_users])
        sub = df_sub.loc[clip_idx]
        try:
            val = metric_func(sub, cfg)
            boot.append(val)
        except Exception:
            continue
    boot = np.array(boot)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), boot

# ── metric callables ────────────────────────────────────────────────────────
def m_macro_f1(d, cfg):
    return f1_score(d["true_label"], d[f"{cfg}_pred"], average="macro", zero_division=0)

def m_accuracy(d, cfg):
    return accuracy_score(d["true_label"], d[f"{cfg}_pred"])

def m_macro_precision(d, cfg):
    return precision_score(d["true_label"], d[f"{cfg}_pred"], average="macro", zero_division=0)

def m_macro_recall(d, cfg):
    return recall_score(d["true_label"], d[f"{cfg}_pred"], average="macro", zero_division=0)

def m_macro_auroc(d, cfg):
    prob = d[[f"{cfg}_prob_0", f"{cfg}_prob_1", f"{cfg}_prob_2"]].values
    return roc_auc_score(d["true_label"], prob, multi_class="ovr", average="macro")

def m_macro_auprc(d, cfg):
    prob = d[[f"{cfg}_prob_0", f"{cfg}_prob_1", f"{cfg}_prob_2"]].values
    auprc = []
    for c in CLASSES:
        y_bin = (d["true_label"] == c).astype(int)
        auprc.append(average_precision_score(y_bin, prob[:, c]))
    return np.mean(auprc)

def m_overall_ece(d, cfg):
    prob = d[[f"{cfg}_prob_0", f"{cfg}_prob_1", f"{cfg}_prob_2"]].values
    prob_max = prob.max(axis=1)
    pred = d[f"{cfg}_pred"].values
    return compute_ece(d["true_label"].values, prob_max, pred)

# ── (1) Overall metrics × 4 configs ─────────────────────────────────────────
print("\n=== Overall metrics × 4 configs ===")
overall_rows = []
for cfg in CONFIGS_ALL:
    point = {
        "config": cfg,
        "macro_F1": m_macro_f1(df, cfg),
        "accuracy": m_accuracy(df, cfg),
        "macro_precision": m_macro_precision(df, cfg),
        "macro_recall": m_macro_recall(df, cfg),
        "macro_AUROC": m_macro_auroc(df, cfg),
        "macro_AUPRC": m_macro_auprc(df, cfg),
        "overall_ECE": m_overall_ece(df, cfg),
    }
    # CIs only for 4 main configs (cost limit)
    if cfg in CONFIGS_TEXT:
        for metric_name, fn in [("macro_F1", m_macro_f1), ("accuracy", m_accuracy),
                                  ("macro_AUROC", m_macro_auroc), ("macro_AUPRC", m_macro_auprc)]:
            lo, hi, _ = cluster_bootstrap_ci(df, cfg, fn, n_boot=2000)
            point[f"{metric_name}_CI_lo"] = lo
            point[f"{metric_name}_CI_hi"] = hi
    overall_rows.append(point)
    print(f"  {cfg:15s}: F1={point['macro_F1']:.3f}  acc={point['accuracy']:.3f}  AUROC={point['macro_AUROC']:.3f}  AUPRC={point['macro_AUPRC']:.3f}  ECE={point['overall_ECE']:.3f}")

pd.DataFrame(overall_rows).to_csv(OUT_DIR / "v4_overall_metrics_7config.csv", index=False)

# ── (2) Per-class metrics × 4 configs ───────────────────────────────────────
print("\n=== Per-class F1 × 4 configs ===")
def per_class_metric(d, cfg, c, name):
    p = d[f"{cfg}_pred"].values; y = d["true_label"].values
    y_bin = (y == c).astype(int); p_bin = (p == c).astype(int)
    if name == "F1":
        return f1_score(y_bin, p_bin, zero_division=0)
    elif name == "precision":
        return precision_score(y_bin, p_bin, zero_division=0)
    elif name == "recall":
        return recall_score(y_bin, p_bin, zero_division=0)

per_class_rows = []
for cfg in CONFIGS_TEXT:
    for c, cname in zip(CLASSES, CLASS_NAMES):
        f1 = per_class_metric(df, cfg, c, "F1")
        prec = per_class_metric(df, cfg, c, "precision")
        rec = per_class_metric(df, cfg, c, "recall")
        # F1 cluster bootstrap CI
        def fn_f1(d, cfg=cfg, c=c): return per_class_metric(d, cfg, c, "F1")
        lo_f1, hi_f1, _ = cluster_bootstrap_ci(df, cfg, fn_f1, n_boot=2000)
        row = {"config": cfg, "class": c, "class_name": cname,
               "F1": f1, "F1_CI_lo": lo_f1, "F1_CI_hi": hi_f1,
               "precision": prec, "recall": rec,
               "support": int((df["true_label"] == c).sum())}
        per_class_rows.append(row)
        print(f"  {cfg:15s} {cname:10s} F1={f1:.3f} [{lo_f1:.3f}, {hi_f1:.3f}]  P={prec:.3f}  R={rec:.3f}")

pd.DataFrame(per_class_rows).to_csv(OUT_DIR / "v4_perclass_metrics_4config.csv", index=False)

# ── (3) Confusion matrices × 4 configs ──────────────────────────────────────
print("\n=== Confusion matrices × 4 configs ===")
cm_records = []
for cfg in CONFIGS_TEXT:
    cm = confusion_matrix(df["true_label"], df[f"{cfg}_pred"], labels=CLASSES)
    print(f"\n  {cfg}:")
    print(f"    {cm}")
    for i, ti in enumerate(CLASSES):
        for j, pj in enumerate(CLASSES):
            cm_records.append({"config": cfg, "true": ti, "pred": pj, "count": int(cm[i, j])})

pd.DataFrame(cm_records).to_csv(OUT_DIR / "v4_confusion_4config.csv", index=False)

# ── (4) Per-class ECE × 4 configs ───────────────────────────────────────────
print("\n=== Per-class ECE × 4 configs ===")
def compute_class_ece(y_true, prob, c, n_bins=10):
    y_bin = (y_true == c).astype(float)
    p_bin = prob[:, c]
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_bin >= lo) & (p_bin < hi)
        if mask.sum() == 0: continue
        ece += abs(y_bin[mask].mean() - p_bin[mask].mean()) * mask.sum() / len(y_true)
    return ece

calib_rows = []
for cfg in CONFIGS_TEXT:
    prob = df[[f"{cfg}_prob_0", f"{cfg}_prob_1", f"{cfg}_prob_2"]].values
    overall = compute_ece(df["true_label"].values, prob.max(axis=1), df[f"{cfg}_pred"].values)
    row = {"config": cfg, "overall_ECE": overall}
    for c, cname in zip(CLASSES, CLASS_NAMES):
        ece_c = compute_class_ece(df["true_label"].values, prob, c)
        row[f"ECE_{cname}"] = ece_c
    calib_rows.append(row)
    print(f"  {cfg:15s} overall={row['overall_ECE']:.3f}  healthy={row['ECE_healthy']:.3f} moderate={row['ECE_moderate']:.3f} unwell={row['ECE_unwell']:.3f}")

pd.DataFrame(calib_rows).to_csv(OUT_DIR / "v4_calibration_4config.csv", index=False)

# ── (5) User-level video count stats ────────────────────────────────────────
print("\n=== User-level engagement ===")
user_counts = df.groupby("user_id").size()
print(f"  n_users: {len(user_counts)}")
print(f"  clips/user: mean={user_counts.mean():.1f}  SD={user_counts.std():.1f}  median={user_counts.median():.0f}  min={user_counts.min()}  max={user_counts.max()}")
user_counts.to_csv(OUT_DIR / "v4_user_clip_counts.csv")

print("\nDone.")
