"""op_paired_ttest_configs — Full vs 他 config の 25-fold paired t-test + Wilcoxon.

Group 4 operation: 「多角的評価」の構成要素。
run_multi_perspective.py から subprocess で呼ばれるが単独実行も可能。

Usage:
  python3 scripts/paired_ttest_configs.py --model M4_attmil_late_v3
  python3 scripts/paired_ttest_configs.py --model M4_attmil_late_v3_noLA
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SAVE_DIR = Path("/workspace/revision/analyses/bakeoff")
CONFIGS  = ["Full", "Audio+Text", "Face+Text", "Text-only",
            "Audio-only", "Face-only", "Audio+Face"]
METRICS  = ["f1_macro", "rec_class1", "rec_class2",
            "overpred_ratio_class2", "ece_overall", "mae_ordinal", "auroc_macro"]


def _fn_base(cfg: str) -> str:
    return cfg.replace("+", "_").replace(" ", "_")


def load_ckpt(model: str, cfg: str) -> pd.DataFrame | None:
    fb  = _fn_base(cfg)
    csv = SAVE_DIR / f"{model}_{fb}_ckpt.csv"
    if not csv.exists():
        csv = SAVE_DIR / f"{model}_{fb}_all25fold.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df = df.drop_duplicates(["repeat", "fold"], keep="last").sort_values(["repeat", "fold"]).reset_index(drop=True)
    # derive far_err and overpred2 from confusion matrix if not present
    if "far_err" not in df.columns and "cm_02" in df.columns:
        df["far_err"] = df["cm_02"] + df["cm_20"]
    if "overpred_ratio_class2" not in df.columns and "cm_02" in df.columns:
        pred2 = df["cm_02"] + df["cm_12"] + df["cm_22"]
        true2 = df["cm_20"] + df["cm_21"] + df["cm_22"]
        df["overpred_ratio_class2"] = pred2 / true2.replace(0, np.nan)
    return df


def sig_stars(p: float) -> str:
    if np.isnan(p): return ""
    return "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else ""))


def main(model: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs: dict[str, pd.DataFrame] = {}
    for cfg in CONFIGS:
        df = load_ckpt(model, cfg)
        if df is not None:
            dfs[cfg] = df

    if "Full" not in dfs:
        print(f"[op_paired_ttest] SKIP: Full config not found for {model}")
        return

    all_metrics = METRICS + (["far_err"] if "far_err" in dfs["Full"].columns else [])
    full = dfs["Full"]

    rows = []
    lines = [f"# Paired t-test: {model} — Full vs other configs\n",
             f"n_folds = {len(full)}\n\n"]

    for cfg in [c for c in CONFIGS if c != "Full"]:
        if cfg not in dfs:
            lines.append(f"## {cfg}: no data\n\n")
            continue
        other = dfs[cfg]
        n = min(len(full), len(other))
        lines.append(f"## Full vs {cfg}  (n={n} folds)\n\n")
        lines.append(f"| metric | Δ (Full−other) | t | p | sig | Wilcoxon p |\n")
        lines.append(f"|---|---|---|---|---|---|\n")

        for m in all_metrics:
            if m not in full.columns or m not in other.columns:
                continue
            a = full[m].values[:n]
            b = other[m].values[:n]
            d = float(a.mean() - b.mean())
            t_stat, p_t = stats.ttest_rel(a, b)
            try:
                _, p_w = stats.wilcoxon(a, b, zero_method="wilcox")
            except Exception:
                p_w = np.nan
            sig = sig_stars(p_t)
            lines.append(f"| {m} | {d:+.4f} | {t_stat:+.2f} | {p_t:.4f} | {sig} | {p_w:.4f} |\n")
            rows.append({
                "model": model, "config_vs": cfg, "metric": m,
                "delta_full_minus_other": d,
                "t_stat": float(t_stat), "p_ttest": float(p_t),
                "p_wilcoxon": float(p_w) if not np.isnan(p_w) else np.nan,
                "sig": sig, "n_folds": n,
            })
        lines.append("\n")

    # summary: count how many metrics Full wins significantly
    lines.append("## Summary: Full 有意優位 metric 数 (p < .05, paired-t)\n\n")
    lines.append("| vs | Full>other (sig) | Full<other (sig) |\n|---|---|---|\n")
    for cfg in [c for c in CONFIGS if c != "Full"]:
        sub = [r for r in rows if r["config_vs"] == cfg]
        n_pos = sum(1 for r in sub if r["delta_full_minus_other"] > 0 and r["p_ttest"] < .05)
        n_neg = sum(1 for r in sub if r["delta_full_minus_other"] < 0 and r["p_ttest"] < .05)
        lines.append(f"| {cfg} | {n_pos} | {n_neg} |\n")

    out_csv = out_dir / "paired_ttest_full_vs_others.csv"
    out_md  = out_dir / "paired_ttest_full_vs_others.md"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_md.write_text("".join(lines))
    print(f"[op_paired_ttest] Saved: {out_csv}")
    print(f"[op_paired_ttest] Saved: {out_md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out = Path(args.out_dir) if args.out_dir else SAVE_DIR / "multi_perspective" / args.model / "group4"
    main(args.model, out)
