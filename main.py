"""Reproduce the published numbers and figures for PONE-D-26-07436.

This is the single entry point that reproduces every numerical result and
figure reported in the manuscript:

    "Estimating Presenteeism from Repeated Smartphone-based Multimodal
     Behavioral Responses" (PLOS ONE, under revision).

Usage
-----
Run the full reproduction pipeline (Tables 1-3, conditional capture table,
subgroup heatmap, calibration figure, and the proposed-method diagram):

    python main.py reproduce

Train the model from scratch for every modality configuration (slow):

    python main.py train --all-configs

Train just one configuration (e.g. when reproducing a specific row):

    python main.py train --config Full

Re-run only one step of the reproduction pipeline:

    python main.py reproduce --only ablation_figure

Skip ahead to step N (useful for resuming after a transient failure):

    python main.py reproduce --from-step 4

The pipeline expects the canonical dataset to be available at the path
configured by ``DATA_ROOT`` (default ``${WORKSPACE_ROOT}/data/2024/cohort``).
See ``docs/DATA_SCHEMA.md`` for the expected directory layout.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# ── Reproduction pipeline ─────────────────────────────────────────────────
# Each entry is (label, script_path_relative_to_repo, extra_args).
# Labels are used for --only filtering and for log output.

REPRODUCE_STEPS = [
    ("01_build_per_clip_predictions",
     "scripts/build_per_clip_predictions.py",
     []),
    ("02a_overall_metrics",
     "scripts/compute_overall_metrics.py",
     []),
    ("02b_reference_tables",
     "scripts/build_reference_tables.py",
     []),
    ("03a_paired_ttest",
     "scripts/paired_ttest_configs.py",
     []),
    ("03b_modality_ablation_figure",
     "scripts/figure_modality_ablation.py",
     []),
    ("04a_subgroup_metrics",
     "scripts/compute_subgroup_metrics.py",
     []),
    ("04b_subgroup_heatmap",
     "scripts/figure_subgroup_heatmap.py",
     []),
    ("05_conditional_capture",
     "scripts/simulate_conditional_capture.py",
     []),
    ("06_calibration_figure",
     "scripts/figure_calibration.py",
     []),
    ("07_proposed_method_figure",
     "scripts/figure_proposed_method.py",
     []),
]

MODALITY_CONFIGS = [
    "Full", "Audio+Text", "Face+Text", "Text-only",
    "Audio-only", "Face-only", "Audio+Face",
]


def _run(label: str, cmd: list[str], env: dict[str, str] | None = None) -> int:
    print(f"\n[{label}] $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        print(f"[{label}] FAILED (exit {result.returncode})", file=sys.stderr,
              flush=True)
    else:
        print(f"[{label}] OK", flush=True)
    return result.returncode


def cmd_reproduce(args: argparse.Namespace) -> int:
    """Run the reproduction steps in order."""
    for i, (label, script, extra) in enumerate(REPRODUCE_STEPS, start=1):
        if i < args.from_step:
            print(f"[{label}] skipped (--from-step={args.from_step})", flush=True)
            continue
        if args.only and args.only not in label:
            continue
        if args.dry_run:
            print(f"[{label}] would run: {script} {' '.join(extra)}")
            continue
        cmd = [sys.executable, str(REPO_ROOT / script), *extra]
        if _run(label, cmd) != 0:
            return 1
    print("\nAll reproduction steps completed.", flush=True)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Launch training for one or all modality configurations."""
    configs = MODALITY_CONFIGS if args.all_configs else [args.config]
    if not configs or configs == [None]:
        print("ERROR: pass --config <name> or --all-configs", file=sys.stderr)
        return 1

    script = REPO_ROOT / "scripts" / "train.py"
    for cfg in configs:
        if cfg not in MODALITY_CONFIGS:
            print(f"ERROR: unknown config '{cfg}'. Choose from {MODALITY_CONFIGS}",
                  file=sys.stderr)
            return 1
        env = os.environ.copy()
        env["CONFIG"] = cfg
        env.setdefault("MOD_DROP", "0.3")
        cmd = [sys.executable, str(script)]
        if args.dry_run:
            print(f"[train:{cfg}] would run: CONFIG={cfg} MOD_DROP={env['MOD_DROP']} {script}")
            continue
        if _run(f"train:{cfg}", cmd, env=env) != 0:
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduction and training entry point for PONE-D-26-07436.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rp = subparsers.add_parser(
        "reproduce",
        help="Run the post-training reproduction pipeline (default).")
    rp.add_argument("--from-step", type=int, default=1, metavar="N",
                    help="Start from step N (1-based; see --help for the order).")
    rp.add_argument("--only", type=str, default=None, metavar="LABEL",
                    help="Run only the step whose label contains LABEL.")
    rp.add_argument("--dry-run", action="store_true",
                    help="Print the commands that would run, then exit.")

    tr = subparsers.add_parser(
        "train",
        help="Train the three-stream Attention-MIL model from scratch.")
    g = tr.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", type=str, default=None,
                   help="Train a single modality configuration "
                        f"(choices: {MODALITY_CONFIGS}).")
    g.add_argument("--all-configs", action="store_true",
                   help="Train every modality configuration sequentially.")
    tr.add_argument("--dry-run", action="store_true",
                    help="Print the commands that would run, then exit.")

    args = parser.parse_args(argv)
    if args.command == "reproduce":
        return cmd_reproduce(args)
    if args.command == "train":
        return cmd_train(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
