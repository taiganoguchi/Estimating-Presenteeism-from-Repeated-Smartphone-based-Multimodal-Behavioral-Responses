"""CLI entry point — runs the full pipeline end-to-end.

Usage:
    python -m src.pipeline --config /path/to/config.yaml
    python -m src.pipeline --config config.yaml --stage import
    python -m src.pipeline --config config.yaml --stage train,evaluate

Stages (in order):
    import      -> index_metadata.parquet
    preprocess  -> sample_manifest.parquet
    static      -> features/static_features.parquet (optional)
    sequences   -> sequences/*.npz + seq_index.parquet
    splits      -> sequences/split.json + split.holdout.json
    optuna      -> sequences/best_params.json
    train       -> sequences/best_model.pt
    evaluate    -> reports/metrics.json + classification_report.txt
"""
from __future__ import annotations
import argparse
from pathlib import Path

from .config import load_and_prepare
from .utils import seed_everything, write_run_meta
from .logging_setup import setup_logging, get_logger
from .data_import import run_data_import
from .data_preprocess import run_preprocess
from .build_static_features import run_build_static
from .build_sequences import run_build_sequences
from .splits import run_splits
from .model_train_eval import run_optuna, run_train
from .evaluate import run_evaluate


STAGES_ALL = [
    "import", "preprocess", "static", "sequences",
    "splits", "optuna", "train", "evaluate",
]


def _parse_stages(arg: str) -> list[str]:
    if arg in (None, "", "all"):
        return STAGES_ALL
    return [s.strip() for s in arg.split(",") if s.strip()]


def main():
    ap = argparse.ArgumentParser(description="PLOS_ONE_2025 pipeline")
    ap.add_argument("--config", required=True, help="path to config.yaml")
    ap.add_argument("--stage", default="all", help="comma-separated stages or 'all'")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    log = get_logger(__name__)

    cfg, root, out = load_and_prepare(args.config)
    seed_everything(int(cfg.get("runtime", {}).get("seed", 42)))
    write_run_meta(out, cfg)
    log.info("ROOT=%s  OUT=%s", root, out)

    stages = _parse_stages(args.stage)
    log.info("stages: %s", stages)

    if "import" in stages:
        run_data_import(cfg)
    if "preprocess" in stages:
        man = (out / cfg["paths"]["manifest_filename"])
        run_preprocess(cfg, out / "index_metadata.parquet")
    if "static" in stages and cfg.get("features", {}).get("enable", False):
        run_build_static(cfg)
    if "sequences" in stages:
        run_build_sequences(cfg)
    if "splits" in stages:
        run_splits(cfg)
    if "optuna" in stages and cfg.get("optuna", {}).get("enable", False):
        run_optuna(cfg)
    if "train" in stages:
        run_train(cfg)
    if "evaluate" in stages:
        run_evaluate(cfg)

    log.info("done.")


if __name__ == "__main__":
    main()
