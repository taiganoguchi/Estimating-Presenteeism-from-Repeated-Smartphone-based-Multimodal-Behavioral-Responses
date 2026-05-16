"""PLOS ONE 2025 — presenteeism estimation pipeline.

Modules ported from pipeline.ipynb to make the workflow runnable as a CLI:

    python -m src.pipeline --config config.yaml [--stage all|...]

The notebook (pipeline.ipynb) remains the primary interactive
analysis surface; src/ provides the importable, reproducible core.
"""
