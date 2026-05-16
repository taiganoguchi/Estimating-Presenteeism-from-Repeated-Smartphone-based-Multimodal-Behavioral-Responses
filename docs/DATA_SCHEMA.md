# Data Schema — PLOS_ONE_2025 v4 Pipeline

This document describes the dataset layout and column conventions expected
by the v4 training and evaluation pipeline. The raw data themselves are
**not** distributed with this repository owing to participant-privacy
restrictions; data access requests should be directed to the institutional
contact listed in the Data Availability Statement of the submitted
manuscript.

## Top-level layout

The path referenced by ``paths.root`` in ``config.yaml``
(``${WORKSPACE_ROOT}/data/2024/cohort`` by default) must contain:

```
cohort/
├── data_all/
│   ├── cohort-surveys.csv     # daily survey responses
│   ├── cohort-users.csv       # static demographic / role attributes
│   └── surveys/               # per-clip raw artefacts (one folder per user)
│       └── S?????/
│           ├── S?????_YYYYMMDDhhmmss_result.csv       # OpenFace AU/pose/gaze
│           ├── S?????_YYYYMMDDhhmmss_whisper.csv      # ASR transcription
│           └── S?????_YYYYMMDDhhmmss_parselmouth.csv  # Praat acoustics
├── results/
│   └── doctor_evaluation.xlsx # psychiatrist ratings spreadsheet
└── outputs_clean_v7/
    └── sequences/
        ├── seq_index.parquet  # per-clip metadata + labels (auto-generated)
        ├── seg_text/          # per-segment text features (auto-generated)
        └── splits_repeats_v4_ncd.json   # 25-fold splits (auto-generated)
```

The participant id ``S?????`` is a 5-digit zero-padded counter (e.g.,
``S00101``). The timestamp suffix ``YYYYMMDDhhmmss`` is parsed by
``_compute_dropout_users`` to enforce the ≥15-day recording-span criterion.

## Key columns in ``seq_index.parquet``

| Column | Type | Description |
| --- | --- | --- |
| ``user_id`` | str | Per-participant identifier (``S?????``). |
| ``video_id`` | str | ``{user_id}_{timestamp}``; primary key. |
| ``label`` | int (0–2) | Consensus ordinal label (0 = healthy, 1 = moderate, 2 = unwell). |
| ``label_consensus_123`` | int | Same as ``label`` but using the 1/2/3 input scale. |
| ``seq_path`` | str | Path to the per-clip sequence ``.npz``. |
| ``n_steps`` | int | Sequence length after 100 Hz → 50 ms window aggregation. |
| ``dim`` | int | Feature dimensionality per frame. |
| ``latency_q2a`` | float | Latency (s) between the prompt and the response onset. |
| ``cohort`` | str | Cohort stratification token. |
| ``label_dok`` / ``label_nis`` / ``label_srk`` | int | Per-rater labels for psychiatrists R1 / R2 / R3 (the column names retain anonymous three-letter identifiers for backwards compatibility with the upstream schema). |
| ``soft_p1`` / ``soft_p2`` / ``soft_p3`` | float | Soft-label probabilities (mean over the three raters). |
| ``n_raters_used`` | int | Number of raters with non-missing ratings. |
| ``sleep_bin`` / ``workh_bin`` / ``place_bin`` / ``state_bin`` | str | Coarse strata used for subgroup analyses. |

## Preprocessing chain (high level)

1. **Ingestion.** Surveys CSV / users CSV / per-clip artefact directories
   are scanned by ``src/pipeline.py``.
2. **Quality filtering.** Clips with OpenFace tracking-success ratio < 0.20
   or Parselmouth voiced-frame ratio < the configured floor are dropped.
3. **Time alignment.** Per-window blended normalisation; resample to
   100 Hz with 50 ms windows / 50 ms hop (20 Hz).
4. **Text features.** Whisper transcription, Sentence-BERT embedding (768-d).
5. **Label aggregation.** Majority vote over three psychiatrist raters;
   complete-dispersion clips (one rater per class) are dropped.
6. **Cohort filter.** Participants with a first-to-last recording span of
   fewer than ``MIN_RECORDING_DAYS`` days (default 15) are excluded
   as early dropouts.

## Reproducing the manuscript's numbers

Once ``seq_index.parquet`` and the per-clip NPZ predictions exist, run
the single command:

```bash
python main.py reproduce
```

See ``README.md`` "Quick start" for the full table of reproduction steps;
together they regenerate every numerical row of Tables 1, 2, 3, the
conditional-capture table, and the subgroup heatmap reported in the
manuscript.
