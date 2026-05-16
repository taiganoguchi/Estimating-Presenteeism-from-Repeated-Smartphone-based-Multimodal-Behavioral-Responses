# Estimating Presenteeism from Repeated Smartphone-based Multimodal Behavioral Responses

Source code accompanying **PONE-D-26-07436** (PLOS ONE, under revision).

The pipeline reproduces the evaluation reported in the manuscript: 1,768
out-of-fold clips from 29 participants under 25-fold repeated
participant-disjoint cross-validation (5 × 5-fold GroupKFold), with a
three-stream Attention-MIL model, a CORN ordinal head, logit adjustment,
and late logit-level fusion of face / voice / text streams.

## Quick start

Install the dependencies into a Python 3.11+ environment and reproduce
every numerical result and figure in the manuscript with a single command:

```bash
pip install -r requirements.txt
python main.py reproduce
```

Re-train the model from scratch (requires GPU; expects about six hours per
configuration on a single A100):

```bash
python main.py train --config Full           # one configuration
python main.py train --all-configs           # all seven configurations
```

Additional `main.py reproduce` options:

```bash
python main.py reproduce --from-step 4       # resume from step N
python main.py reproduce --only ablation     # run only the matching step
python main.py reproduce --dry-run           # print the plan without executing
```

The reproduction pipeline runs ten Python scripts in order; each maps
one-to-one to a numerical row or figure in the manuscript:

| # | Script | Output |
| --- | --- | --- |
| 01 | `scripts/build_per_clip_predictions.py`     | Canonical per-clip parquet (OOF predictions for seven modality configurations) |
| 02a | `scripts/compute_overall_metrics.py`        | Tables 1 and 2: overall + per-class metrics with cluster-bootstrap CIs |
| 02b | `scripts/build_reference_tables.py`         | Four-configuration reference tables |
| 03a | `scripts/paired_ttest_configs.py`           | Fold-level paired *t*-tests + Holm-Bonferroni correction |
| 03b | `scripts/figure_modality_ablation.py`       | Modality-ablation figure |
| 04a | `scripts/compute_subgroup_metrics.py`       | Subgroup metrics (speech tertile × clinical state) |
| 04b | `scripts/figure_subgroup_heatmap.py`        | Subgroup-heatmap figure |
| 05 | `scripts/simulate_conditional_capture.py`   | Conditional-capture deployment table |
| 06 | `scripts/figure_calibration.py`             | Per-class reliability diagram |
| 07 | `scripts/figure_proposed_method.py`         | Proposed-method architecture figure |

## Data access

The raw multimodal data (smartphone-recorded audio, video, transcripts,
psychiatrist ratings) cannot be redistributed because of participant-privacy
restrictions. Researchers may request access through the institutional
contact named in the Data Availability Statement of the submitted
manuscript. See
[`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md) for the directory layout the
code expects once you have obtained the dataset; the path is configured
under `paths.root` in `config.yaml`.

Early-dropout participants (first-to-last recording span shorter than
`MIN_RECORDING_DAYS`, default **15 days**) are excluded automatically;
cohort identifiers are never hard-coded.

## Repository layout

```
.
├── main.py                            single entry point (reproduce / train)
├── config.yaml                        paths and pipeline configuration
├── requirements.txt                   Python dependencies
├── README.md
├── LICENSE
├── .env.example
├── .gitignore
├── docs/
│   ├── DATA_SCHEMA.md
│   └── diagram/pipeline.drawio.svg
├── src/                               shared Python package
│   ├── paths.py                       environment-driven path resolution
│   ├── config.py, utils.py, ...       pipeline utilities
│   ├── model_define.py                network and head definitions
│   ├── pipeline.py, sequence_build.py, normalize.py, ...
│   └── (other modules used by the entry points)
└── scripts/                           entry-point scripts
    ├── train.py                       three-stream Attention-MIL training
    ├── feature_loader.py              metadata + speech-segment helpers
    ├── build_per_clip_predictions.py
    ├── compute_overall_metrics.py
    ├── build_reference_tables.py
    ├── compute_subgroup_metrics.py
    ├── simulate_conditional_capture.py
    ├── paired_ttest_configs.py
    ├── figure_modality_ablation.py
    ├── figure_calibration.py
    ├── figure_subgroup_heatmap.py
    ├── figure_proposed_method.py
    ├── pr_curves.py, roc_curves.py, reliability_diagrams.py
    └── __init__.py
```

## Glossary

The codebase uses three short identifiers that may need clarification.

- **OOF (out-of-fold)** — predictions held out by each cross-validation
  fold and then aggregated across folds; the canonical OOF set is the
  1,768 clips from 29 participants reported in the manuscript.
- **CORN** — *cumulative ordinal regression for neural networks*
  (Shi et al., 2023); the rank-consistent ordinal head used here.
- **LA** — *logit adjustment* (Menon et al., 2021); the class-prior
  correction applied at the fusion logit.

## License

MIT License — see [`LICENSE`](LICENSE).
