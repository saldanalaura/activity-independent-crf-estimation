# Training and Evaluation Scripts

This directory contains the portable scripts used to train the models and reproduce the principal experiments.

## Scripts

### `met_regression_loso_loao.py`

Supports:

- Full Stage 1 training.
- MET LOSO evaluation.
- MET leave-one-reference-level-out evaluation.
- Optimized LOSO evaluation followed by final-model training.

Inspect its interface:

```powershell
py training\met_regression_loso_loao.py --help
```

### `vo2max_regression_loso.py`

Evaluates Stage 2 across 2, 5, 10, 30, 60, and 90 seconds, with and without stability filtering. It saves participant-level results, aggregate metrics, optional fold models, and optional all-participant models.

```powershell
py training\vo2max_regression_loso.py --help
```

### `vo2max_regression_loao.py`

Combines leave-one-activity-out and leave-one-subject-out evaluation. It supports selected activity IDs, optional stability filtering, optional fold-model storage, and saved diagnostic plots.

```powershell
py training\vo2max_regression_loao.py --help
```

## Inputs

The scripts expect precomputed NPZ feature files and a private subject-information workbook. Exact schemas are documented in `docs/data_format.md`.

## Outputs

Write generated artifacts beneath `outputs/` or another excluded directory. Do not commit participant-level results until their distribution has been reviewed and approved.

## Reproduction commands

See `docs/reproducibility.md` for complete PowerShell commands and expected reference metrics.
