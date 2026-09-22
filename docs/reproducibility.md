# Reproducing the Training and Evaluation

## Reproducibility scope

The repository scripts reproduce the MET training/evaluation, VO₂max LOSO comparison, and VO₂max LOAO analysis when the required feature files and private subject-information workbook are available.

Raw participant data and private metadata are not distributed in the repository. Therefore, a reviewer without authorized data access can inspect the complete processing and evaluation logic, run the offline demonstration with approved example inputs, and inspect the saved trained models, but cannot regenerate every training feature file from private study data unless access is granted separately.

## Recommended private data layout

Keep private inputs outside the Git repository:

```text
ThesisPrivateData/
├── SubjectsInfo.xlsx
├── InputMetFeatures/
│   └── *_imu_5s_features.npz
├── AllFeatures116_2s/
├── AllFeatures116_5s/
├── AllFeatures116_10s/
├── AllFeatures116_30s/
├── AllFeatures116_60s/
├── AllFeatures116_90s/
└── AllFeatures/
    ├── RestOut/
    ├── FoldClothesOut/
    ├── SweepOut/
    ├── WalkOut/
    ├── BoxCarryingOut/
    └── CyclingOut/
```

Each Stage 2 feature directory contains `*_hr_features.npz` files. See [data_format.md](data_format.md).

## Environment

From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For a final archived release, install from `requirements-lock.txt` if it is available.

## Stage 1: MET regression

### LOSO evaluation and final model

```powershell
py training\met_regression_loso_loao.py `
  --features-root "C:\ThesisPrivateData\InputMetFeatures" `
  --mode best `
  --output-dir "outputs\met"
```

This command performs LOSO evaluation using the optimized parameters and trains the final all-participant model. Principal outputs include:

```text
outputs/met/met_subject_loso_errors.csv
outputs/met/met_model_full.pt
```

### MET LOAO example

```powershell
py training\met_regression_loso_loao.py `
  --features-root "C:\ThesisPrivateData\InputMetFeatures" `
  --mode loao `
  --leave-out-met 4.5 `
  --output-dir "outputs\met_loao"
```

Repeat the command for each reference MET level required by the analysis. The script holds out every window carrying the selected reference MET value.

### Select CPU or GPU

The default is automatic selection. To force CPU execution:

```powershell
py training\met_regression_loso_loao.py `
  --features-root "C:\ThesisPrivateData\InputMetFeatures" `
  --mode best `
  --output-dir "outputs\met" `
  --device cpu
```

## Stage 2: LOSO temporal-context comparison

```powershell
py training\vo2max_regression_loso.py `
  --data-root-template "C:\ThesisPrivateData\AllFeatures116_{window_size}s" `
  --subject-info "C:\ThesisPrivateData\SubjectsInfo.xlsx" `
  --output-dir "outputs\loso"
```

By default, this evaluates 2, 5, 10, 30, 60, and 90 seconds with and without stability filtering. It saves:

- Participant-level fold results.
- Configuration-level metrics.
- Fold models and normalization metadata.
- One final all-participant model per configuration.
- `LOSO_window_comparison_summary.csv`.

To reproduce only the final thesis configuration without saving every fold model:

```powershell
py training\vo2max_regression_loso.py `
  --data-root-template "C:\ThesisPrivateData\AllFeatures116_{window_size}s" `
  --subject-info "C:\ThesisPrivateData\SubjectsInfo.xlsx" `
  --output-dir "outputs\loso_final" `
  --window-sizes 60 `
  --stability-configurations without `
  --skip-fold-models
```

## Stage 2: LOAO analysis

To run all activities using the stability-filtered configuration preserved by the original analysis script:

```powershell
py training\vo2max_regression_loao.py `
  --data-root "C:\ThesisPrivateData\AllFeatures" `
  --subject-info "C:\ThesisPrivateData\SubjectsInfo.xlsx" `
  --output-dir "outputs\loao_with_threshold" `
  --stability-filter with `
  --save-plots
```

To evaluate all valid windows without stability filtering:

```powershell
py training\vo2max_regression_loao.py `
  --data-root "C:\ThesisPrivateData\AllFeatures" `
  --subject-info "C:\ThesisPrivateData\SubjectsInfo.xlsx" `
  --output-dir "outputs\loao_without_threshold" `
  --stability-filter without `
  --save-plots
```

To run only selected activities:

```powershell
py training\vo2max_regression_loao.py `
  --data-root "C:\ThesisPrivateData\AllFeatures" `
  --subject-info "C:\ThesisPrivateData\SubjectsInfo.xlsx" `
  --output-dir "outputs\loao_selected" `
  --activities 0 5 `
  --stability-filter without
```

Activity `0` is rest and activity `5` is cycling.

## Expected reference results

The final 60-second configuration without stability filtering produced approximately:

```text
Average participant-level RMSE: 5.4529 mL·kg⁻¹·min⁻¹
Global RMSE:                    6.7205 mL·kg⁻¹·min⁻¹
R²:                             0.4173
Pearson r:                      0.6498
```

Stage 1 LOSO produced approximately:

```text
RMSE: 0.885 MET
Bias: 0.018 MET
```

Small numerical differences may occur across operating systems or package versions. Large deviations should be investigated before release.

## Reproducibility record

For every formal run, retain:

- Git commit hash.
- Release tag.
- Python version.
- `requirements-lock.txt`.
- Command executed.
- Input feature-generation version.
- Output summary CSV.
- Random seed.
- Selected stability configuration.
- Selected observation duration.

Do not commit private input data with this record.
