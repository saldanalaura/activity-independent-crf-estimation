# Troubleshooting

## `ModuleNotFoundError`

Activate the repository environment and reinstall the dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## PowerShell will not activate the environment

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

This changes the policy only for the current PowerShell session.

## Streamlit command is not recognized

Use:

```powershell
python -m streamlit run app.py
```

## IMU and physiological recordings have no overlap

Confirm that:

- Both files are from the same session.
- Their dates and time zones are consistent.
- Timestamps were not converted to elapsed seconds in only one file.
- Timestamp columns are named `time` for IMU and `Time` for physiological data.

## Missing-column warning

Compare the uploaded headers with [data_format.md](data_format.md). The IMU loader converts column names to lowercase, while the physiological loader currently expects the exact names `Time`, `Oxygen Level`, and `Pulse Rate`.

## No valid 60-second windows

The final demonstration requires at least 60 seconds of valid overlapping data. Missing signals, timestamp gaps, or a shorter common interval can prevent window creation.

## Feature count mismatch

The Stage 2 model expects 116 features. Use a model and normalization file from the same directory. Do not combine artifacts from different observation durations or threshold configurations.

## Blank or undefined global estimate

For `Window_60s/Without_threshold`, disable stability filtering in the current interface. The no-threshold metadata has no applicable numerical stability threshold. A future application revision should read `stability_filter` from the normalization metadata automatically.

## Predictions appear but no trained model was supplied

Do not interpret them. The formal application should require all trained-model artifacts and should not use untrained placeholders. Until that behavior is removed from `app.py`, verify that the Stage 1 checkpoint, Stage 2 JSON model, and Stage 2 normalization file were all uploaded before running.

## Training script cannot find files

Run the script with `--help` and verify every supplied path:

```powershell
py training\vo2max_regression_loso.py --help
```

For LOSO, the data-root template must retain the literal `{window_size}` field:

```text
C:\ThesisPrivateData\AllFeatures116_{window_size}s
```

## `openpyxl` error when reading Excel

```powershell
pip install openpyxl
```

## CUDA was requested but is unavailable

Run MET training on the CPU:

```powershell
py training\met_regression_loso_loao.py ... --device cpu
```

## Results differ substantially from the thesis

Check:

- The feature schema contains 116 columns in the correct order.
- Participant-level aggregation uses the arithmetic mean.
- The same observation duration and overlap were used.
- Stability filtering matches the intended configuration.
- The subject-information workbook contains the same QCST recovery heart-rate and gender values.
- Extreme targets use the established 20th/80th percentile weighting with a factor of 10.
- The same XGBoost parameters and random seed are used.
- The feature-generation code matches the version associated with the saved models.
