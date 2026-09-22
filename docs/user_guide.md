# Offline Demonstration User Guide

## Purpose

The Streamlit application demonstrates the complete two-stage inference pipeline using previously recorded wearable signals. It is an offline research prototype, not a real-time acquisition system or medical device.

The application performs:

1. Input validation and cleaning.
2. Temporal synchronization of inertial and physiological recordings.
3. IMU filtering and normalization.
4. Five-second Stage 1 window construction.
5. Continuous MET estimation and post-processing.
6. Stage 2 observation-window construction.
7. Construction of the 116-feature multimodal representation.
8. Window-level VO₂max prediction.
9. Participant-level aggregation by arithmetic mean.

## Start the application

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

## Select the model artifacts

For the final thesis configuration, use:

```text
Stage 1 model:
models/met_model_full

Stage 2 model:
models/Window_60s/Without_threshold/final_all_subjects_model.json

Stage 2 normalization:
models/Window_60s/Without_threshold/final_all_subjects_normalization.npz
```

Do not mix a model JSON file with a normalization file from a different window duration or threshold configuration.

The current application constructs 60-second Stage 2 windows. Models trained for 2, 5, 10, 30, or 90 seconds are retained as experimental artifacts but should not be selected in the 60-second demonstration unless the application windowing configuration is changed accordingly.

For the final no-threshold model, leave the stability-filter option disabled. A no-threshold normalization file stores no applicable stability threshold.

## Provide participant characteristics

Enter:

- Gender: Male or Female.
- Age in years.
- Height in meters.
- Weight in kilograms.
- Body fat percentage as a conventional percentage, such as `20.0` for 20%.
- Resting or baseline heart rate in beats per minute.
- Resting or baseline SpO₂ as a percentage.

The interface calculates BMI automatically. Internally, body fat percentage is converted from percentage units to a fraction to match the training representation.

## Upload the recordings

Upload:

1. One IMU CSV file.
2. One physiological CSV file containing heart rate and SpO₂.

Both files must come from the same session and have overlapping timestamps. See [data_format.md](data_format.md) for the required columns.

The physiological and inertial files should contain at least 60 seconds of valid overlapping data because the final demonstration uses 60-second Stage 2 windows.

## Run and interpret the demonstration

Select **Run pipeline** after the recordings, model files, and participant characteristics are ready.

The interface displays:

- The synchronized time interval.
- The number of five-second and 60-second windows.
- Window-level VO₂max estimates.
- The stability status associated with each window.
- The participant-level VO₂max estimate obtained from the arithmetic mean of the selected window predictions.

The animated option controls only how results are revealed on screen. It does not reproduce real-time sensing or alter the predictions.

## Validity checks

Do not interpret an output unless:

- All required model artifacts were loaded successfully.
- The model reports 116 expected features.
- The model window duration matches the application's Stage 2 window duration.
- The two input files overlap in time.
- No critical signal-quality warning remains unresolved.
- The application produced at least one valid Stage 2 window.

## Limitations

- The target is QCST-derived VO₂max rather than direct CPET VO₂max.
- The application processes previously recorded data.
- The model was developed from the study population and protocol described in the thesis.
- Performance outside the represented population, sensor configuration, and signal ranges has not been established.
- The interface is for research demonstration only.
