# Trained Models

## Purpose

This directory contains the trained artifacts used by the offline demonstration and the observation-duration experiments.

## Stage 1

```text
met_model_full
```

The PyTorch checkpoint contains:

- `model_state_dict`
- `scaler_mean`
- `scaler_scale`
- `input_dim`
- Optimized model parameters

It converts the structured five-second IMU feature representation into continuous MET estimates.

## Stage 2 directory convention

```text
Window_<duration>s/
├── Without_threshold/
│   ├── final_all_subjects_model.json
│   └── final_all_subjects_normalization.npz
└── With_threshold/
    ├── final_all_subjects_model.json
    └── final_all_subjects_normalization.npz
```

Available durations are 2, 5, 10, 30, 60, and 90 seconds.

The JSON file stores the XGBoost regressor. The corresponding NPZ file stores target-normalization values and training metadata, including:

- `y_mean`
- `y_std`
- Target-weighting thresholds
- Number of features
- Observation duration
- Stability-filter status and threshold
- Number of training windows
- Aggregation method
- Training scope

The JSON and NPZ files must always be used as a matched pair from the same directory.

## Final thesis model

The final Stage 2 model is:

```text
Window_60s/Without_threshold/final_all_subjects_model.json
Window_60s/Without_threshold/final_all_subjects_normalization.npz
```

This model expects:

- 60-second observation windows.
- 116 input features.
- No stability-based window exclusion.
- Participant-level aggregation by arithmetic mean.

## Fold models versus final models

LOSO fold models are trained without one participant and are used only for cross-validation analysis. Final all-participant models are trained on every available participant and are the appropriate artifacts for the offline demonstration.

Do not use a held-out fold model as the general demonstration model unless reproducing that specific fold.

## Limitations

The models are research artifacts developed from the study population and acquisition protocol described in the thesis. They are not medical devices and must not be used for clinical decisions.
