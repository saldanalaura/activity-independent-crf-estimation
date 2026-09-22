# Input and Intermediate Data Formats

## Demonstration inputs

The offline application requires two CSV files from the same recording session.

### IMU CSV

The timestamp column is named `time`. IMU column names are converted to lowercase by the loader. Expected columns are:

```text
time
q_w_chest
q_x_chest
q_y_chest
q_z_chest
q_w_left_hand
q_x_left_hand
q_y_left_hand
q_z_left_hand
q_w_right_knee
q_x_right_knee
q_y_right_knee
q_z_right_knee
a_x_chest
a_y_chest
a_z_chest
g_x_chest
g_y_chest
g_z_chest
a_x_left_knee
a_y_left_knee
a_z_left_knee
g_x_left_knee
g_y_left_knee
g_z_left_knee
a_x_right_hand
a_y_right_hand
a_z_right_hand
g_x_right_hand
g_y_right_hand
g_z_right_hand
```

The expected IMU sampling rate is 50 Hz. The application uses the feature order established during model training. Do not reorder columns inside stored arrays or modify the positional feature-selection logic without retraining and revalidating Stage 1.

### Physiological CSV

These column names are currently case-sensitive:

```text
Time
Oxygen Level
Pulse Rate
```

The physiological signals were acquired at approximately 0.5 Hz in the study protocol.

### Timestamp format

Both timestamp columns must be readable by `pandas.to_datetime`. The recommended format is:

```text
YYYY-MM-DD HH:MM:SS.fff
```

Example:

```text
2026-01-15 10:30:00.000
```

The application keeps the common interval:

```text
shared_start = max(IMU start, physiological start)
shared_end   = min(IMU end, physiological end)
```

The files are rejected when `shared_start` is not earlier than `shared_end`.

## Participant characteristics

The Stage 2 demographic vector uses this order:

```text
[gender, age, height, weight, fat_fraction, BMI, resting_HR, resting_SpO2]
```

In the application, Male is encoded as `1` and Female as `0`. Body fat is entered as a percentage and divided by 100 before feature construction.

## MET training files

`training/met_regression_loso_loao.py` expects files matching:

```text
*_imu_5s_features.npz
```

Each archive must contain:

| Array | Description |
|---|---|
| `imu_5s_features` | Two-dimensional Stage 1 feature matrix |
| `imu_5s_t_start` | Start time in seconds for each five-second window |
| `imu_5s_t_end` | End time in seconds for each five-second window |

Reference MET labels are assigned from the midpoint of each window using the protocol schedule encoded in the script.

## VO₂max LOSO feature files

`training/vo2max_regression_loso.py` expects one directory for each window duration and files matching:

```text
*_hr_features.npz
```

Each filename must begin with the numeric participant identifier followed by a hyphen, for example:

```text
014-Participant14_hr_features.npz
```

Required arrays:

| Array | Required when | Description |
|---|---|---|
| `X` | Always | Two-dimensional 116-feature matrix |
| `stability_max` | Stability filtering enabled | Maximum stability duration associated with each row |

## VO₂max LOAO feature files

`training/vo2max_regression_loao.py` expects these subdirectories beneath `--data-root`:

```text
RestOut/
FoldClothesOut/
SweepOut/
WalkOut/
BoxCarryingOut/
CyclingOut/
```

Each `*_hr_features.npz` file must contain:

| Array | Description |
|---|---|
| `X` | Two-dimensional 116-feature matrix |
| `activity` | Integer activity identifier for every row |
| `stability_max` | Stability duration associated with every row |

Activity identifiers are:

| ID | Activity |
|---:|---|
| 0 | Rest |
| 1 | Folding clothes |
| 2 | Sweeping |
| 3 | Walking |
| 4 | Box carrying |
| 5 | Stationary cycling |

## Subject-information workbook

The LOSO and LOAO Stage 2 scripts require an Excel workbook containing exactly these columns:

```text
Participant ID
HR step test(bpm)
Gender
```

This workbook is private study metadata and must not be committed. The `.gitignore` excludes `SubjectsInfo.xlsx`.
