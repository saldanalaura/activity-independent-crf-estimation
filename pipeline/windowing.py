"""
Stage 4 - Window segmentation. Produces two parallel window sets from the
synchronized, filtered signals: 5-second IMU-only windows for the Stage 1
(MET) model, and paired 60-second IMU + biomarker windows for the Stage 2
(VO2max) model.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

IMU_FS = 50.0


def load_with_common_time(imu_df: pd.DataFrame, bio_df: pd.DataFrame):
    imu = imu_df.rename(columns={"time": "time"}).copy()
    bio = bio_df.rename(columns={"Time": "time"}).copy()
    imu.columns = [c.lower() if c != "time" else "time" for c in imu.columns]
    bio.columns = [c.lower() if c != "time" else "time" for c in bio.columns]

    imu = imu.dropna(subset=["time"])
    bio = bio.dropna(subset=["time"])

    base_time = max(imu["time"].min(), bio["time"].min())
    end_time = min(imu["time"].max(), bio["time"].max())

    imu = imu.copy()
    bio = bio.copy()
    imu["t_sec"] = (imu["time"] - base_time).dt.total_seconds()
    bio["t_sec"] = (bio["time"] - base_time).dt.total_seconds()

    duration = (end_time - base_time).total_seconds()

    imu = imu[(imu["t_sec"] >= 0) & (imu["t_sec"] <= duration)]
    bio = bio[(bio["t_sec"] >= 0) & (bio["t_sec"] <= duration)]

    return imu, bio, duration


def extract_windows(df, window_size, step, timeline_end):
    windows, t_start, t_end = [], [], []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c != "t_sec"]

    t0 = 0.0
    while t0 + window_size <= timeline_end:
        w0, w1 = t0, t0 + window_size
        seg = df[(df["t_sec"] >= w0) & (df["t_sec"] < w1)]
        windows.append(seg[feature_cols].to_numpy())
        t_start.append(w0)
        t_end.append(w1)
        t0 += step

    return windows, np.array(t_start), np.array(t_end), feature_cols


def segment_windows(
    imu_df: pd.DataFrame,
    bio_df: pd.DataFrame,
    imu_window=5.0, imu_overlap=0.5,
    bio_window=60.0, bio_overlap=0.5,
):
    imu, bio, duration = load_with_common_time(imu_df, bio_df)

    imu_step = imu_window * (1 - imu_overlap)
    bio_step = bio_window * (1 - bio_overlap)

    # 5-second IMU-only windows (Stage 1 input). Windows are kept regardless
    # of exact sample count: coarse/duplicate timestamps in the raw IMU file
    # can leave a handful of windows with slightly fewer samples than the
    # nominal count, and these are still valid observations rather than
    # missing data.
    imu_windows_5s, t0_5s, t1_5s, imu_cols_5s = extract_windows(imu, imu_window, imu_step, duration)

    # 60-second paired IMU + biomarker windows (Stage 2 input)
    imu_cols_bio = [c for c in imu.select_dtypes(include=[np.number]).columns if c != "t_sec"]
    bio_cols = [c for c in bio.select_dtypes(include=[np.number]).columns if c != "t_sec"]

    paired_imu, paired_bio, paired_t0, paired_t1 = [], [], [], []

    t0 = 0.0
    while t0 + bio_window <= duration:
        w0, w1 = t0, t0 + bio_window
        imu_seg = imu[(imu["t_sec"] >= w0) & (imu["t_sec"] < w1)]
        bio_seg = bio[(bio["t_sec"] >= w0) & (bio["t_sec"] < w1)]

        imu_array = imu_seg[imu_cols_bio].to_numpy()
        bio_array = bio_seg[bio_cols].to_numpy()

        valid = (
            imu_array.size > 0 and bio_array.size > 0
            and not np.isnan(imu_array).any() and not np.isnan(bio_array).any()
            and imu_array.shape[0] > 0 and np.abs(bio_array).sum() > 0
        )

        if valid:
            paired_imu.append(imu_array)
            paired_bio.append(bio_array)
            paired_t0.append(w0)
            paired_t1.append(w1)

        t0 += bio_step

    result = {
        "duration": duration,
        "imu_5s": {
            "windows": imu_windows_5s, "t_start": t0_5s, "t_end": t1_5s, "feature_columns": imu_cols_5s,
        },
        "paired_60s": {
            "imu": paired_imu, "bio": paired_bio,
            "t_start": np.array(paired_t0), "t_end": np.array(paired_t1),
            "imu_cols": imu_cols_bio, "bio_cols": bio_cols,
        },
    }
    return result
