"""Stage 3 — Filtering. Mirrors filtering.py's process_file() logic, in-memory."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def butter_highpass_filter(data, cutoff, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="high")
    return filtfilt(b, a, data)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, data)


def filter_imu(imu_df: pd.DataFrame, hp_cutoff=0.1, lp_cutoff=10.0, imu_fs=50.0) -> pd.DataFrame:
    """High-pass (DC removal) -> low-pass (10 Hz) -> per-column z-score. Time column passed through."""
    out = pd.DataFrame(index=imu_df.index)
    for col in imu_df.columns:
        if not np.issubdtype(imu_df[col].dtype, np.number):
            out[col] = imu_df[col]
            continue
        series = imu_df[col].astype(float)
        try:
            hp = butter_highpass_filter(series, hp_cutoff, imu_fs)
            lp = butter_lowpass_filter(hp, lp_cutoff, imu_fs)
            norm = (lp - np.mean(lp)) / (np.std(lp) + 1e-8)
            out[col] = norm
        except Exception:
            out[col] = series
    return out
