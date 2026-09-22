"""
Stage 1 - Load & clean.

Selects the expected IMU and biomarker columns, parses timestamps, and
applies a NaN-percentage threshold gate on the critical biomarker columns
(heart rate, SpO2). Assumes the uploaded CSVs carry full
"YYYY-MM-DD HH:MM:SS[.fff]" timestamps.
"""
from __future__ import annotations
import pandas as pd

IMU_KEEP_COLS = [
    "time",
    "q_w_chest", "q_x_chest", "q_y_chest", "q_z_chest",
    "q_w_left_hand", "q_x_left_hand", "q_y_left_hand", "q_z_left_hand",
    "q_w_right_knee", "q_x_right_knee", "q_y_right_knee", "q_z_right_knee",
    "a_x_chest", "a_y_chest", "a_z_chest",
    "g_x_chest", "g_y_chest", "g_z_chest",
    "a_x_left_knee", "a_y_left_knee", "a_z_left_knee",
    "g_x_left_knee", "g_y_left_knee", "g_z_left_knee",
    "a_x_right_hand", "a_y_right_hand", "a_z_right_hand",
    "g_x_right_hand", "g_y_right_hand", "g_z_right_hand",
]
BIO_KEEP_COLS = ["Time", "Oxygen Level", "Pulse Rate"]
BIO_CRITICAL_COLS = ["Oxygen Level", "Pulse Rate"]


class CleaningResult:
    def __init__(self, imu_df, bio_df, imu_nan_pct, bio_nan_pct, bio_exceeds_threshold, warnings):
        self.imu_df = imu_df
        self.bio_df = bio_df
        self.imu_nan_pct = imu_nan_pct
        self.bio_nan_pct = bio_nan_pct
        self.bio_exceeds_threshold = bio_exceeds_threshold
        self.warnings = warnings


def _overall_nan_pct(df: pd.DataFrame) -> float:
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return 0.0
    return float(df.isna().sum().sum()) / total_cells


def clean_by_threshold(df, critical_cols, threshold):
    """Drops rows with NaNs in the critical columns, unless the NaN rate on
    any of those columns is at or above the threshold, in which case the
    data is left untouched and flagged as not cleaned."""
    total = len(df) if len(df) > 0 else 1
    nan_pct = {}
    for c in critical_cols:
        nan_pct[c] = float(df[c].isna().sum()) / total if c in df.columns else 1.0
    exceeds = any(p >= threshold for p in nan_pct.values())
    if exceeds:
        return df.copy(), nan_pct, False
    mask = pd.Series(True, index=df.index)
    for c in critical_cols:
        if c in df.columns:
            mask &= df[c].notna()
    return df.loc[mask].copy(), nan_pct, True


def load_and_clean(imu_raw: pd.DataFrame, bio_raw: pd.DataFrame, nan_threshold: float = 0.10) -> CleaningResult:
    warnings = []

    imu = imu_raw.copy()
    imu.columns = [c.lower() for c in imu.columns]
    if "time_sec" in imu.columns:
        imu = imu.drop(columns=["time_sec"])
    imu = imu[[c for c in IMU_KEEP_COLS if c in imu.columns]]
    missing_imu_cols = [c for c in IMU_KEEP_COLS if c not in imu.columns]
    if missing_imu_cols:
        warnings.append(f"IMU file is missing expected columns: {missing_imu_cols}")
    imu["time"] = pd.to_datetime(imu["time"], errors="coerce")
    imu_nan_pct = _overall_nan_pct(imu) * 100.0

    bio = bio_raw.copy()
    bio = bio[[c for c in BIO_KEEP_COLS if c in bio.columns]]
    missing_bio_cols = [c for c in BIO_KEEP_COLS if c not in bio.columns]
    if missing_bio_cols:
        warnings.append(f"Biomarkers file is missing expected columns: {missing_bio_cols}")
    bio["Time"] = pd.to_datetime(bio["Time"], errors="coerce")
    bio_nan_pct_overall = _overall_nan_pct(bio) * 100.0

    bio_clean, bio_pct_per_col, ok = clean_by_threshold(bio, BIO_CRITICAL_COLS, nan_threshold)
    bio_exceeds = not ok
    if bio_exceeds:
        warnings.append(
            f"Biomarker NaN% exceeds the {nan_threshold*100:.0f}% threshold on one or more critical "
            f"columns ({bio_pct_per_col}) — rows were NOT dropped. Proceeding anyway for this demo, "
            f"but results may be unreliable."
        )
    else:
        bio = bio_clean

    return CleaningResult(
        imu_df=imu,
        bio_df=bio,
        imu_nan_pct=imu_nan_pct,
        bio_nan_pct=bio_nan_pct_overall,
        bio_exceeds_threshold=bio_exceeds,
        warnings=warnings,
    )
