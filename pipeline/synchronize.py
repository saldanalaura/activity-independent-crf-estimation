"""Stage 2 — Synchronization. Mirrors synchronize_dataset.py's synchronize_subject()."""
from __future__ import annotations
import pandas as pd


def synchronize(imu_df: pd.DataFrame, bio_df: pd.DataFrame):
    imu_df = imu_df.dropna(subset=["time"]).copy()
    bio_df = bio_df.dropna(subset=["Time"]).copy()

    if imu_df.empty or bio_df.empty:
        raise ValueError("Empty timestamps after parsing — check the uploaded files.")

    imu_start, imu_end = imu_df["time"].min(), imu_df["time"].max()
    bio_start, bio_end = bio_df["Time"].min(), bio_df["Time"].max()

    shared_start = max(imu_start, bio_start)
    shared_end = min(imu_end, bio_end)

    if shared_start >= shared_end:
        raise ValueError("IMU and Biomarker files have no overlapping time window.")

    imu_sync = imu_df[(imu_df["time"] >= shared_start) & (imu_df["time"] <= shared_end)].copy()
    bio_sync = bio_df[(bio_df["Time"] >= shared_start) & (bio_df["Time"] <= shared_end)].copy()

    info = {
        "shared_start": shared_start,
        "shared_end": shared_end,
        "duration_sec": (shared_end - shared_start).total_seconds(),
        "imu_rows": len(imu_sync),
        "bio_rows": len(bio_sync),
    }
    return imu_sync, bio_sync, info
