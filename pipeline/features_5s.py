"""Stage 5 - 5-second IMU feature extraction."""
from __future__ import annotations
import numpy as np


def compute_structured_intensity(signal):
    if len(signal) == 0:
        return np.zeros(12)

    signal = np.asarray(signal)
    rms = np.sqrt(np.mean(signal**2))
    mav = np.mean(np.abs(signal))
    power = np.mean(signal**2)
    p95 = np.percentile(signal, 95)
    max_val = np.max(signal)
    ptp = np.ptp(signal)
    std = np.std(signal)
    mean = np.mean(signal)
    cv = std / (mean + 1e-8)

    N = len(signal)
    chunk_size = N // 4
    chunk_rms = []
    for i in range(4):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < 3 else N
        chunk = signal[start:end]
        chunk_rms.append(np.sqrt(np.mean(chunk**2)) if len(chunk) else 0.0)

    return np.array([
        rms, mav, power, p95, max_val, ptp, std, cv,
        chunk_rms[0], chunk_rms[1], chunk_rms[2], chunk_rms[3]
    ], dtype=float)


def extract_5s_features(imu_windows_5s: list, feature_columns: list) -> np.ndarray:
    """
    Computes a 12-value structured-intensity summary (RMS, mean absolute
    value, power, percentile, peak-to-peak, coefficient of variation, and
    per-quarter RMS) for six magnitude signals per window, taken from
    columns 0:9 and 9:18 of each window array. This positional selection
    is intentional and matches how the Stage 1 model's training features
    were generated, and must be kept as-is for compatibility with the
    trained checkpoint.
    """
    feature_matrix = []
    for seg in imu_windows_5s:
        segment = np.asarray(seg, dtype=float)
        feature_vec = []
        for j in range(0, 9, 3):
            x, y, z = segment[:, j:j + 3].T
            magnitude = np.sqrt(x**2 + y**2 + z**2)
            feature_vec.extend(compute_structured_intensity(magnitude))
        for j in range(9, 18, 3):
            x, y, z = segment[:, j:j + 3].T
            magnitude = np.sqrt(x**2 + y**2 + z**2)
            feature_vec.extend(compute_structured_intensity(magnitude))
        feature_matrix.append(feature_vec)
    return np.array(feature_matrix, dtype=float)
