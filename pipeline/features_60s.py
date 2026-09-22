"""Stage 7 - 60-second feature vector construction (116-feature schema)."""
from __future__ import annotations
import numpy as np

IMU_FS = 50.0


def accel_magnitude(seg):
    seg = np.asarray(seg, dtype=float)
    ax, ay, az = seg.T
    return np.sqrt(ax**2 + ay**2 + az**2)


def safe_polyfit_slope(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 2 or np.std(y) < 1e-8:
        return 0.0
    try:
        return np.polyfit(np.arange(len(y)), y, 1)[0]
    except Exception:
        return 0.0


def movement_time_features(x):
    sma = np.sum(np.abs(x)) / len(x)
    enmo = np.mean(np.maximum(x - 1.0, 0))
    return np.array([np.mean(x), np.std(x), np.sqrt(np.mean(x**2)), np.max(x), np.ptp(x), sma, enmo])


def movement_freq_features(x, fs):
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    power = np.abs(X) ** 2
    total_energy = np.log1p(np.sum(power))
    dom_freq = freqs[np.argmax(power)]
    p = power / (np.sum(power) + 1e-8)
    spectral_entropy = -np.sum(p * np.log(p + 1e-8))
    spectral_centroid = np.sum(freqs * power) / (np.sum(power) + 1e-8)

    def band_energy(fmin, fmax):
        mask = (freqs >= fmin) & (freqs < fmax)
        return np.log1p(np.sum(power[mask]))

    return np.array([total_energy, dom_freq, spectral_entropy, spectral_centroid,
                      band_energy(0, 1), band_energy(1, 3), band_energy(3, 6)])


def movement_dynamics(x):
    dx = np.diff(x)
    slope = safe_polyfit_slope(x)
    var_dx = np.var(dx)
    return np.array([slope, var_dx])


def sensor_coordination(chest, knee, hand):
    c1 = np.corrcoef(chest, knee)[0, 1]
    c2 = np.corrcoef(chest, hand)[0, 1]
    c3 = np.corrcoef(knee, hand)[0, 1]
    c1 = 0 if np.isnan(c1) else c1
    c2 = 0 if np.isnan(c2) else c2
    c3 = 0 if np.isnan(c3) else c3
    return np.array([c1, c2, c3])


def hr_features(hr):
    hr = np.asarray(hr, dtype=float)
    hr = hr[np.isfinite(hr)]
    if len(hr) < 2:
        return np.zeros(10)
    dhr = np.diff(hr)
    try:
        slope = safe_polyfit_slope(hr)
    except Exception:
        slope = 0
    start, end = hr[0], hr[-1]
    delta = end - start
    energy = np.log1p(np.mean(hr**2))
    sign_changes = np.sum(np.diff(np.sign(dhr)) != 0) if len(dhr) > 1 else 0
    return np.array([np.mean(hr), np.std(hr), np.ptp(hr), slope, start, end,
                      delta, np.std(dhr), energy, sign_changes])


def biomarker_features(bio_seg):
    bio_seg = np.asarray(bio_seg, dtype=float)
    spo2 = bio_seg[:, 0]
    if len(spo2) < 2:
        return np.zeros(5)
    slope = safe_polyfit_slope(spo2)
    return np.array([np.mean(spo2), np.std(spo2), slope, np.ptp(spo2), spo2[-1] - spo2[0]])


def safe_corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2 or len(y) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return np.corrcoef(x, y)[0, 1]


def delayed_cross_corr(movement, hr, max_lag=30):
    best_corr, best_lag = 0, 0
    for lag in range(max_lag):
        a = movement if lag == 0 else movement[:-lag]
        b = hr if lag == 0 else hr[lag:]
        if len(a) < 5:
            continue
        corr = safe_corr(a, b)
        if np.isnan(corr):
            corr = 0
        if abs(corr) > abs(best_corr):
            best_corr, best_lag = corr, lag
    return best_corr, best_lag


def met_features(met_block, prev_met_mean, prev_met_std):
    met_vals = met_block[:, 0]
    abs_delta = met_block[:, 2]
    stability = met_block[:, 3]
    zone = met_block[:, 4]
    slope = safe_polyfit_slope(met_vals)
    return np.array([
        np.mean(met_vals), np.std(met_vals), np.max(met_vals), slope,
        np.mean(abs_delta), np.mean(stability), np.max(stability),
        np.mean(zone == 0), np.mean(zone == 1), np.mean(zone == 2), np.mean(zone == 3),
        prev_met_mean, prev_met_std
    ])


def hr_response_features(hr, met_vals, hr_rest, age):
    hr = np.asarray(hr)
    met_vals = np.asarray(met_vals)
    if len(hr) < 5 or len(met_vals) < 2:
        return np.zeros(7)

    met_resampled = np.interp(np.linspace(0, len(met_vals) - 1, len(hr)), np.arange(len(met_vals)), met_vals)
    met_mean = np.mean(met_resampled) + 1e-6
    response_gain = (np.mean(hr) - hr_rest) / met_mean

    dhr = np.diff(hr)
    dmet = np.diff(met_resampled)
    valid = np.abs(dmet) > 1e-6
    local_resp = np.mean(dhr[valid] / dmet[valid]) if np.any(valid) else 0

    rise = np.mean(dhr[dhr > 0]) if np.any(dhr > 0) else 0
    recovery = np.mean(-dhr[dhr < 0]) if np.any(dhr < 0) else 0

    corr_full = np.correlate(hr - np.mean(hr), met_resampled - np.mean(met_resampled), mode="full")
    lag = np.argmax(corr_full) - (len(hr) - 1)

    recovery_tc = np.sum(dhr < 0) / len(dhr)

    hr_max_est = 208 - 0.7 * age
    hr_reserve = max(hr_max_est - hr_rest, 1e-6)
    hr_reserve_util = (np.mean(hr) - hr_rest) / hr_reserve

    return np.array([response_gain, local_resp, rise, recovery, lag, recovery_tc, hr_reserve_util])


def hr_met_relationship(hr, met_vals):
    hr = np.asarray(hr, dtype=float)
    met_vals = np.asarray(met_vals, dtype=float)
    hr = hr[np.isfinite(hr)]
    met_vals = met_vals[np.isfinite(met_vals)]
    if len(hr) < 5 or len(met_vals) < 2:
        return np.zeros(6)

    met_resampled = np.interp(np.linspace(0, len(met_vals) - 1, len(hr)), np.arange(len(met_vals)), met_vals)
    if (np.std(met_resampled) < 1e-8 or np.std(hr) < 1e-8
            or not np.all(np.isfinite(hr)) or not np.all(np.isfinite(met_resampled))):
        return np.zeros(6)
    try:
        a, b = np.polyfit(met_resampled, hr, 1)
    except Exception:
        return np.zeros(6)

    pred = a * met_resampled + b
    error = hr - pred
    ss_res = np.sum(error**2)
    ss_tot = np.sum((hr - np.mean(hr))**2) + 1e-8
    r2 = 1 - ss_res / ss_tot
    return np.array([a, b, r2, np.mean(error), np.std(error), np.max(np.abs(error))])


def stability_features(hr, movement):
    hr = np.asarray(hr)
    movement = np.asarray(movement)
    if len(hr) < 5:
        return np.zeros(4)
    hr_auto = safe_corr(hr[:-1], hr[1:])
    move_auto = safe_corr(movement[:-1], movement[1:])
    window = 5
    corrs = [safe_corr(hr[i:i+window], movement[i:i+window]) for i in range(len(hr) - window)]
    corrs = np.array(corrs) if len(corrs) > 0 else np.array([0])
    return np.array([hr_auto, move_auto, np.mean(corrs), np.var(corrs)])


def overlapping_met_values(window_start, window_end, met_matrix, met_start, met_end):
    """
    Returns the MET estimates whose 5-second intervals overlap a given
    60-second Stage 2 window, used for the heart-rate/MET relationship
    features. Falls back to the nearest MET estimate at the window
    midpoint if no interval strictly overlaps.
    """
    overlap_mask = (met_start < window_end) & (met_end > window_start)
    met_values = np.asarray(met_matrix[overlap_mask, 0], dtype=float)
    met_values = met_values[np.isfinite(met_values)]
    if len(met_values) > 0:
        return met_values

    window_midpoint = 0.5 * (window_start + window_end)
    index = np.searchsorted(met_start, window_midpoint, side="right") - 1
    if index < 0 or index >= len(met_matrix):
        return np.array([], dtype=float)
    fallback_value = float(met_matrix[index, 0])
    if not np.isfinite(fallback_value):
        return np.array([], dtype=float)
    return np.array([fallback_value], dtype=float)


def get_feature_names():
    names = []
    sensors = ["chest", "knee", "hand"]
    time_feats = ["mean", "std", "rms", "max", "ptp", "sma", "enmo"]
    freq_feats = ["energy", "dom_freq", "entropy", "centroid", "band_0_1", "band_1_3", "band_3_6"]
    dyn_feats = ["slope", "var_dx"]
    for s in sensors:
        for f in time_feats:
            names.append(f"{s}_time_{f}")
        for f in freq_feats:
            names.append(f"{s}_freq_{f}")
        for f in dyn_feats:
            names.append(f"{s}_dyn_{f}")
    names += ["coord_chest_knee", "coord_chest_hand", "coord_knee_hand"]
    names += ["hr_mean", "hr_std", "hr_range", "hr_slope", "hr_start", "hr_end",
              "hr_delta", "hr_diff_std", "hr_energy", "hr_sign_changes"]
    names += ["hr_response_gain", "hr_response_local", "hr_rise_slope", "hr_recovery_slope",
              "hr_response_delay", "hr_recovery_time_constant", "hr_reserve_utilization"]
    names += ["hr_met_slope", "hr_met_intercept", "hr_met_r2",
              "hr_met_error_mean", "hr_met_error_std", "hr_met_error_max"]
    names += ["spo2_mean", "spo2_std", "spo2_slope", "spo2_range", "spo2_delta"]
    for s in sensors:
        names += [f"{s}_hr_corr", f"{s}_hr_lag", f"{s}_dhr_corr"]
    names += ["hr_autocorr", "movement_autocorr", "corr_mean", "corr_var"]
    names += ["met_mean", "met_std", "met_max", "met_slope", "met_abs_delta_mean",
              "met_stability_mean", "met_stability_max",
              "met_zone0", "met_zone1", "met_zone2", "met_zone3",
              "prev_met_mean", "prev_met_std"]
    names += ["hr_met_ratio", "hr_reserve_per_met", "efficiency_inv"]
    names += ["gender", "age", "height", "weight", "fat_pct", "bmi", "hr_rest", "spo2_rest"]
    return names


def build_feature_vectors(windowed, met_matrix, met_t_start, met_t_end, demo_vector, hr_rest, age):
    """
    Builds the 116-column feature matrix for a subject's 60-second windows:
    movement (time/frequency/dynamics per sensor), sensor coordination,
    heart-rate and SpO2 features, heart-rate/MET relationship features,
    movement-heart-rate coupling, internal stability, MET summary features,
    efficiency ratios, and the subject's demographic vector.

    windowed: result["paired_60s"] dict from windowing.segment_windows
    met_matrix: output of met_stage1.run_met_stage1 (5-second resolution)
    met_t_start / met_t_end: start/end timestamps of the 5-second windows
    demo_vector: [gender, age, height, weight, fat_pct, bmi, hr_rest, spo2_rest]

    The chest/knee/hand accelerometer magnitudes are taken from a fixed
    column-position slice of each 60-second window array, matching the
    original training feature extraction.
    """
    imu_windows = windowed["imu"]
    bio_windows = windowed["bio"]
    t_start = windowed["t_start"]
    t_end = windowed["t_end"]

    rows = []
    window_times = []
    window_stability_seconds = []
    prev_met_mean, prev_met_std = 0, 0

    for i in range(len(imu_windows)):
        imu_seg = np.asarray(imu_windows[i], dtype=float)
        bio_seg = np.asarray(bio_windows[i], dtype=float)

        window_mid = 0.5 * (t_start[i] + t_end[i])
        idx = np.searchsorted(met_t_start, window_mid, side="right") - 1
        if idx < 0:
            continue

        met_block = met_matrix[idx:idx + 1]
        met_feats = met_features(met_block, prev_met_mean, prev_met_std)
        prev_met_mean, prev_met_std = met_feats[0], met_feats[1]

        chest = accel_magnitude(imu_seg[:, 0:3])
        knee = accel_magnitude(imu_seg[:, 3:6])
        hand = accel_magnitude(imu_seg[:, 6:9])

        move_feats = []
        for sig in [chest, knee, hand]:
            move_feats.extend(movement_time_features(sig))
            move_feats.extend(movement_freq_features(sig, IMU_FS))
            move_feats.extend(movement_dynamics(sig))
        move_feats = np.array(move_feats)

        coord = sensor_coordination(chest, knee, hand)

        hr = bio_seg[:, 1].astype(float)  # biomarker columns: [SpO2, heart rate]
        hr_feats = hr_features(hr)
        met_vals = met_block[:, 0]
        met_vals_relationship = overlapping_met_values(
            window_start=t_start[i], window_end=t_end[i],
            met_matrix=met_matrix, met_start=met_t_start, met_end=met_t_end,
        )

        hr_mean = np.mean(hr)
        met_mean = np.mean(met_vals) + 1e-6
        hr_met_ratio = hr_mean / met_mean
        hr_reserve = hr_mean - hr_rest
        hr_reserve_per_met = hr_reserve / met_mean
        efficiency_inv = met_mean / (hr_mean + 1e-6)
        efficiency_feats = np.array([hr_met_ratio, hr_reserve_per_met, efficiency_inv])

        hr_resp = hr_response_features(hr, met_vals, hr_rest, age)
        hr_met = hr_met_relationship(hr, met_vals_relationship)
        internal_stab = stability_features(hr, chest[:len(hr)])

        bio_feats = biomarker_features(bio_seg)

        coup_feats = []
        for sig in [chest, knee, hand]:
            corr, lag = delayed_cross_corr(sig[:len(hr)], hr)
            coup_feats.extend([corr, lag])
            d_corr = safe_corr(np.diff(sig[:len(hr)]), np.diff(hr))
            coup_feats.append(0 if np.isnan(d_corr) else d_corr)
        coup_feats = np.array(coup_feats)

        row = np.concatenate([
            move_feats, coord, hr_feats, hr_resp, hr_met, bio_feats,
            coup_feats, internal_stab, met_feats, efficiency_feats, demo_vector
        ])
        rows.append(row)
        window_times.append((t_start[i], t_end[i]))
        window_stability_seconds.append(met_block[0, 3])

    if not rows:
        return np.zeros((0, len(get_feature_names()))), [], np.array([])

    X = np.vstack(rows)
    names = get_feature_names()
    if X.shape[1] != len(names):
        raise ValueError(f"Feature mismatch: X has {X.shape[1]} cols, expected {len(names)}")
    return X, window_times, np.array(window_stability_seconds)
