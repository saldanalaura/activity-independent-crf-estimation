"""
Stage 6 - Stage 1 model: takes the 5-second IMU intensity features and
produces a smoothed MET (metabolic equivalent) curve plus its derived
features (rate of change, short-term stability, and intensity zone).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import medfilt

WINDOW_STEP_SECONDS = 2.5   # 5s windows @ 50% overlap
ROLLING_WINDOW = 4          # 4 x 2.5s = 10 seconds
CV_THRESHOLD = 0.08


class METRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.43483262700140846):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_met_model(checkpoint_path, device="cpu"):
    """Loads the trained Stage 1 checkpoint, including the feature scaler
    parameters that were saved alongside the model weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dim = checkpoint.get("best_params", {}).get("hidden_dim", 128)
    dropout = checkpoint.get("best_params", {}).get("dropout", 0.43483262700140846)

    model = METRegressor(checkpoint["input_dim"], hidden_dim=hidden_dim, dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler_mean = checkpoint["scaler_mean"]
    scaler_scale = checkpoint["scaler_scale"]
    return model, scaler_mean, scaler_scale


def run_met_stage1(imu_5s_features: np.ndarray, model, scaler_mean, scaler_scale, device="cpu") -> np.ndarray:
    """
    Runs the MET regressor over all 5-second windows of a session, then
    smooths the raw predictions with a median filter followed by a moving
    average. From the smoothed curve, derives the step-to-step change,
    a rolling stability counter (how long the coefficient of variation has
    stayed below a threshold), and a discrete intensity zone.

    Returns an array with columns:
    [met_smooth, met_delta, abs_delta, stability_seconds, met_zone]
    """
    X_scaled = (imu_5s_features - scaler_mean) / scaler_scale

    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        met_pred = model(X_tensor).cpu().numpy()

    met_med = medfilt(met_pred, kernel_size=7)
    met_smooth = np.convolve(met_med, np.ones(5) / 5, mode="same")
    N = len(met_smooth)

    met_delta = np.diff(met_smooth, prepend=met_smooth[0])
    abs_delta = np.abs(met_delta)

    rolling_mean = np.zeros(N)
    rolling_std = np.zeros(N)
    rolling_cv = np.zeros(N)
    for i in range(N):
        start = max(0, i - ROLLING_WINDOW + 1)
        segment = met_smooth[start:i + 1]
        rolling_mean[i] = np.mean(segment)
        rolling_std[i] = np.std(segment)
        rolling_cv[i] = rolling_std[i] / rolling_mean[i] if rolling_mean[i] > 1e-6 else 0

    stability_seconds = np.zeros(N)
    counter = 0
    for i in range(N):
        counter = counter + 1 if rolling_cv[i] < CV_THRESHOLD else 0
        stability_seconds[i] = counter * WINDOW_STEP_SECONDS

    met_zone = np.zeros(N)
    met_zone[met_smooth < 1.5] = 0
    met_zone[(met_smooth >= 1.5) & (met_smooth < 3.0)] = 1
    met_zone[(met_smooth >= 3.0) & (met_smooth < 4.5)] = 2
    met_zone[met_smooth >= 4.5] = 3

    return np.column_stack([met_smooth, met_delta, abs_delta, stability_seconds, met_zone])
