"""
Stage 8 - Stage 2 model: loads the trained VO2max regressor and its target
normalization, runs inference on the 116-feature windows, and de-normalizes
the predictions back to ml/kg/min.

Stage 9 - Aggregation: combines per-window predictions into a single
subject-level estimate.
"""
from __future__ import annotations
import numpy as np


def load_vo2_model(model_json_path, normalization_npz_path):
    """Loads the trained XGBoost regressor (native JSON format) plus its
    target normalization constants and training metadata."""
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(model_json_path)

    norm = np.load(normalization_npz_path, allow_pickle=True)
    y_mean, y_std = float(norm["y_mean"]), float(norm["y_std"])

    meta = {k: norm[k].item() if norm[k].shape == () else norm[k] for k in norm.files}
    return model, y_mean, y_std, meta


def run_vo2_stage2(X: np.ndarray, model, y_mean: float, y_std: float) -> np.ndarray:
    preds_norm = model.predict(X)
    return preds_norm * y_std + y_mean


def global_mean_estimate(preds_so_far: np.ndarray, stable_mask: np.ndarray = None) -> float:
    """
    Subject-level VO2max estimate: the mean of the per-window predictions,
    optionally restricted to windows that pass the stability filter used
    during training.
    """
    if len(preds_so_far) == 0:
        return float("nan")
    if stable_mask is not None:
        preds_so_far = preds_so_far[stable_mask[: len(preds_so_far)]]
        if len(preds_so_far) == 0:
            return float("nan")
    return float(np.mean(preds_so_far))
