#!/usr/bin/env python3
"""
MET regression training and evaluation.

Modes:
    --mode full   → Train on all subjects
    --mode loso   → Leave-One-Subject-Out
    --mode loao   → Leave-One-Activity-Out (controlled MET)

Example:
    python met_regression_loso_loao.py --features-root data/met_features \
        --mode best --output-dir outputs/met
"""
import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

PROTOCOL_SEGMENTS = [
    (0, 180, 1.0),
    (180, 480, 2.0),
    (480, 780, 2.3),
    (780, 960, 1.0),
    (960, 1260, 3.8),
    (1260, 1560, 4.5),
    (1560, 1740, 1.0),
    (1740, 2040, 6.8),
    (2040, 2220, 1.0),
]

BEST_PARAMS = {
    "epochs": 87,
    "lr": 0.0007112998432594841,
    "weight_decay": 7.598069868038522e-05,
    "batch_size": 64,
    "hidden_dim": 128,
    "dropout": 0.43483262700140846,
}

# ============================================================
# DATASET
# ============================================================

class METDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# MODEL
# ============================================================
   
class METRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
# FEATURE SELECTION
# ============================================================

def select_motion_features(data):
    return data["imu_5s_features"]


# ============================================================
# MET LABEL ASSIGNMENT
# ============================================================

def assign_met_from_time(t_start, t_end):
    mid_time = (t_start + t_end) / 2.0
    for start, end, met in PROTOCOL_SEGMENTS:
        if start <= mid_time < end:
            return met
    return 1.0


# ============================================================
# LOAD DATA
# ============================================================

def load_subjects(root):

    if not root.exists():
        raise FileNotFoundError(f"Feature directory does not exist: {root}")

    subjects = {}

    for file in sorted(root.glob("*_imu_5s_features.npz")):
        subj_name = file.stem.replace("_imu_5s_features", "")
        data = np.load(file, allow_pickle=True)

        X = select_motion_features(data)
        t_start = data["imu_5s_t_start"]
        t_end   = data["imu_5s_t_end"]
        y = np.array([
            assign_met_from_time(ts, te)
            for ts, te in zip(t_start, t_end)
        ])

        subjects[subj_name] = (X, y)

    if not subjects:
        raise FileNotFoundError(
            f"No *_imu_5s_features.npz files were found in: {root}"
        )

    return subjects


# ============================================================
# TRAINING CORE
# ============================================================

def train_model(
    train_X, train_y,
    test_X=None, test_y=None,
    epochs=50,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=264,
    hidden_dim=64,
    dropout=0.2,
    verbose=True
):

    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)

    train_ds = METDataset(train_X, train_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = METRegressor(
        input_dim=train_X.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if verbose:
            print(f"Epoch {epoch+1:03d} | Loss {total_loss/len(train_loader):.4f}")

    mae = None
    rmse = None
    r2   = None
    bias = None
    mean_pred = None

    if test_X is not None:

        test_X = scaler.transform(test_X)
        test_ds = METDataset(test_X, test_y)
        test_loader = DataLoader(test_ds, batch_size=256)

        model.eval()
        preds_all = []
        y_all = []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(DEVICE)
                preds = model(xb).cpu().numpy()
                preds_all.append(preds)
                y_all.append(yb.numpy())

        preds_all = np.concatenate(preds_all)
        y_all = np.concatenate(y_all)

        rmse = np.sqrt(mean_squared_error(y_all, preds_all))
        r2   = r2_score(y_all, preds_all)
        mae = mean_absolute_error(y_all, preds_all)
        bias = np.mean(preds_all - y_all)
        mean_pred = np.mean(preds_all)

    return model, scaler, rmse, r2, bias, mean_pred, mae

# ============================================================
# FULL MODE
# ============================================================
def run_full(subjects, save_path):

    save_path.parent.mkdir(parents=True, exist_ok=True)

    X_all, y_all = [], []

    for X, y in subjects.values():
        X_all.append(X)
        y_all.append(y)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    model, scaler, *_ = train_model(
        X_all,
        y_all,
        epochs=BEST_PARAMS["epochs"],
        lr=BEST_PARAMS["lr"],
        weight_decay=BEST_PARAMS["weight_decay"],
        batch_size=BEST_PARAMS["batch_size"],
        hidden_dim=BEST_PARAMS["hidden_dim"],
        dropout=BEST_PARAMS["dropout"],
        verbose=True
    )

    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "input_dim": X_all.shape[1],
        "best_params": BEST_PARAMS
    }, save_path)

    print(f"\nFull model saved to {save_path}")

def evaluate_loso(subjects, params, verbose=False, save_errors_path=None):

    rmses = []
    maes = []
    biases = []
    r2s = []
    subject_rows = []

    for test_subj in subjects.keys():

        train_X, train_y = [], []

        for subj, (X, y) in subjects.items():
            if subj == test_subj:
                test_X, test_y = X, y
            else:
                train_X.append(X)
                train_y.append(y)

        train_X = np.vstack(train_X)
        train_y = np.concatenate(train_y)

        _, _, rmse, r2, bias, mean_pred, mae = train_model(
            train_X,
            train_y,
            test_X,
            test_y,
            epochs=params["epochs"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            batch_size=params["batch_size"],
            hidden_dim=params["hidden_dim"],
            dropout=params["dropout"],
            verbose=verbose
        )

        rmses.append(rmse)
        maes.append(mae)
        biases.append(bias)

        if r2 is not None and not np.isnan(r2):
            r2s.append(r2)

        subject_rows.append({
            "subject": test_subj,
            "met_rmse": rmse,
            "met_mae": mae,
            "met_bias": bias,
            "met_mean_pred": mean_pred,
            "met_r2": r2
        })

    results = {
        "mean_rmse": float(np.mean(rmses)),
        "std_rmse": float(np.std(rmses)),
        "mean_mae": float(np.mean(maes)),
        "mean_bias": float(np.mean(biases)),
        "mean_r2": float(np.mean(r2s)) if len(r2s) > 0 else None,
    }

    subject_errors_df = pd.DataFrame(subject_rows)

    if save_errors_path is not None:
        save_errors_path = Path(save_errors_path)
        save_errors_path.parent.mkdir(parents=True, exist_ok=True)
        subject_errors_df.to_csv(save_errors_path, index=False)
        print(f"\nSaved subject-level MET errors to: {save_errors_path}")

    return results, subject_errors_df

def run_best_model(subjects, save_path):

    save_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n==============================")
    print("USING OPTIMIZED PARAMETERS")
    print("==============================")
    for k, v in BEST_PARAMS.items():
        print(f"{k}: {v}")

    print("\n==============================")
    print("LOSO EVALUATION")
    print("==============================")

    met_errors_path = save_path.parent / "met_subject_loso_errors.csv"
    loso_results, met_subject_errors = evaluate_loso(
        subjects,
        BEST_PARAMS,
        verbose=False,
        save_errors_path=met_errors_path
    )

    print(f"Mean RMSE : {loso_results['mean_rmse']:.4f}")
    print(f"Std RMSE  : {loso_results['std_rmse']:.4f}")
    print(f"Mean MAE  : {loso_results['mean_mae']:.4f}")
    print(f"Mean Bias : {loso_results['mean_bias']:.4f}")

    if loso_results["mean_r2"] is not None:
        print(f"Mean R2   : {loso_results['mean_r2']:.4f}")
    else:
        print("Mean R2   : N/A")

    print("\n==============================")
    print("TRAINING FINAL MODEL")
    print("==============================")

    X_all = []
    y_all = []

    for X, y in subjects.values():
        X_all.append(X)
        y_all.append(y)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    model, scaler, _, _, _, _, _ = train_model(
        X_all,
        y_all,
        epochs=BEST_PARAMS["epochs"],
        lr=BEST_PARAMS["lr"],
        weight_decay=BEST_PARAMS["weight_decay"],
        batch_size=BEST_PARAMS["batch_size"],
        hidden_dim=BEST_PARAMS["hidden_dim"],
        dropout=BEST_PARAMS["dropout"],
        verbose=True
    )

    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "input_dim": X_all.shape[1],
        "best_params": BEST_PARAMS,
        "loso_results": loso_results,
    }, save_path)

    print(f"\nModel saved to {save_path}")

# ============================================================
# LOSO MODE
# ============================================================

def run_loso(subjects, output_dir):

    rmses = []
    subject_rows = []

    for test_subj in subjects.keys():

        print("\n====================================")
        print(f"LOSO Test Subject: {test_subj}")
        print("====================================")

        train_X, train_y = [], []

        for subj, (X, y) in subjects.items():
            if subj == test_subj:
                test_X, test_y = X, y
            else:
                train_X.append(X)
                train_y.append(y)

        train_X = np.vstack(train_X)
        train_y = np.concatenate(train_y)

        _, _, rmse, r2, bias, mean_pred, mae = train_model(
            train_X,
            train_y,
            test_X,
            test_y,
            epochs=BEST_PARAMS["epochs"],
            lr=BEST_PARAMS["lr"],
            weight_decay=BEST_PARAMS["weight_decay"],
            batch_size=BEST_PARAMS["batch_size"],
            hidden_dim=BEST_PARAMS["hidden_dim"],
            dropout=BEST_PARAMS["dropout"],
            verbose=False,
        )

        print(f"RMSE: {rmse:.4f}")
        print(f"R2  : {r2:.4f}")
        print(f"MAE  : {mae:.4f}")
        print(f"Bias  : {bias:.4f}")

        rmses.append(rmse)

        subject_rows.append({
            "subject": test_subj,
            "met_rmse": rmse,
            "met_mae": mae,
            "met_bias": bias,
            "met_mean_pred": mean_pred,
            "met_r2": r2
        })

    print("\nMean LOSO RMSE:", np.mean(rmses))

    subject_errors_df = pd.DataFrame(subject_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "met_subject_loso_errors.csv"
    subject_errors_df.to_csv(output_path, index=False)
    print(f"Saved subject-level MET errors to: {output_path}")


# ============================================================
# LOAO MODE (Controlled)
# ============================================================

def run_loao(subjects, met_holdout, save_path=None):

    train_X, train_y = [], []
    test_X, test_y = [], []

    for X, y in subjects.values():

        train_mask = y != met_holdout
        test_mask  = y == met_holdout

        if np.any(train_mask):
            train_X.append(X[train_mask])
            train_y.append(y[train_mask])

        if np.any(test_mask):
            test_X.append(X[test_mask])
            test_y.append(y[test_mask])

    train_X = np.vstack(train_X)
    train_y = np.concatenate(train_y)
    test_X  = np.vstack(test_X)
    test_y  = np.concatenate(test_y)

    model, scaler, rmse, r2, bias, mean_pred, mae = train_model(
    train_X,
    train_y,
    test_X,
    test_y,
    epochs=BEST_PARAMS["epochs"],
    lr=BEST_PARAMS["lr"],
    weight_decay=BEST_PARAMS["weight_decay"],
    batch_size=BEST_PARAMS["batch_size"],
    hidden_dim=BEST_PARAMS["hidden_dim"],
    dropout=BEST_PARAMS["dropout"],
    verbose=True
)

    print(f"\nLOAO Hold-out MET: {met_holdout}")
    print(f"RMSE               : {rmse:.3f}")
    print(f"MAE                : {mae:.3f}")
    print(f"Bias               : {bias:.3f}")
    print(f"Mean predicted MET : {mean_pred:.3f}")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "input_dim": train_X.shape[1],
            "held_out_met": met_holdout,
            "best_params": BEST_PARAMS
        }, save_path)

        print(f"\nLOAO model saved to {save_path}")


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the MET regressor or run LOSO/LOAO evaluation using "
            "precomputed 5-second movement features."
        )
    )
    parser.add_argument(
        "--features-root",
        type=Path,
        required=True,
        help="Directory containing *_imu_5s_features.npz files.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["full", "loso", "loao", "best"],
        help=(
            "full: train on all data; loso: participant evaluation; "
            "loao: hold out one reference MET level; best: LOSO plus final model."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/met"),
        help="Directory for evaluation tables and default model artifacts.",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Optional model output path. Defaults inside --output-dir.",
    )
    parser.add_argument(
        "--leave-out-met",
        type=float,
        default=None,
        help="Reference MET value excluded in LOAO mode.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Torch execution device.",
    )
    return parser.parse_args()


def main():
    global DEVICE
    args = parse_args()

    if args.device == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")

    features_root = args.features_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    subjects = load_subjects(features_root)

    print("Running on:", DEVICE)
    print("Loaded subjects:", len(subjects))
    print("Path:", features_root)

    if args.mode == "full":
        save_model = args.save_model or output_dir / "met_model_full.pt"
        run_full(subjects, save_model.resolve())

    elif args.mode == "loso":
        run_loso(subjects, output_dir)

    elif args.mode == "loao":
        if args.leave_out_met is None:
            raise ValueError("LOAO mode requires --leave-out-met")
        save_model = args.save_model
        if save_model is None:
            met_label = str(args.leave_out_met).replace(".", "p")
            save_model = output_dir / f"met_model_loao_{met_label}.pt"
        run_loao(subjects, args.leave_out_met, save_model.resolve())

    elif args.mode == "best":
        save_model = args.save_model or output_dir / "met_model_full.pt"
        run_best_model(subjects, save_model.resolve())


if __name__ == "__main__":
    main()
