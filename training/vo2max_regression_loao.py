#!/usr/bin/env python3
"""Evaluate VO2max generalization to activities excluded from training.

For each requested activity, the script combines leave-one-activity-out
training with leave-one-subject-out evaluation. Results, model artifacts, and
optional diagnostic plots are written to a user-selected output directory.

Example:
    python vo2max_regression_loao.py --data-root data/AllFeatures \
        --subject-info data/SubjectsInfo.xlsx --output-dir outputs/loao
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# ==========================================================
# CONFIG
# ==========================================================

DEFAULT_STABILITY_THRESHOLD = 60.0

# 0=rest, 1=fold, 2=sweep, 3=walk, 4=box, 5=cycling
HELDOUT_ACTIVITIES = [0, 1, 2, 3, 4, 5]

XGB_PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=14,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    min_child_weight=2.0,
    gamma=0.0,
    objective="reg:pseudohubererror",
    random_state=42,
    n_jobs=-1,
)

# ==========================================================
# FOLDER MAP
# ==========================================================

def activity_folder_name(activity):
    FOLDER_MAP = {
        0: "RestOut",
        1: "FoldClothesOut",
        2: "SweepOut",
        3: "WalkOut",
        4: "BoxCarryingOut",
        5: "CyclingOut",
    }
    return FOLDER_MAP[activity]


def activity_name(activity):
    NAME_MAP = {
        0: "Rest",
        1: "Fold Clothes",
        2: "Sweep",
        3: "Walk",
        4: "Box Carrying",
        5: "Cycling",
    }
    return NAME_MAP[activity]


def get_data_root(base_data_root, heldout_activity):
    return base_data_root / activity_folder_name(heldout_activity)


# ==========================================================
# SUBJECT METADATA AND VO2MAX TARGETS
# ==========================================================

def compute_vo2max(hr, sex):
    """Calculate QCST-derived VO2max using the sex-specific equation."""
    if str(sex).strip().upper() == "M":
        return 111.33 - 0.42 * hr
    return 65.81 - 0.1847 * hr


def load_subject_metadata(subject_info_path):
    if not subject_info_path.exists():
        raise FileNotFoundError(
            f"Subject-information file does not exist: {subject_info_path}"
        )

    subjects_info = pd.read_excel(subject_info_path)
    required_columns = {
        "Participant ID",
        "HR step test(bpm)",
        "Gender",
    }
    missing_columns = required_columns.difference(subjects_info.columns)
    if missing_columns:
        raise ValueError(
            "The subject-information file is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    participant_ids = subjects_info["Participant ID"].astype(int)
    hr_map = dict(zip(participant_ids, subjects_info["HR step test(bpm)"]))
    sex_map = dict(zip(participant_ids, subjects_info["Gender"]))

    return {
        subject_id: compute_vo2max(hr_map[subject_id], sex_map[subject_id])
        for subject_id in hr_map
    }


# ==========================================================
# HELPERS
# ==========================================================

def cumulative_predictions(preds, step=5):
    out = []

    for k in range(step, len(preds) + 1, step):
        out.append(np.mean(preds[:k]))

    return np.array(out)


def subject_id_from_file(file_path):
    """Extract the numeric participant identifier from a feature filename."""
    return int(file_path.stem.split("-")[0])


def find_subject_file(data_root, subject_id):
    matches = [
        path
        for path in data_root.glob("*_hr_features.npz")
        if subject_id_from_file(path) == subject_id
    ]
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple feature files were found for participant {subject_id}: "
            + ", ".join(path.name for path in matches)
        )
    return matches[0]


# ==========================================================
# LOAD TRAIN
# ==========================================================

def load_all_subjects(
    data_root,
    vo2_map,
    stability_threshold,
    use_stability_filter,
    exclude_subject=None,
    heldout_activity=None,
):

    X_all = []
    y_all = []
    subj_ids = []

    files = sorted(data_root.glob("*_hr_features.npz"))

    for file in files:

        subj_name = file.stem.replace("_hr_features", "")
        subj_id = int(subj_name.split("-")[0])

        if exclude_subject is not None and subj_id == exclude_subject:
            continue

        data = np.load(file, allow_pickle=True)

        X = data["X"]
        activity = data["activity"]
        stability = data["stability_max"]

        if subj_id not in vo2_map:
            raise KeyError(
                f"Participant {subj_id} from {file.name} is missing from "
                "the subject-information file."
            )

        y_value = vo2_map[subj_id]
        y = np.full(len(X), y_value)

        if use_stability_filter:
            mask = stability >= stability_threshold
        else:
            mask = np.ones(len(X), dtype=bool)

        # remove held-out activity from training
        if heldout_activity is not None:
            mask &= activity != heldout_activity

        if np.sum(mask) == 0:
            continue

        X_all.append(X[mask])
        y_all.append(y[mask])
        subj_ids.extend([subj_id] * int(np.sum(mask)))

    if len(X_all) == 0:
        return None, None, None

    return (
        np.vstack(X_all),
        np.concatenate(y_all),
        np.array(subj_ids),
    )


# ==========================================================
# LOAD TEST SUBJECT
# ==========================================================

def load_subject(
    data_root,
    subj_id,
    heldout_activity,
    vo2_map,
    stability_threshold,
    use_stability_filter,
):

    file = find_subject_file(data_root, subj_id)

    if file is None:
        return None, None

    data = np.load(file, allow_pickle=True)

    X = data["X"]
    activity = data["activity"]
    stability = data["stability_max"]

    if subj_id not in vo2_map:
        raise KeyError(
            f"Participant {subj_id} is missing from the subject-information file."
        )

    y_value = vo2_map[subj_id]
    y = np.full(len(X), y_value)

    if use_stability_filter:
        mask = stability >= stability_threshold
    else:
        mask = np.ones(len(X), dtype=bool)

    # keep ONLY held-out activity
    mask &= activity == heldout_activity

    if np.sum(mask) == 0:
        return None, None

    return X[mask], y[mask]


# ==========================================================
# LOSO + LOAO
# ==========================================================

def train_loso_loao(
    test_subject,
    heldout_activity,
    data_root,
    output_root,
    vo2_map,
    stability_threshold,
    use_stability_filter,
    save_fold_models,
    params,
):

    X_train, y_train, _ = load_all_subjects(
        data_root=data_root,
        vo2_map=vo2_map,
        stability_threshold=stability_threshold,
        use_stability_filter=use_stability_filter,
        exclude_subject=test_subject,
        heldout_activity=heldout_activity
    )

    if X_train is None:
        return None

    X_test, y_test = load_subject(
        data_root=data_root,
        subj_id=test_subject,
        heldout_activity=heldout_activity,
        vo2_map=vo2_map,
        stability_threshold=stability_threshold,
        use_stability_filter=use_stability_filter,
    )

    if X_test is None:
        return None

    print(
        f"Subject {test_subject:03d} | "
        f"Train {X_train.shape} | "
        f"Test {X_test.shape}"
    )

    # normalize target
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)

    if y_std < 1e-6:
        y_std = 1.0

    y_train_norm = (y_train - y_mean) / y_std

    # weight extremes
    low_thr = np.percentile(y_train, 20)
    high_thr = np.percentile(y_train, 80)

    weights = np.ones_like(y_train)
    weights[y_train < low_thr] = 10.0
    weights[y_train > high_thr] = 10.0

    model = XGBRegressor(**params)

    model.fit(
        X_train,
        y_train_norm,
        sample_weight=weights
    )

    activity_model_directory = (
        output_root / activity_folder_name(heldout_activity)
    )
    activity_model_directory.mkdir(parents=True, exist_ok=True)

    model_path = ""
    normalization_path = ""
    if save_fold_models:
        model_path_object = (
            activity_model_directory
            / f"heldout_subject_{test_subject:03d}_model.json"
        )
        normalization_path_object = (
            activity_model_directory
            / f"heldout_subject_{test_subject:03d}_normalization.npz"
        )

        model.save_model(str(model_path_object))
        np.savez(
            normalization_path_object,
            y_mean=y_mean,
            y_std=y_std,
            low_threshold=low_thr,
            high_threshold=high_thr,
            low_percentile=20,
            high_percentile=80,
            extreme_weight=10.0,
            n_features=X_train.shape[1],
            stability_filter=use_stability_filter,
            stability_threshold=(
                stability_threshold if use_stability_filter else np.nan
            ),
            heldout_subject=test_subject,
            heldout_activity=heldout_activity,
            heldout_activity_name=activity_name(heldout_activity),
            aggregation="mean_of_all_valid_participant_windows",
        )
        model_path = str(model_path_object)
        normalization_path = str(normalization_path_object)

    preds_norm = model.predict(X_test)
    preds_real = preds_norm * y_std + y_mean

    true_subject = float(y_test[0])

    # cumulative curve
    cum_preds = cumulative_predictions(preds_real, step=5)
    cum_errors = np.abs(cum_preds - true_subject)

    # Normal participant-level aggregation used in the thesis methodology:
    # average every valid window prediction for the held-out participant.
    pred_subject = float(np.mean(preds_real))

    # Each LOSO fold contains one held-out participant, so this fold-level
    # RMSE is equal to the absolute participant-level prediction error.
    rmse = float(np.sqrt((pred_subject - true_subject) ** 2))

    residual = pred_subject - true_subject

    return {
        "subject": test_subject,
        "rmse": rmse,
        "pred": pred_subject,
        "true": true_subject,
        "residual": residual,
        "absolute_residual": abs(residual),
        "n_test_windows": len(preds_real),
        "model_path": model_path,
        "normalization_path": normalization_path,
        "curve": cum_errors,
    }


# ==========================================================
# EVALUATE ONE ACTIVITY
# ==========================================================

def evaluate_activity(
    heldout_activity,
    base_data_root,
    output_root,
    vo2_map,
    stability_threshold,
    use_stability_filter,
    save_fold_models,
    save_plots,
):

    print("\n================================================")
    print(f"HOLD OUT: {activity_name(heldout_activity)}")
    print("================================================")

    data_root = get_data_root(base_data_root, heldout_activity)
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_root}")

    subjects = sorted({
        subject_id_from_file(f)
        for f in data_root.glob("*_hr_features.npz")
    })

    results = []
    all_curves = []

    for subj in subjects:

        result = train_loso_loao(
            subj,
            heldout_activity,
            data_root,
            output_root,
            vo2_map,
            stability_threshold,
            use_stability_filter,
            save_fold_models,
            XGB_PARAMS,
        )

        if result is None:
            continue

        results.append(result)

        if len(result["curve"]) > 0:
            all_curves.append(result["curve"])

    if len(results) == 0:
        return None

    print("\n==============================")
    print("SUMMARY")
    print("==============================")

    for r in sorted(results, key=lambda x: x["rmse"], reverse=True):
        print(
            f"{r['subject']:03d} | "
            f"True {r['true']:.1f} | "
            f"Pred {r['pred']:.1f} | "
            f"RMSE {r['rmse']:.2f}"
        )

    true_vals = np.array([r["true"] for r in results])
    pred_vals = np.array([r["pred"] for r in results])

    avg_rmse = np.mean([r["rmse"] for r in results])
    global_rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
    r2 = r2_score(true_vals, pred_vals)
    r, p_value = pearsonr(true_vals, pred_vals)

    a, b = np.polyfit(true_vals, pred_vals, 1)

    print("\n==============================")
    print("FINAL METRICS")
    print("==============================")
    print("AVG RMSE:", avg_rmse)
    print("Global RMSE:", global_rmse)
    print("R2:", r2)
    print("r:", r)
    print(f"Fit: pred = {a:.3f} * true + {b:.3f}")

    # Save all participant-level predictions and residuals for this activity.
    results_for_csv = [
        {key: value for key, value in result.items() if key != "curve"}
        for result in results
    ]
    results_df = pd.DataFrame(results_for_csv).sort_values("subject")

    activity_output_directory = (
        output_root / activity_folder_name(heldout_activity)
    )
    activity_output_directory.mkdir(parents=True, exist_ok=True)

    participant_results_path = (
        activity_output_directory / "subject_results.csv"
    )
    results_df.to_csv(participant_results_path, index=False)

    print(f"Participant results saved to: {participant_results_path}")

    if save_plots:
        plt.figure()
        plt.scatter(true_vals, pred_vals)

        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        plt.plot([min_val, max_val], [min_val, max_val])

        x_fit = np.linspace(min_val, max_val, 100)
        y_fit = a * x_fit + b
        plt.plot(x_fit, y_fit)

        plt.xlabel("True VO2max")
        plt.ylabel("Predicted VO2max")
        plt.title(f"Held-out {activity_name(heldout_activity)}")
        plt.tight_layout()
        plt.savefig(
            activity_output_directory / "prediction_scatter.png",
            dpi=300,
        )
        plt.close()

        if all_curves:
            min_len = min(len(curve) for curve in all_curves)
            aligned_curves = [curve[:min_len] for curve in all_curves]
            mean_curve = np.mean(aligned_curves, axis=0)
            std_curve = np.std(aligned_curves, axis=0)

            plt.figure()
            x = np.arange(5, 5 * (len(mean_curve) + 1), 5)
            plt.plot(x, mean_curve)
            plt.fill_between(
                x,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.2,
            )
            plt.xlabel("Number of windows")
            plt.ylabel("MAE")
            plt.title(f"Error curve, {activity_name(heldout_activity)}")
            plt.tight_layout()
            plt.savefig(
                activity_output_directory / "convergence_curve.png",
                dpi=300,
            )
            plt.close()

    return {
        "activity": activity_name(heldout_activity),
        "avg_rmse": avg_rmse,
        "global_rmse": global_rmse,
        "r2": r2,
        "r": r,
        "pearson_p_value": p_value,
        "slope": a,
        "intercept": b,
        "n_subjects": len(results_df),
        "stability_filter": use_stability_filter,
        "stability_threshold": (
            stability_threshold if use_stability_filter else np.nan
        ),
    }


# ==========================================================
# COMMAND-LINE INTERFACE
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run combined leave-one-activity-out and leave-one-subject-out "
            "evaluation for VO2max estimation."
        )
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help=(
            "Directory containing RestOut, FoldClothesOut, SweepOut, "
            "WalkOut, BoxCarryingOut, and CyclingOut feature folders."
        ),
    )
    parser.add_argument(
        "--subject-info",
        required=True,
        type=Path,
        help="Excel file containing Participant ID, HR step test(bpm), and Gender.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory in which models, results, and plots are saved.",
    )
    parser.add_argument(
        "--activities",
        nargs="+",
        type=int,
        choices=HELDOUT_ACTIVITIES,
        default=HELDOUT_ACTIVITIES,
        help="Activity identifiers to evaluate: 0 rest, 1 folding, 2 sweeping, 3 walking, 4 box carrying, 5 cycling.",
    )
    parser.add_argument(
        "--stability-filter",
        choices=["with", "without"],
        default="with",
        help="Retain or omit the established stability filter.",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=DEFAULT_STABILITY_THRESHOLD,
        help="Minimum stable duration in seconds when filtering is enabled.",
    )
    parser.add_argument(
        "--skip-fold-models",
        action="store_true",
        help="Do not save a model for every participant/activity fold.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save prediction and convergence plots as PNG files.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue evaluating other activities if one activity fails.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_data_root = args.data_root.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    vo2_map = load_subject_metadata(args.subject_info.resolve())
    use_stability_filter = args.stability_filter == "with"
    summary = []

    for heldout_activity in args.activities:
        try:
            result = evaluate_activity(
                heldout_activity=heldout_activity,
                base_data_root=base_data_root,
                output_root=output_root,
                vo2_map=vo2_map,
                stability_threshold=args.stability_threshold,
                use_stability_filter=use_stability_filter,
                save_fold_models=not args.skip_fold_models,
                save_plots=args.save_plots,
            )
            if result is not None:
                summary.append(result)
        except Exception as error:
            print(
                f"\nACTIVITY FAILED: {activity_name(heldout_activity)} | {error}"
            )
            if not args.continue_on_error:
                raise

    if not summary:
        raise RuntimeError("No activity evaluation completed successfully.")

    summary_df = pd.DataFrame(summary)
    print("\n================================================")
    print("FINAL SUMMARY")
    print("================================================")
    print(summary_df.to_string(index=False))

    save_path = output_root / "LOSO_LOAO_summary.csv"
    summary_df.to_csv(save_path, index=False)
    print(f"\nSaved to: {save_path}")


if __name__ == "__main__":
    main()
