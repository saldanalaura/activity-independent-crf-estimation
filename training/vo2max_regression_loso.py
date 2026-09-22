#!/usr/bin/env python3

"""
Batch LOSO evaluation for multiple observation-window durations.

For every requested window size, this script evaluates two configurations:

1. Without stability filtering: every available window is retained.
2. With stability filtering: only windows with stability_max >= 60 seconds
   are retained.

All activities remain in both training and testing. In each LOSO fold, the
held-out participant is absent from training. The final participant-level
VO2max estimate is the arithmetic mean of all valid window predictions.

The script does not generate plots. It saves fold models, fold normalization
metadata, one final all-participant model per configuration, participant-level
results, and one consolidated metrics table.

Example:
    python vo2max_regression_loso.py \
        --data-root-template "data/AllFeatures116_{window_size}s" \
        --subject-info data/SubjectsInfo.xlsx --output-dir outputs/loso
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


# ==========================================================
# CONFIGURATION
# ==========================================================

DEFAULT_WINDOW_SIZES = [2, 5, 10, 30, 60, 90]

STABILITY_THRESHOLD = 60.0
LOW_TARGET_PERCENTILE = 20
HIGH_TARGET_PERCENTILE = 80
EXTREME_TARGET_WEIGHT = 10.0

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
# PATH AND LABEL HELPERS
# ==========================================================

def get_data_root(data_root_template, window_size):
    """Resolve a feature directory from a template containing {window_size}."""
    try:
        return Path(data_root_template.format(window_size=window_size))
    except KeyError as error:
        raise ValueError(
            "--data-root-template may only use the {window_size} field."
        ) from error


def stability_label(use_stability_filter):
    if use_stability_filter:
        return f"with_threshold_{STABILITY_THRESHOLD:g}s"
    return "without_threshold"


def get_configuration_directory(
    experiment_root,
    window_size,
    use_stability_filter,
):
    return (
        experiment_root
        / f"window_{window_size}s"
        / stability_label(use_stability_filter)
    )


def subject_id_from_file(file_path):
    return int(file_path.stem.split("-")[0])


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

    hr_map = dict(
        zip(
            participant_ids,
            subjects_info["HR step test(bpm)"],
        )
    )

    sex_map = dict(
        zip(
            participant_ids,
            subjects_info["Gender"],
        )
    )

    return {
        subject_id: compute_vo2max(
            hr_map[subject_id],
            sex_map[subject_id],
        )
        for subject_id in hr_map
    }


# ==========================================================
# LOAD ONE COMPLETE CONFIGURATION
# ==========================================================

def load_configuration_data(
    data_root,
    vo2_map,
    use_stability_filter,
):
    """Load one window-size/filter configuration into memory once."""
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_root}")

    files = sorted(data_root.glob("*_hr_features.npz"))

    if not files:
        raise FileNotFoundError(
            f"No *_hr_features.npz files were found in: {data_root}"
        )

    X_all = []
    y_all = []
    subject_ids_all = []
    expected_feature_count = None

    for file in files:
        subject_id = subject_id_from_file(file)

        if subject_id not in vo2_map:
            raise KeyError(
                f"Participant {subject_id} from {file.name} is not present "
                "in the subject-information file."
            )

        data = np.load(file, allow_pickle=True)

        if "X" not in data:
            raise KeyError(f"Array 'X' was not found in {file.name}.")

        X = data["X"]

        if X.ndim != 2:
            raise ValueError(
                f"X must be two-dimensional in {file.name}; got {X.shape}."
            )

        if expected_feature_count is None:
            expected_feature_count = X.shape[1]
        elif X.shape[1] != expected_feature_count:
            raise ValueError(
                f"Feature count mismatch in {file.name}: expected "
                f"{expected_feature_count}, found {X.shape[1]}."
            )

        if use_stability_filter:
            if "stability_max" not in data:
                raise KeyError(
                    f"Array 'stability_max' was not found in {file.name}."
                )

            stability = data["stability_max"]

            if len(X) != len(stability):
                raise ValueError(
                    f"X and stability_max have different lengths in "
                    f"{file.name}."
                )

            mask = stability >= STABILITY_THRESHOLD
        else:
            mask = np.ones(len(X), dtype=bool)

        retained_count = int(np.sum(mask))

        if retained_count == 0:
            print(
                f"  Participant {subject_id:03d}: no valid windows; skipped"
            )
            continue

        X_subject = X[mask]
        y_subject = np.full(
            retained_count,
            vo2_map[subject_id],
            dtype=float,
        )

        X_all.append(X_subject)
        y_all.append(y_subject)
        subject_ids_all.append(
            np.full(retained_count, subject_id, dtype=int)
        )

    if not X_all:
        raise RuntimeError(
            f"No valid windows remained for configuration: {data_root}"
        )

    return (
        np.vstack(X_all),
        np.concatenate(y_all),
        np.concatenate(subject_ids_all),
    )


# ==========================================================
# TARGET NORMALIZATION AND WEIGHTING
# ==========================================================

def prepare_training_target(y_train):
    """Normalize the fold target and assign the established tail weights."""
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))

    if y_std < 1e-6:
        y_std = 1.0

    y_train_normalized = (y_train - y_mean) / y_std

    low_threshold = float(
        np.percentile(y_train, LOW_TARGET_PERCENTILE)
    )
    high_threshold = float(
        np.percentile(y_train, HIGH_TARGET_PERCENTILE)
    )

    weights = np.ones_like(y_train, dtype=float)
    weights[y_train < low_threshold] = EXTREME_TARGET_WEIGHT
    weights[y_train > high_threshold] = EXTREME_TARGET_WEIGHT

    target_info = {
        "y_mean": y_mean,
        "y_std": y_std,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
    }

    return y_train_normalized, weights, target_info


def save_normalization_metadata(
    path,
    target_info,
    n_features,
    window_size,
    use_stability_filter,
    training_scope,
    heldout_subject=-1,
    n_training_windows=-1,
):
    """Save the inverse-transformation values and configuration metadata."""
    np.savez(
        path,
        y_mean=target_info["y_mean"],
        y_std=target_info["y_std"],
        low_threshold=target_info["low_threshold"],
        high_threshold=target_info["high_threshold"],
        low_percentile=LOW_TARGET_PERCENTILE,
        high_percentile=HIGH_TARGET_PERCENTILE,
        extreme_weight=EXTREME_TARGET_WEIGHT,
        n_features=n_features,
        window_size_seconds=window_size,
        stability_filter=use_stability_filter,
        stability_threshold=(
            STABILITY_THRESHOLD if use_stability_filter else np.nan
        ),
        heldout_subject=heldout_subject,
        n_training_windows=n_training_windows,
        aggregation="mean_of_all_valid_participant_windows",
        training_scope=training_scope,
    )


# ==========================================================
# ONE LOSO FOLD
# ==========================================================

def train_loso_fold(
    test_subject,
    X_all,
    y_all,
    subject_ids,
    window_size,
    use_stability_filter,
    configuration_directory,
    save_fold_models,
):
    train_mask = subject_ids != test_subject
    test_mask = subject_ids == test_subject

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    X_test = X_all[test_mask]
    y_test = y_all[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        return None

    y_train_normalized, weights, target_info = prepare_training_target(
        y_train
    )

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train,
        y_train_normalized,
        sample_weight=weights,
    )

    model_path = ""
    normalization_path = ""

    if save_fold_models:
        fold_directory = configuration_directory / "fold_models"
        fold_directory.mkdir(parents=True, exist_ok=True)

        model_path_object = (
            fold_directory
            / f"heldout_subject_{test_subject:03d}_model.json"
        )
        normalization_path_object = (
            fold_directory
            / f"heldout_subject_{test_subject:03d}_normalization.npz"
        )

        model.save_model(str(model_path_object))

        save_normalization_metadata(
            path=normalization_path_object,
            target_info=target_info,
            n_features=X_train.shape[1],
            window_size=window_size,
            use_stability_filter=use_stability_filter,
            heldout_subject=test_subject,
            n_training_windows=len(X_train),
            training_scope="all_activities_except_heldout_subject",
        )

        model_path = str(model_path_object)
        normalization_path = str(normalization_path_object)

    predictions_normalized = model.predict(X_test)
    predictions = (
        predictions_normalized * target_info["y_std"]
        + target_info["y_mean"]
    )

    true_subject = float(y_test[0])

    # One participant-level prediction from all valid windows.
    predicted_subject = float(np.mean(predictions))

    residual = predicted_subject - true_subject

    # There is one participant-level prediction in each LOSO fold.
    fold_rmse = float(np.sqrt(residual ** 2))

    return {
        "subject": test_subject,
        "true": true_subject,
        "pred": predicted_subject,
        "rmse": fold_rmse,
        "residual": residual,
        "absolute_residual": abs(residual),
        "n_test_windows": len(X_test),
        "model_path": model_path,
        "normalization_path": normalization_path,
    }


# ==========================================================
# FINAL ALL-PARTICIPANT MODEL FOR ONE CONFIGURATION
# ==========================================================

def train_and_save_final_model(
    X_all,
    y_all,
    window_size,
    use_stability_filter,
    configuration_directory,
):
    y_normalized, weights, target_info = prepare_training_target(y_all)

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_all,
        y_normalized,
        sample_weight=weights,
    )

    model_path = configuration_directory / "final_all_subjects_model.json"
    normalization_path = (
        configuration_directory
        / "final_all_subjects_normalization.npz"
    )

    model.save_model(str(model_path))

    save_normalization_metadata(
        path=normalization_path,
        target_info=target_info,
        n_features=X_all.shape[1],
        window_size=window_size,
        use_stability_filter=use_stability_filter,
        heldout_subject=-1,
        n_training_windows=len(X_all),
        training_scope="all_subjects_all_activities",
    )

    return model_path, normalization_path


# ==========================================================
# EVALUATE ONE WINDOW-SIZE/FILTER CONFIGURATION
# ==========================================================

def evaluate_configuration(
    window_size,
    use_stability_filter,
    vo2_map,
    data_root_template,
    experiment_root,
    save_fold_models,
    save_final_models,
):
    data_root = get_data_root(data_root_template, window_size)
    configuration_directory = get_configuration_directory(
        experiment_root,
        window_size,
        use_stability_filter,
    )
    configuration_directory.mkdir(parents=True, exist_ok=True)

    label = stability_label(use_stability_filter)

    print("\n============================================================")
    print(f"WINDOW: {window_size} s | CONFIGURATION: {label}")
    print("============================================================")
    print(f"Input: {data_root}")

    X_all, y_all, subject_ids = load_configuration_data(
        data_root=data_root,
        vo2_map=vo2_map,
        use_stability_filter=use_stability_filter,
    )

    subjects = sorted(np.unique(subject_ids).tolist())
    results = []

    for fold_number, test_subject in enumerate(subjects, start=1):
        print(
            f"  Fold {fold_number:02d}/{len(subjects):02d}: "
            f"participant {test_subject:03d}"
        )

        result = train_loso_fold(
            test_subject=test_subject,
            X_all=X_all,
            y_all=y_all,
            subject_ids=subject_ids,
            window_size=window_size,
            use_stability_filter=use_stability_filter,
            configuration_directory=configuration_directory,
            save_fold_models=save_fold_models,
        )

        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError(
            f"No LOSO folds were completed for {window_size} s, {label}."
        )

    results_df = pd.DataFrame(results).sort_values("subject")
    participant_results_path = (
        configuration_directory / "subject_results.csv"
    )
    results_df.to_csv(participant_results_path, index=False)

    true_values = results_df["true"].to_numpy(dtype=float)
    predicted_values = results_df["pred"].to_numpy(dtype=float)

    avg_rmse = float(results_df["rmse"].mean())
    global_rmse = float(
        np.sqrt(mean_squared_error(true_values, predicted_values))
    )
    r2 = float(r2_score(true_values, predicted_values))
    correlation, correlation_p_value = pearsonr(
        true_values,
        predicted_values,
    )
    slope, intercept = np.polyfit(true_values, predicted_values, 1)

    final_model_path = ""
    final_normalization_path = ""

    if save_final_models:
        final_model, final_normalization = train_and_save_final_model(
            X_all=X_all,
            y_all=y_all,
            window_size=window_size,
            use_stability_filter=use_stability_filter,
            configuration_directory=configuration_directory,
        )
        final_model_path = str(final_model)
        final_normalization_path = str(final_normalization)

    metrics = {
        "window_seconds": window_size,
        "stability_filter": use_stability_filter,
        "configuration": label,
        "avg_rmse": avg_rmse,
        "global_rmse": global_rmse,
        "r2": r2,
        "r": float(correlation),
        "pearson_p_value": float(correlation_p_value),
        "slope": float(slope),
        "intercept": float(intercept),
        "n_subjects": len(results_df),
        "n_windows": len(X_all),
        "n_features": X_all.shape[1],
        "participant_results_path": str(participant_results_path),
        "final_model_path": final_model_path,
        "final_normalization_path": final_normalization_path,
    }

    pd.DataFrame([metrics]).to_csv(
        configuration_directory / "metrics.csv",
        index=False,
    )

    print(
        f"Completed: Avg RMSE={avg_rmse:.4f}, "
        f"Global RMSE={global_rmse:.4f}, "
        f"R2={r2:.4f}, r={float(correlation):.4f}"
    )

    return metrics


# ==========================================================
# COMMAND-LINE INTERFACE
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run participant-level LOSO evaluation for one or more "
            "VO2max observation-window durations."
        )
    )
    parser.add_argument(
        "--data-root-template",
        required=True,
        help=(
            "Feature-directory template containing {window_size}, for "
            "example data/AllFeatures116_{window_size}s."
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
        help="Directory in which models, fold results, and summaries are saved.",
    )
    parser.add_argument(
        "--window-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_WINDOW_SIZES,
        help="Observation-window durations in seconds.",
    )
    parser.add_argument(
        "--stability-configurations",
        choices=["both", "without", "with"],
        default="both",
        help="Evaluate both stability configurations or only one.",
    )
    parser.add_argument(
        "--skip-fold-models",
        action="store_true",
        help="Do not save the model from every LOSO fold.",
    )
    parser.add_argument(
        "--skip-final-models",
        action="store_true",
        help="Do not train and save all-participant models.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with other configurations if one configuration fails.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_root = args.output_dir.resolve()
    experiment_root.mkdir(parents=True, exist_ok=True)

    vo2_map = load_subject_metadata(args.subject_info.resolve())
    all_metrics = []
    failed_configurations = []

    stability_options = {
        "both": [False, True],
        "without": [False],
        "with": [True],
    }[args.stability_configurations]

    for window_size in args.window_sizes:
        for use_stability_filter in stability_options:
            try:
                metrics = evaluate_configuration(
                    window_size=window_size,
                    use_stability_filter=use_stability_filter,
                    vo2_map=vo2_map,
                    data_root_template=args.data_root_template,
                    experiment_root=experiment_root,
                    save_fold_models=not args.skip_fold_models,
                    save_final_models=not args.skip_final_models,
                )
                all_metrics.append(metrics)
            except Exception as error:
                label = stability_label(use_stability_filter)
                message = f"{window_size} s | {label} | {error}"
                failed_configurations.append(message)
                print(f"\nCONFIGURATION FAILED: {message}")
                if not args.continue_on_error:
                    raise

    if not all_metrics:
        raise RuntimeError("Every requested configuration failed.")

    summary_df = pd.DataFrame(all_metrics).sort_values(
        ["window_seconds", "stability_filter"]
    )

    summary_path = experiment_root / "LOSO_window_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    display_columns = [
        "window_seconds",
        "configuration",
        "avg_rmse",
        "global_rmse",
        "r2",
        "r",
        "slope",
        "intercept",
        "n_subjects",
        "n_windows",
    ]

    print("\n\n============================================================")
    print("FINAL METRICS: ALL WINDOW AND THRESHOLD CONFIGURATIONS")
    print("============================================================")
    print(
        summary_df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print(f"\nComplete summary saved to: {summary_path}")

    if failed_configurations:
        print("\nConfigurations that did not complete:")
        for message in failed_configurations:
            print(f"  - {message}")


if __name__ == "__main__":
    main()
