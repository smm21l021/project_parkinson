"""
ml_models.py
============
Training and evaluation helpers for classical ML baselines.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.5
NON_FEATURE_COLUMNS = {
    "file_path",
    "filename",
    "patient_id",
    "original_patient_id",
    "label",
    "label_name",
    "split",
    "split_origin",
    "dataset",
    "source_dataset",
    "dataset_role",
    "scenario_name",
    "scenario_kind",
    "recording_id",
    "audio_path",
    "processed_path",
    "dataset_root",
    "patient_folder",
    "group_folder",
    "speech_type",
}


def get_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    """Returns numeric ML feature columns only."""
    return [
        column
        for column in feature_df.columns
        if column not in NON_FEATURE_COLUMNS
        and is_numeric_dtype(feature_df[column])
        and column != "label"
    ]


def get_model_spaces(n_jobs: int) -> dict[str, dict[str, object]]:
    """Baseline models and modest hyperparameter grids."""
    return {
        "logreg": {
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "params": {
                "model__C": [0.1, 1.0, 5.0],
                "model__class_weight": [None, "balanced"],
                "model__solver": ["liblinear"],
            },
        },
        "svm": {
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        SVC(
                            probability=True,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "params": {
                "model__C": [0.5, 1.0, 5.0],
                "model__gamma": ["scale", "auto"],
                "model__class_weight": [None, "balanced"],
                "model__kernel": ["rbf"],
            },
        },
        "random_forest": {
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        RandomForestClassifier(
                            random_state=RANDOM_STATE,
                            n_jobs=n_jobs,
                        ),
                    ),
                ]
            ),
            "params": {
                "model__n_estimators": [300],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_leaf": [1, 2, 4],
                "model__class_weight": [None, "balanced"],
            },
        },
    }


def _predict_scores(model, features: pd.DataFrame) -> np.ndarray:
    """Returns positive-class scores for ROC-AUC and thresholding."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        decision = model.decision_function(features)
        decision = np.asarray(decision, dtype=float)
        return (decision - decision.min()) / (decision.max() - decision.min() + 1e-10)
    raise AttributeError("Model does not support probability-like scores.")


def _metric_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    *,
    scenario_name: str,
    model_name: str,
    split_name: str,
    level: str,
    subset_name: str,
) -> dict[str, float | str | int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    try:
        roc_auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
    except ValueError:
        roc_auc = np.nan

    return {
        "scenario": scenario_name,
        "model": model_name,
        "split": split_name,
        "evaluation_level": level,
        "subset": subset_name,
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc) if np.isfinite(roc_auc) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _build_prediction_table(
    source_df: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    scenario_name: str,
    model_name: str,
    split_name: str,
) -> pd.DataFrame:
    prediction_columns = ["filename", "patient_id", "dataset", "label", "split"]
    optional_columns = [column for column in ["source_dataset", "dataset_role"] if column in source_df.columns]
    pred_df = source_df[prediction_columns + optional_columns].copy()
    pred_df["scenario"] = scenario_name
    pred_df["model"] = model_name
    pred_df["score"] = y_score
    pred_df["pred_label"] = (y_score >= DEFAULT_THRESHOLD).astype(int)
    pred_df["true_label"] = y_true
    pred_df["evaluation_source"] = split_name
    return pred_df


def _aggregate_patient_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["scenario", "model", "evaluation_source", "dataset", "patient_id"]
    if "source_dataset" in pred_df.columns:
        group_columns.append("source_dataset")

    grouped = (
        pred_df.groupby(group_columns, as_index=False)
        .agg(
            true_label=("true_label", "first"),
            score=("score", "mean"),
            split=("split", "first"),
        )
    )
    grouped["pred_label"] = (grouped["score"] >= DEFAULT_THRESHOLD).astype(int)
    return grouped


def _evaluate_prediction_tables(
    file_pred_df: pd.DataFrame,
    *,
    scenario_name: str,
    model_name: str,
    split_name: str,
) -> list[dict[str, float | str | int]]:
    metric_rows: list[dict[str, float | str | int]] = []

    metric_rows.append(
        _metric_row(
            y_true=file_pred_df["true_label"].to_numpy(),
            y_pred=file_pred_df["pred_label"].to_numpy(),
            y_score=file_pred_df["score"].to_numpy(),
            scenario_name=scenario_name,
            model_name=model_name,
            split_name=split_name,
            level="file",
            subset_name="all",
        )
    )

    for dataset_name, subset in file_pred_df.groupby("dataset"):
        metric_rows.append(
            _metric_row(
                y_true=subset["true_label"].to_numpy(),
                y_pred=subset["pred_label"].to_numpy(),
                y_score=subset["score"].to_numpy(),
                scenario_name=scenario_name,
                model_name=model_name,
                split_name=split_name,
                level="file",
                subset_name=str(dataset_name),
            )
        )

    patient_pred_df = _aggregate_patient_predictions(file_pred_df)
    metric_rows.append(
        _metric_row(
            y_true=patient_pred_df["true_label"].to_numpy(),
            y_pred=patient_pred_df["pred_label"].to_numpy(),
            y_score=patient_pred_df["score"].to_numpy(),
            scenario_name=scenario_name,
            model_name=model_name,
            split_name=split_name,
            level="patient",
            subset_name="all",
        )
    )

    for dataset_name, subset in patient_pred_df.groupby("dataset"):
        metric_rows.append(
            _metric_row(
                y_true=subset["true_label"].to_numpy(),
                y_pred=subset["pred_label"].to_numpy(),
                y_score=subset["score"].to_numpy(),
                scenario_name=scenario_name,
                model_name=model_name,
                split_name=split_name,
                level="patient",
                subset_name=str(dataset_name),
            )
        )

    return metric_rows


def _save_json(path: Path, payload: dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def _make_cv(y_train: np.ndarray) -> StratifiedKFold:
    class_counts = np.bincount(y_train.astype(int))
    non_zero_counts = class_counts[class_counts > 0]
    if non_zero_counts.size < 2:
        raise ValueError("Training split must contain both classes.")

    min_class_size = int(non_zero_counts.min())
    if min_class_size < 2:
        raise ValueError("Training split needs at least two samples in each class for CV.")

    n_splits = min(5, min_class_size)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def train_ml_baselines(
    feature_df: pd.DataFrame,
    *,
    model_dir: str | Path,
    result_dir: str | Path,
    scenario_name: str,
    final_fit_splits: tuple[str, ...] = ("train", "val"),
    scoring: str = "f1",
    n_jobs: int = -1,
) -> dict[str, object]:
    """
    Trains baseline classical ML models, compares them on validation,
    and refits the selected model using the requested final-fit split policy.
    """
    model_dir = Path(model_dir)
    result_dir = Path(result_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    required_cols = {"split", "label", "patient_id", "filename", "dataset"}
    missing = required_cols - set(feature_df.columns)
    if missing:
        raise ValueError(f"Feature table missing required columns: {sorted(missing)}")

    feature_columns = get_feature_columns(feature_df)
    if not feature_columns:
        raise ValueError("No numeric acoustic feature columns were found.")

    train_df = feature_df[feature_df["split"] == "train"].reset_index(drop=True)
    val_df = feature_df[feature_df["split"] == "val"].reset_index(drop=True)
    test_df = feature_df[feature_df["split"] == "test"].reset_index(drop=True)
    final_fit_df = feature_df[feature_df["split"].isin(final_fit_splits)].reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"Scenario {scenario_name} requires non-empty train/val/test splits "
            f"(got train={len(train_df)}, val={len(val_df)}, test={len(test_df)})."
        )
    if final_fit_df.empty:
        raise ValueError(f"Scenario {scenario_name} produced an empty final-fit split.")

    X_train = train_df[feature_columns]
    y_train = train_df["label"].to_numpy()
    X_val = val_df[feature_columns]
    y_val = val_df["label"].to_numpy()
    X_test = test_df[feature_columns]
    y_test = test_df["label"].to_numpy()
    X_final_fit = final_fit_df[feature_columns]
    y_final_fit = final_fit_df["label"].to_numpy()

    model_spaces = get_model_spaces(n_jobs=n_jobs)
    cv = _make_cv(y_train)
    cv_n_splits = cv.get_n_splits()

    all_metrics: list[dict[str, float | str | int]] = []
    all_predictions: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []

    for model_name, spec in model_spaces.items():
        logger.info("Training model for %s: %s", scenario_name, model_name)

        search = GridSearchCV(
            estimator=spec["pipeline"],
            param_grid=spec["params"],
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            verbose=0,
        )
        search.fit(X_train, y_train)

        best_estimator = search.best_estimator_
        best_params = search.best_params_
        best_score = float(search.best_score_)

        cv_results = pd.DataFrame(search.cv_results_)
        cv_results["scenario"] = scenario_name
        cv_results.to_csv(result_dir / f"ml_cv_results_{model_name}.csv", index=False)

        joblib.dump(
            {
                "scenario": scenario_name,
                "model_name": model_name,
                "feature_columns": feature_columns,
                "best_params": best_params,
                "cv_best_score": best_score,
                "cv_n_splits": cv_n_splits,
                "estimator": best_estimator,
            },
            model_dir / f"ml_{model_name}.joblib",
        )

        val_scores = _predict_scores(best_estimator, X_val)
        val_pred_df = _build_prediction_table(
            val_df,
            y_true=y_val,
            y_score=val_scores,
            scenario_name=scenario_name,
            model_name=model_name,
            split_name="val",
        )
        test_scores = _predict_scores(best_estimator, X_test)
        test_pred_df = _build_prediction_table(
            test_df,
            y_true=y_test,
            y_score=test_scores,
            scenario_name=scenario_name,
            model_name=model_name,
            split_name="test",
        )

        all_predictions.extend([val_pred_df, test_pred_df])
        all_metrics.extend(
            _evaluate_prediction_tables(
                val_pred_df,
                scenario_name=scenario_name,
                model_name=model_name,
                split_name="val",
            )
        )
        all_metrics.extend(
            _evaluate_prediction_tables(
                test_pred_df,
                scenario_name=scenario_name,
                model_name=model_name,
                split_name="test",
            )
        )

        patient_val_metrics = [
            row
            for row in _evaluate_prediction_tables(
                val_pred_df,
                scenario_name=scenario_name,
                model_name=model_name,
                split_name="val",
            )
            if row["evaluation_level"] == "patient" and row["subset"] == "all"
        ][0]
        selection_rows.append(
            {
                "scenario": scenario_name,
                "model": model_name,
                "cv_best_score": best_score,
                "cv_n_splits": cv_n_splits,
                "val_patient_f1": patient_val_metrics["f1"],
                "val_patient_roc_auc": patient_val_metrics["roc_auc"],
                "best_params": json.dumps(best_params, ensure_ascii=False),
            }
        )

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    selection_df = pd.DataFrame(selection_rows).sort_values(
        by=["val_patient_f1", "val_patient_roc_auc"],
        ascending=False,
    )

    metrics_df.to_csv(result_dir / "ml_metrics_summary.csv", index=False)
    predictions_df.to_csv(result_dir / "ml_predictions_all.csv", index=False)
    selection_df.to_csv(result_dir / "ml_model_selection.csv", index=False)

    best_model_name = str(selection_df.iloc[0]["model"])
    logger.info("Selected best model for %s: %s", scenario_name, best_model_name)

    best_bundle = joblib.load(model_dir / f"ml_{best_model_name}.joblib")
    final_model = clone(best_bundle["estimator"])
    final_model.fit(X_final_fit, y_final_fit)

    final_scores = _predict_scores(final_model, X_test)
    final_pred_df = _build_prediction_table(
        test_df,
        y_true=y_test,
        y_score=final_scores,
        scenario_name=scenario_name,
        model_name=f"{best_model_name}_final",
        split_name="test",
    )
    final_metrics = _evaluate_prediction_tables(
        final_pred_df,
        scenario_name=scenario_name,
        model_name=f"{best_model_name}_final",
        split_name="test",
    )

    final_metrics_df = pd.DataFrame(final_metrics)
    final_metrics_df.to_csv(result_dir / "ml_final_test_metrics.csv", index=False)
    final_pred_df.to_csv(result_dir / "ml_final_test_predictions.csv", index=False)

    joblib.dump(
        {
            "scenario": scenario_name,
            "model_name": best_model_name,
            "feature_columns": feature_columns,
            "selected_params": best_bundle["best_params"],
            "final_fit_splits": list(final_fit_splits),
            "estimator": final_model,
        },
        model_dir / "ml_selected_final_model.joblib",
    )

    importance = permutation_importance(
        final_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="f1",
        n_jobs=n_jobs,
    )
    importance_df = pd.DataFrame(
        {
            "scenario": scenario_name,
            "feature": feature_columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(result_dir / "ml_permutation_importance.csv", index=False)

    summary = {
        "scenario": scenario_name,
        "feature_count": len(feature_columns),
        "models_trained": list(model_spaces.keys()),
        "selected_model": best_model_name,
        "selection_metric": "validation patient-level F1",
        "cv_n_splits": cv_n_splits,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_final_fit": int(len(final_fit_df)),
        "final_fit_splits": list(final_fit_splits),
    }
    _save_json(result_dir / "ml_training_summary.json", summary)

    return summary
