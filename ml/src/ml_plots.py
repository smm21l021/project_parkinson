"""
ml_plots.py
===========
Builds the final set of ML plots used in the project.

Without arguments, the script rebuilds plots for every scenario that already
has saved training outputs under ``ml/data/scenarios``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.tree import plot_tree

from ml_scenarios import (
    ML_ROOT,
    canonicalize_scenario_name,
    get_scenario_catalog,
    resolve_existing_scenario_dir,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"

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

MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "svm": "SVM",
    "random_forest": "Random Forest",
}


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def _save_fig(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", output_path)


def _feature_columns(feature_df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in feature_df.columns
        if column not in NON_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(feature_df[column])
        and column != "label"
    ]


def _scenario_title_suffix(scenario_label: str | None) -> str:
    return f" | {scenario_label}" if scenario_label else ""


def _aggregate_patient_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["scenario", "model", "evaluation_source", "patient_id"]
    if "dataset" in pred_df.columns:
        group_columns.append("dataset")

    grouped = (
        pred_df.groupby(group_columns, as_index=False)
        .agg(
            true_label=("true_label", "first"),
            score=("score", "mean"),
        )
    )
    grouped["pred_label"] = (grouped["score"] >= 0.5).astype(int)
    return grouped


def plot_confusion_matrices(
    predictions_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    scenario_label: str | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["val", "test"]:
        split_subset = predictions_df[predictions_df["evaluation_source"] == split_name]
        for model_name, subset in split_subset.groupby("model"):
            fig, ax = plt.subplots(figsize=(5.2, 4.6))
            ConfusionMatrixDisplay.from_predictions(
                y_true=subset["true_label"],
                y_pred=subset["pred_label"],
                labels=[0, 1],
                display_labels=["Healthy", "Parkinson"],
                cmap="Blues",
                colorbar=False,
                ax=ax,
            )
            model_label = MODEL_LABELS.get(model_name, model_name)
            ax.set_title(
                f"{model_label}: confusion matrix ({split_name})"
                f"{_scenario_title_suffix(scenario_label)}"
            )
            _save_fig(fig, output_dir / f"ml_confusion_{split_name}_{model_name}.png")


def plot_logreg_top20(
    models_dir: str | Path,
    output_dir: str | Path,
    *,
    scenario_label: str | None = None,
) -> None:
    bundle = joblib.load(Path(models_dir) / "ml_logreg.joblib")
    model = bundle["estimator"].named_steps["model"]
    coef_df = pd.DataFrame(
        {
            "feature": bundle["feature_columns"],
            "score": np.abs(model.coef_.ravel()),
        }
    ).sort_values("score", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    sns.barplot(
        data=coef_df,
        x="score",
        y="feature",
        hue="feature",
        legend=False,
        palette="Blues_r",
        ax=ax,
    )
    ax.set_title(
        "Logistic Regression: top-20 features by |coef|"
        f"{_scenario_title_suffix(scenario_label)}"
    )
    ax.set_xlabel("|coef|")
    ax.set_ylabel("")
    _save_fig(fig, Path(output_dir) / "ml_logreg_top20_coefficients.png")


def plot_logreg_cv_heatmap(
    results_dir: str | Path,
    output_dir: str | Path,
    *,
    scenario_label: str | None = None,
) -> None:
    cv_df = _read_csv(Path(results_dir) / "ml_cv_results_logreg.csv")
    cv_df = cv_df.copy()
    cv_df["class_weight_name"] = cv_df["param_model__class_weight"].fillna("none").astype(str)
    cv_df["C"] = cv_df["param_model__C"].astype(float)

    pivot = cv_df.pivot_table(
        index="class_weight_name",
        columns="C",
        values="mean_test_score",
        aggfunc="mean",
    )
    pivot = pivot.reindex(index=["none", "balanced"])

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues", ax=ax)
    ax.set_title(
        "Logistic Regression CV F1"
        f"{_scenario_title_suffix(scenario_label)}"
    )
    ax.set_xlabel("C")
    ax.set_ylabel("class_weight")
    _save_fig(fig, Path(output_dir) / "ml_logreg_cv_heatmap.png")


def plot_svm_top20_permutation(
    models_dir: str | Path,
    feature_table_path: str | Path,
    output_dir: str | Path,
    *,
    scenario_label: str | None = None,
    n_jobs: int = -1,
    n_repeats: int = 8,
) -> None:
    bundle = joblib.load(Path(models_dir) / "ml_svm.joblib")
    feature_df = _read_csv(feature_table_path)
    feature_columns = bundle["feature_columns"]
    test_df = feature_df[feature_df["split"] == "test"].reset_index(drop=True)

    importance = permutation_importance(
        bundle["estimator"],
        test_df[feature_columns],
        test_df["label"].to_numpy(),
        n_repeats=n_repeats,
        random_state=42,
        scoring="f1",
        n_jobs=n_jobs,
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "score": importance.importances_mean,
        }
    ).sort_values("score", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    sns.barplot(
        data=importance_df,
        x="score",
        y="feature",
        hue="feature",
        legend=False,
        palette="Oranges_r",
        ax=ax,
    )
    ax.set_title(
        "SVM: top-20 features by permutation importance"
        f"{_scenario_title_suffix(scenario_label)}"
    )
    ax.set_xlabel("Permutation importance (F1 drop)")
    ax.set_ylabel("")
    _save_fig(fig, Path(output_dir) / "ml_svm_top20_permutation_importance.png")


def plot_rf_top20(
    models_dir: str | Path,
    output_dir: str | Path,
    *,
    scenario_label: str | None = None,
) -> None:
    bundle = joblib.load(Path(models_dir) / "ml_random_forest.joblib")
    model = bundle["estimator"].named_steps["model"]
    importance_df = pd.DataFrame(
        {
            "feature": bundle["feature_columns"],
            "score": model.feature_importances_,
        }
    ).sort_values("score", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    sns.barplot(
        data=importance_df,
        x="score",
        y="feature",
        hue="feature",
        legend=False,
        palette="Greens_r",
        ax=ax,
    )
    ax.set_title(
        "Random Forest: top-20 features by feature importance"
        f"{_scenario_title_suffix(scenario_label)}"
    )
    ax.set_xlabel("Gini importance")
    ax.set_ylabel("")
    _save_fig(fig, Path(output_dir) / "ml_rf_top20_feature_importance.png")


def plot_svm_cv_heatmap(
    results_dir: str | Path,
    output_dir: str | Path,
    *,
    scenario_label: str | None = None,
) -> None:
    cv_df = _read_csv(Path(results_dir) / "ml_cv_results_svm.csv")
    cv_df = cv_df.copy()
    cv_df["class_weight_name"] = cv_df["param_model__class_weight"].fillna("none").astype(str)
    cv_df["C"] = cv_df["param_model__C"].astype(float)
    cv_df["gamma"] = cv_df["param_model__gamma"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True)
    for ax, class_weight_name in zip(axes, ["none", "balanced"], strict=False):
        subset = cv_df[cv_df["class_weight_name"] == class_weight_name]
        pivot = subset.pivot_table(index="gamma", columns="C", values="mean_test_score", aggfunc="mean")
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="PuBu", ax=ax)
        ax.set_title(
            f"SVM CV F1 | class_weight={class_weight_name}"
            f"{_scenario_title_suffix(scenario_label)}"
        )
        ax.set_xlabel("C")
        ax.set_ylabel("gamma")

    _save_fig(fig, Path(output_dir) / "ml_svm_cv_heatmap.png")


def plot_rf_cv_heatmap(
    results_dir: str | Path,
    output_dir: str | Path,
    *,
    scenario_label: str | None = None,
) -> None:
    cv_df = _read_csv(Path(results_dir) / "ml_cv_results_random_forest.csv")
    cv_df = cv_df.copy()
    cv_df["class_weight_name"] = cv_df["param_model__class_weight"].fillna("none").astype(str)
    cv_df["max_depth"] = cv_df["param_model__max_depth"].astype(str)
    cv_df["min_samples_leaf"] = cv_df["param_model__min_samples_leaf"].astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True)
    for ax, class_weight_name in zip(axes, ["none", "balanced"], strict=False):
        subset = cv_df[cv_df["class_weight_name"] == class_weight_name]
        pivot = subset.pivot_table(
            index="min_samples_leaf",
            columns="max_depth",
            values="mean_test_score",
            aggfunc="mean",
        )
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Greens", ax=ax)
        ax.set_title(
            f"RF CV F1 | class_weight={class_weight_name}"
            f"{_scenario_title_suffix(scenario_label)}"
        )
        ax.set_xlabel("max_depth")
        ax.set_ylabel("min_samples_leaf")

    _save_fig(fig, Path(output_dir) / "ml_rf_cv_heatmap.png")


def plot_rf_tree_visualization(
    models_dir: str | Path,
    output_dir: str | Path,
    *,
    scenario_label: str | None = None,
) -> None:
    bundle = joblib.load(Path(models_dir) / "ml_random_forest.joblib")
    pipeline = bundle["estimator"]
    forest = pipeline.named_steps["model"]
    feature_names = bundle["feature_columns"]

    depths = [tree.tree_.max_depth for tree in forest.estimators_]
    chosen_idx = int(np.argmin(depths))
    estimator = forest.estimators_[chosen_idx]

    fig, ax = plt.subplots(figsize=(35, 14))
    plot_tree(
        estimator,
        feature_names=feature_names,
        class_names=["Healthy", "Parkinson"],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=12,
        impurity=False,
        proportion=True,
        ax=ax,
    )
    ax.set_title(
        "Random Forest: visualization of one tree from the ensemble "
        f"(tree #{chosen_idx}, first 4 levels shown)"
        f"{_scenario_title_suffix(scenario_label)}"
    )
    _save_fig(fig, Path(output_dir) / "ml_rf_tree_visualization.png")


def plot_roc_test_all_models(
    predictions_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    scenario_label: str | None = None,
) -> None:
    patient_df = _aggregate_patient_predictions(predictions_df)
    test_df = patient_df[patient_df["evaluation_source"] == "test"].copy()
    if test_df.empty:
        logger.warning("Skipping ROC plot because no test predictions were found.")
        return

    fig, ax = plt.subplots(figsize=(6.3, 5.4))
    plotted_any = False

    for model_name, subset in test_df.groupby("model"):
        y_true = subset["true_label"].to_numpy()
        y_score = subset["score"].to_numpy()
        if len(np.unique(y_true)) < 2:
            logger.warning(
                "Skipping ROC curve for %s in %s because test labels contain one class only.",
                model_name,
                scenario_label or "scenario",
            )
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_value = roc_auc_score(y_true, y_score)
        model_label = MODEL_LABELS.get(model_name, model_name)
        ax.plot(fpr, tpr, linewidth=2.0, label=f"{model_label} (AUC={auc_value:.3f})")
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        logger.warning("Skipping ROC plot because no valid model curves were available.")
        return

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, label="Chance")
    ax.set_title(
        "ROC curves on patient-level test predictions"
        f"{_scenario_title_suffix(scenario_label)}"
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right", frameon=True)
    _save_fig(fig, Path(output_dir) / "ml_roc_test_all_models.png")


def build_ml_plots(
    *,
    feature_table_path: str | Path,
    predictions_path: str | Path,
    results_dir: str | Path,
    models_dir: str | Path,
    output_dir: str | Path,
    scenario_label: str | None = None,
    n_jobs: int = -1,
) -> None:
    predictions_df = _read_csv(predictions_path)

    plot_confusion_matrices(
        predictions_df=predictions_df,
        output_dir=output_dir,
        scenario_label=scenario_label,
    )
    plot_logreg_top20(models_dir=models_dir, output_dir=output_dir, scenario_label=scenario_label)
    plot_logreg_cv_heatmap(results_dir=results_dir, output_dir=output_dir, scenario_label=scenario_label)
    plot_svm_top20_permutation(
        models_dir=models_dir,
        feature_table_path=feature_table_path,
        output_dir=output_dir,
        scenario_label=scenario_label,
        n_jobs=n_jobs,
    )
    plot_rf_top20(models_dir=models_dir, output_dir=output_dir, scenario_label=scenario_label)
    plot_svm_cv_heatmap(results_dir=results_dir, output_dir=output_dir, scenario_label=scenario_label)
    plot_rf_cv_heatmap(results_dir=results_dir, output_dir=output_dir, scenario_label=scenario_label)
    plot_rf_tree_visualization(models_dir=models_dir, output_dir=output_dir, scenario_label=scenario_label)
    plot_roc_test_all_models(
        predictions_df=predictions_df,
        output_dir=output_dir,
        scenario_label=scenario_label,
    )


def _scenario_dir(scenario_name: str) -> Path:
    return resolve_existing_scenario_dir(scenario_name)


def _build_for_existing_scenario(scenario_name: str, *, n_jobs: int) -> None:
    scenario_dir = _scenario_dir(scenario_name)
    feature_table_path = scenario_dir / "features" / "ml_feature_table.csv"
    predictions_path = scenario_dir / "results" / "ml_predictions_all.csv"
    results_dir = scenario_dir / "results"
    models_dir = scenario_dir / "models"
    output_dir = scenario_dir / "plots"

    required = [feature_table_path, predictions_path, results_dir, models_dir]
    if not all(path.exists() for path in required):
        logger.warning("Skipping %s because training outputs are incomplete.", scenario_name)
        return

    build_ml_plots(
        feature_table_path=feature_table_path,
        predictions_path=predictions_path,
        results_dir=results_dir,
        models_dir=models_dir,
        output_dir=output_dir,
        scenario_label=scenario_name,
        n_jobs=n_jobs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build plots for classical ML scenarios")
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Scenario name, comma-separated list, or 'all'.",
    )
    parser.add_argument("--feature_table", type=str, default=None)
    parser.add_argument("--predictions", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--models_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--n_jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    explicit_paths = all(
        value is not None
        for value in [args.feature_table, args.predictions, args.results_dir, args.models_dir, args.out_dir]
    )

    if explicit_paths:
        build_ml_plots(
            feature_table_path=Path(args.feature_table),
            predictions_path=Path(args.predictions),
            results_dir=Path(args.results_dir),
            models_dir=Path(args.models_dir),
            output_dir=Path(args.out_dir),
            scenario_label=args.scenario if args.scenario != "all" else None,
            n_jobs=args.n_jobs,
        )
        return

    if args.scenario == "all":
        for scenario_name in get_scenario_catalog():
            _build_for_existing_scenario(scenario_name, n_jobs=args.n_jobs)
        return

    for scenario_name in [
        canonicalize_scenario_name(item)
        for item in args.scenario.split(",")
        if item.strip()
    ]:
        _build_for_existing_scenario(scenario_name, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
