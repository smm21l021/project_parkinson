"""
ml_interpret_feature_groups.py
==============================
Builds model-level feature-group interpretation across all saved scenarios.

For each scenario and each baseline model, the script computes permutation
importance on the scenario test split, aggregates features into semantic groups,
normalizes group weights within the scenario, and then averages those weights
across scenarios. The result is three summary diagrams: one for Logistic
Regression, one for SVM, and one for Random Forest.

Typical usage:

    python ml/src/ml_interpret_feature_groups.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance

from ml_scenarios import (
    canonicalize_scenario_name,
    get_scenario_catalog,
    resolve_existing_scenario_dir,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"

ML_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE_NAMES = {
    "logreg": "ml_logreg.joblib",
    "svm": "ml_svm.joblib",
    "random_forest": "ml_random_forest.joblib",
}
MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "svm": "SVM",
    "random_forest": "Random Forest",
}
SPECIAL_FEATURE_GROUPS = {
    "duration_sec": "Signal Statistics",
    "signal_mean": "Signal Statistics",
    "signal_std": "Signal Statistics",
    "signal_abs_mean": "Signal Statistics",
    "voiced_fraction": "Pitch / Voicing",
    "pitch_strength_mean": "Pitch / Voicing",
    "pitch_range_hz": "Pitch / Voicing",
    "hnr_db": "Perturbation / HNR",
    "jitter_abs": "Perturbation / HNR",
    "jitter_rel": "Perturbation / HNR",
    "shimmer_abs": "Perturbation / HNR",
    "shimmer_rel": "Perturbation / HNR",
}
PREFIX_FEATURE_GROUPS = (
    ("mfcc_delta2_", "Delta-Delta MFCC"),
    ("mfcc_delta_", "Delta MFCC"),
    ("mfcc_", "MFCC"),
    ("chroma_", "Chroma"),
    ("spectral_contrast_", "Spectral Contrast"),
    ("spectral_centroid_", "Spectral Descriptors"),
    ("spectral_bandwidth_", "Spectral Descriptors"),
    ("spectral_rolloff_", "Spectral Descriptors"),
    ("spectral_flatness_", "Spectral Descriptors"),
    ("zcr_", "Energy / ZCR"),
    ("rms_", "Energy / ZCR"),
    ("pitch_hz_", "Pitch / Voicing"),
)


def _scenario_dir(scenario_name: str) -> Path:
    return resolve_existing_scenario_dir(scenario_name)


def _summaries_dir() -> Path:
    return ML_ROOT / "data" / "summaries"


def _plots_dir() -> Path:
    return _summaries_dir() / "plots"


def _feature_group(feature_name: str) -> str:
    if feature_name in SPECIAL_FEATURE_GROUPS:
        return SPECIAL_FEATURE_GROUPS[feature_name]

    for prefix, group_name in PREFIX_FEATURE_GROUPS:
        if feature_name.startswith(prefix):
            return group_name

    return "Other"


def _model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def _resolve_scenarios(selection: str) -> list[str]:
    catalog = get_scenario_catalog()
    if selection == "all":
        return list(catalog.keys())

    names = [canonicalize_scenario_name(item) for item in selection.split(",") if item.strip()]
    unknown = [name for name in names if name not in catalog]
    if unknown:
        raise ValueError(f"Unknown scenario(s): {', '.join(unknown)}")
    return names


def _save_fig(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", output_path)


def _compute_group_importance_for_model(
    *,
    scenario_name: str,
    model_name: str,
    n_repeats: int,
    n_jobs: int,
) -> pd.DataFrame:
    scenario_dir = _scenario_dir(scenario_name)
    feature_table_path = scenario_dir / "features" / "ml_feature_table.csv"
    model_path = scenario_dir / "models" / MODEL_FILE_NAMES[model_name]

    if not feature_table_path.exists():
        raise FileNotFoundError(f"Missing feature table: {feature_table_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model bundle: {model_path}")

    feature_df = pd.read_csv(feature_table_path)
    test_df = feature_df[feature_df["split"] == "test"].reset_index(drop=True)
    if test_df.empty:
        raise ValueError(f"Scenario {scenario_name} has empty test split.")

    bundle = joblib.load(model_path)
    estimator = bundle["estimator"]
    feature_columns = bundle["feature_columns"]

    importance = permutation_importance(
        estimator,
        test_df[feature_columns],
        test_df["label"].to_numpy(),
        n_repeats=n_repeats,
        random_state=42,
        scoring="f1",
        n_jobs=n_jobs,
    )

    feature_importance_df = pd.DataFrame(
        {
            "scenario": scenario_name,
            "model": model_name,
            "feature": feature_columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    )
    feature_importance_df["feature_group"] = feature_importance_df["feature"].map(_feature_group)
    feature_importance_df["positive_importance"] = feature_importance_df["importance_mean"].clip(lower=0.0)

    grouped_df = (
        feature_importance_df.groupby(["scenario", "model", "feature_group"], as_index=False)
        .agg(
            n_features=("feature", "size"),
            importance_sum=("importance_mean", "sum"),
            positive_importance_sum=("positive_importance", "sum"),
            mean_feature_importance=("importance_mean", "mean"),
            max_feature_importance=("importance_mean", "max"),
        )
    )

    total_positive = float(grouped_df["positive_importance_sum"].sum())
    if total_positive > 0:
        grouped_df["importance_share"] = grouped_df["positive_importance_sum"] / total_positive
    else:
        total_raw = float(grouped_df["importance_sum"].clip(lower=0.0).sum())
        grouped_df["importance_share"] = (
            grouped_df["importance_sum"].clip(lower=0.0) / total_raw if total_raw > 0 else 0.0
        )

    return grouped_df.sort_values(
        by=["importance_share", "positive_importance_sum", "feature_group"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _plot_model_summary(
    summary_df: pd.DataFrame,
    *,
    model_name: str,
    output_path: Path,
) -> None:
    model_df = summary_df[summary_df["model"] == model_name].copy()
    model_df = model_df.sort_values(
        by=["mean_importance_share", "mean_positive_importance_sum", "feature_group"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)

    fig_height = max(4.6, 0.55 * len(model_df) + 1.8)
    fig, ax = plt.subplots(figsize=(9.6, fig_height))
    sns.barplot(
        data=model_df,
        x="mean_importance_share",
        y="feature_group",
        hue="feature_group",
        palette="Blues_r",
        legend=False,
        ax=ax,
    )
    ax.set_title(
        f"Average feature-group focus across scenarios\n{_model_label(model_name)}"
    )
    ax.set_xlabel("Mean relative importance share across scenarios")
    ax.set_ylabel("Feature group")
    ax.invert_yaxis()

    max_share = float(model_df["mean_importance_share"].max()) if not model_df.empty else 0.0
    text_offset = max_share * 0.01 if max_share > 0 else 0.001
    for bar, (_, row) in zip(ax.patches, model_df.iterrows(), strict=False):
        ax.text(
            bar.get_width() + text_offset,
            bar.get_y() + bar.get_height() / 2.0,
            f"n={int(row['n_scenarios'])}",
            va="center",
            ha="left",
            fontsize=9,
        )

    _save_fig(fig, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average feature-group interpretation across all ML scenarios."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Scenario name, comma-separated list, or 'all'.",
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=_summaries_dir() / "ml_model_feature_group_summary.csv",
        help="Path to the averaged per-model feature-group CSV.",
    )
    parser.add_argument(
        "--plots_dir",
        type=Path,
        default=_plots_dir(),
        help="Directory for averaged feature-group bar plots.",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=6,
        help="Number of repeats for permutation importance.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Parallel jobs for permutation importance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_names = _resolve_scenarios(args.scenario)

    detail_parts: list[pd.DataFrame] = []
    for scenario_name in scenario_names:
        for model_name in MODEL_FILE_NAMES:
            grouped_df = _compute_group_importance_for_model(
                scenario_name=scenario_name,
                model_name=model_name,
                n_repeats=args.n_repeats,
                n_jobs=args.n_jobs,
            )
            detail_parts.append(grouped_df)
            logger.info(
                "Processed model interpretation: scenario=%s, model=%s",
                scenario_name,
                model_name,
            )

    details_df = pd.concat(detail_parts, ignore_index=True)
    summary_df = (
        details_df.groupby(["model", "feature_group"], as_index=False)
        .agg(
            n_scenarios=("scenario", "nunique"),
            mean_importance_share=("importance_share", "mean"),
            std_importance_share=("importance_share", "std"),
            mean_positive_importance_sum=("positive_importance_sum", "mean"),
            mean_raw_importance_sum=("importance_sum", "mean"),
        )
    )
    summary_df = summary_df.sort_values(
        by=["model", "mean_importance_share", "mean_positive_importance_sum", "feature_group"],
        ascending=[True, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    summary_df["model_label"] = summary_df["model"].map(_model_label)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary_csv, index=False)
    logger.info("Saved averaged model/group interpretation: %s", args.summary_csv)

    for model_name in MODEL_FILE_NAMES:
        _plot_model_summary(
            summary_df,
            model_name=model_name,
            output_path=args.plots_dir / f"ml_feature_group_importance_{model_name}_average.png",
        )


if __name__ == "__main__":
    main()
