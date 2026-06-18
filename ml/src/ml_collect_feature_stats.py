"""
ml_collect_feature_stats.py
===========================
Collects model-independent patient-level feature statistics for the combined
dataset and visualizes only features that differ significantly between healthy
and Parkinson groups.

Typical usage:

    python ml/src/ml_collect_feature_stats.py
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import mannwhitneyu

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"

ML_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENARIO = "combined_all"
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
    return ML_ROOT / "data" / "scenarios" / scenario_name


def _summaries_dir() -> Path:
    return ML_ROOT / "data" / "summaries"


def _plots_dir() -> Path:
    return _summaries_dir() / "plots"


def _feature_columns(feature_df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in feature_df.columns
        if column not in NON_FEATURE_COLUMNS
        and column != "label"
        and pd.api.types.is_numeric_dtype(feature_df[column])
    ]


def _feature_group(feature_name: str) -> str:
    if feature_name in SPECIAL_FEATURE_GROUPS:
        return SPECIAL_FEATURE_GROUPS[feature_name]

    for prefix, group_name in PREFIX_FEATURE_GROUPS:
        if feature_name.startswith(prefix):
            return group_name

    return "Other"


def _patient_level_table(
    feature_df: pd.DataFrame,
    numeric_features: list[str],
) -> pd.DataFrame:
    metadata_df = feature_df.groupby("patient_id", as_index=False).agg(
        label=("label", "first"),
        dataset=("dataset", "first"),
    )
    feature_means_df = feature_df.groupby("patient_id", as_index=False)[numeric_features].mean()
    return metadata_df.merge(feature_means_df, on="patient_id", how="inner")


def _series_stats(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
        }

    return {
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if len(clean) > 1 else np.nan,
        "median": float(clean.median()),
        "q1": float(clean.quantile(0.25)),
        "q3": float(clean.quantile(0.75)),
    }


def _cohens_d(group_a: pd.Series, group_b: pd.Series) -> float:
    clean_a = pd.to_numeric(group_a, errors="coerce").dropna()
    clean_b = pd.to_numeric(group_b, errors="coerce").dropna()

    if len(clean_a) < 2 or len(clean_b) < 2:
        return np.nan

    std_a = float(clean_a.std(ddof=1))
    std_b = float(clean_b.std(ddof=1))
    pooled_var = (
        ((len(clean_a) - 1) * (std_a ** 2))
        + ((len(clean_b) - 1) * (std_b ** 2))
    ) / (len(clean_a) + len(clean_b) - 2)

    if pooled_var <= 0 or not np.isfinite(pooled_var):
        return np.nan

    return float((clean_b.mean() - clean_a.mean()) / np.sqrt(pooled_var))


def _mannwhitney_pvalue(group_a: pd.Series, group_b: pd.Series) -> float:
    clean_a = pd.to_numeric(group_a, errors="coerce").dropna()
    clean_b = pd.to_numeric(group_b, errors="coerce").dropna()

    if len(clean_a) < 2 or len(clean_b) < 2:
        return np.nan

    try:
        result = mannwhitneyu(clean_a, clean_b, alternative="two-sided")
    except ValueError:
        return np.nan
    return float(result.pvalue)


def _benjamini_hochberg(pvalues: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    valid_mask = np.isfinite(arr)
    adjusted = np.full(arr.shape, np.nan, dtype=float)

    valid = arr[valid_mask]
    if valid.size == 0:
        return adjusted

    order = np.argsort(valid)
    ranked = valid[order]
    n_tests = ranked.size
    raw_adjusted = ranked * n_tests / np.arange(1, n_tests + 1)
    monotone_adjusted = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    monotone_adjusted = np.clip(monotone_adjusted, 0.0, 1.0)

    restored = np.empty(n_tests, dtype=float)
    restored[order] = monotone_adjusted
    adjusted[valid_mask] = restored
    return adjusted


def _build_stats_table(patient_df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    healthy_df = patient_df[patient_df["label"] == 0].copy()
    parkinson_df = patient_df[patient_df["label"] == 1].copy()

    rows: list[dict[str, object]] = []
    for feature_name in numeric_features:
        healthy_stats = _series_stats(healthy_df[feature_name])
        parkinson_stats = _series_stats(parkinson_df[feature_name])
        p_value = _mannwhitney_pvalue(healthy_df[feature_name], parkinson_df[feature_name])
        effect_size = _cohens_d(healthy_df[feature_name], parkinson_df[feature_name])

        mean_diff = (
            parkinson_stats["mean"] - healthy_stats["mean"]
            if pd.notna(parkinson_stats["mean"]) and pd.notna(healthy_stats["mean"])
            else np.nan
        )
        median_diff = (
            parkinson_stats["median"] - healthy_stats["median"]
            if pd.notna(parkinson_stats["median"]) and pd.notna(healthy_stats["median"])
            else np.nan
        )

        rows.append(
            {
                "scenario": DEFAULT_SCENARIO,
                "feature": feature_name,
                "feature_group": _feature_group(feature_name),
                "healthy_n_patients": int(len(healthy_df)),
                "parkinson_n_patients": int(len(parkinson_df)),
                "healthy_mean": healthy_stats["mean"],
                "healthy_std": healthy_stats["std"],
                "healthy_median": healthy_stats["median"],
                "healthy_q1": healthy_stats["q1"],
                "healthy_q3": healthy_stats["q3"],
                "parkinson_mean": parkinson_stats["mean"],
                "parkinson_std": parkinson_stats["std"],
                "parkinson_median": parkinson_stats["median"],
                "parkinson_q1": parkinson_stats["q1"],
                "parkinson_q3": parkinson_stats["q3"],
                "mean_difference_pd_minus_healthy": mean_diff,
                "median_difference_pd_minus_healthy": median_diff,
                "cohens_d": effect_size,
                "abs_cohens_d": abs(effect_size) if pd.notna(effect_size) else np.nan,
                "p_value_mannwhitney": p_value,
            }
        )

    stats_df = pd.DataFrame(rows)
    stats_df["q_value_bh"] = _benjamini_hochberg(stats_df["p_value_mannwhitney"])
    return stats_df.sort_values(
        by=["q_value_bh", "abs_cohens_d", "feature"],
        ascending=[True, False, True],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


def _save_fig(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", output_path)


def _plot_significant_boxplots(
    patient_df: pd.DataFrame,
    significant_df: pd.DataFrame,
    *,
    output_path: Path,
    top_n: int,
) -> None:
    if significant_df.empty:
        logger.warning("No significant features found; box-plot figure was skipped.")
        return

    plot_df = patient_df.copy()
    plot_df["class_name"] = plot_df["label"].map({0: "Healthy", 1: "Parkinson"})

    top_df = significant_df.sort_values(
        by=["abs_cohens_d", "q_value_bh", "feature"],
        ascending=[False, True, True],
        kind="stable",
    ).head(top_n)

    features = top_df["feature"].tolist()
    n_features = len(features)
    ncols = 2
    nrows = math.ceil(n_features / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, max(4.8, 3.8 * nrows)))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, feature_name in enumerate(features):
        row_idx, col_idx = divmod(idx, ncols)
        ax = axes[row_idx, col_idx]

        feature_values = plot_df[["class_name", feature_name]].rename(columns={feature_name: "value"})
        feature_values = feature_values.dropna()

        sns.boxplot(
            data=feature_values,
            x="class_name",
            y="value",
            hue="class_name",
            order=["Healthy", "Parkinson"],
            hue_order=["Healthy", "Parkinson"],
            palette=["#4C78A8", "#E45756"],
            showfliers=False,
            legend=False,
            ax=ax,
        )

        stat_row = top_df[top_df["feature"] == feature_name].iloc[0]
        ax.set_title(
            f"{feature_name}\nq={stat_row['q_value_bh']:.4g}, |d|={stat_row['abs_cohens_d']:.3f}"
        )
        ax.set_xlabel("")
        ax.set_ylabel("Value")

    for idx in range(n_features, nrows * ncols):
        row_idx, col_idx = divmod(idx, ncols)
        axes[row_idx, col_idx].axis("off")

    fig.suptitle(
        "Combined dataset: significant feature differences between Healthy and Parkinson",
        fontsize=13,
        y=1.01,
    )
    _save_fig(fig, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect significant feature statistics for the combined dataset."
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=_summaries_dir() / "ml_feature_class_stats_combined_all.csv",
        help="Path to the full combined-dataset feature-statistics CSV.",
    )
    parser.add_argument(
        "--significant_output_csv",
        type=Path,
        default=_summaries_dir() / "ml_feature_class_stats_combined_all_significant.csv",
        help="Path to the significant-only feature-statistics CSV.",
    )
    parser.add_argument(
        "--plots_dir",
        type=Path,
        default=_plots_dir(),
        help="Directory for box-plot figures.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="How many significant features to include in the box-plot figure.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="FDR threshold for Benjamini-Hochberg-corrected q-values.",
    )
    parser.add_argument(
        "--min_effect_size",
        type=float,
        default=0.5,
        help="Minimum |Cohen's d| required to treat a feature as practically different.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    feature_table_path = _scenario_dir(DEFAULT_SCENARIO) / "features" / "ml_feature_table.csv"
    if not feature_table_path.exists():
        raise FileNotFoundError(f"Missing feature table: {feature_table_path}")

    feature_df = pd.read_csv(feature_table_path)
    numeric_features = _feature_columns(feature_df)
    patient_df = _patient_level_table(feature_df, numeric_features)
    stats_df = _build_stats_table(patient_df, numeric_features)

    significant_df = stats_df[
        (stats_df["q_value_bh"] <= args.alpha)
        & (stats_df["abs_cohens_d"] >= args.min_effect_size)
    ].copy()
    significant_df = significant_df.sort_values(
        by=["abs_cohens_d", "q_value_bh", "feature"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.significant_output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    stats_df.to_csv(args.output_csv, index=False)
    significant_df.to_csv(args.significant_output_csv, index=False)
    logger.info("Saved combined-dataset feature statistics: %s", args.output_csv)
    logger.info("Saved significant-only feature statistics: %s", args.significant_output_csv)

    _plot_significant_boxplots(
        patient_df,
        significant_df,
        output_path=args.plots_dir / "ml_feature_boxplots_combined_all_significant.png",
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
