"""
ml_pipeline.py
==============
End-to-end classical ML pipeline for all requested dataset scenarios.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from ml_features import build_feature_table_from_frame
from ml_models import train_ml_baselines
from ml_plots import build_ml_plots
from ml_scenarios import (
    DATASET_ORDER,
    ML_ROOT,
    PROJECT_ROOT,
    build_scenario_feature_table,
    dataset_cache_dir,
    dataset_metadata_path,
    dataset_processed_dir,
    get_scenario_catalog,
    load_dataset_metadata,
    resolve_scenarios,
    summarize_scenario_feature_table,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def _scenario_dir(scenario_name: str) -> Path:
    return ML_ROOT / "data" / "scenarios" / scenario_name


def _scenario_subdirs(scenario_name: str) -> dict[str, Path]:
    scenario_dir = _scenario_dir(scenario_name)
    return {
        "root": scenario_dir,
        "features": scenario_dir / "features",
        "models": scenario_dir / "models",
        "results": scenario_dir / "results",
        "plots": scenario_dir / "plots",
    }


def _ensure_dataset_feature_cache(
    dataset_name: str,
    *,
    cache_root: str | Path,
    force_recompute: bool,
) -> pd.DataFrame:
    cache_dir = dataset_cache_dir(cache_root, dataset_name)
    feature_csv = cache_dir / "ml_feature_table.csv"
    summary_json = cache_dir / "ml_feature_summary.json"

    if force_recompute or not feature_csv.exists():
        logger.info("Extracting cached features for dataset: %s", dataset_name)
        metadata_df = load_dataset_metadata(dataset_name)
        build_feature_table_from_frame(
            metadata_df=metadata_df,
            output_csv=feature_csv,
            summary_json=summary_json,
            processed_dir=dataset_processed_dir(dataset_name),
        )
    else:
        logger.info("Using cached features for dataset: %s", dataset_name)

    return pd.read_csv(feature_csv)


def _extract_overview_row(
    scenario_name: str,
    training_summary: dict[str, object],
    final_metrics_path: Path,
) -> dict[str, object]:
    final_metrics_df = pd.read_csv(final_metrics_path)
    patient_row = final_metrics_df[
        (final_metrics_df["evaluation_level"] == "patient")
        & (final_metrics_df["subset"] == "all")
    ].iloc[0]
    file_row = final_metrics_df[
        (final_metrics_df["evaluation_level"] == "file")
        & (final_metrics_df["subset"] == "all")
    ].iloc[0]

    return {
        "scenario": scenario_name,
        "selected_model": training_summary["selected_model"],
        "final_fit_splits": ",".join(training_summary["final_fit_splits"]),
        "n_train": training_summary["n_train"],
        "n_val": training_summary["n_val"],
        "n_test": training_summary["n_test"],
        "patient_test_accuracy": patient_row["accuracy"],
        "patient_test_balanced_accuracy": patient_row["balanced_accuracy"],
        "patient_test_precision": patient_row["precision"],
        "patient_test_recall": patient_row["recall"],
        "patient_test_f1": patient_row["f1"],
        "patient_test_roc_auc": patient_row["roc_auc"],
        "file_test_accuracy": file_row["accuracy"],
        "file_test_balanced_accuracy": file_row["balanced_accuracy"],
        "file_test_precision": file_row["precision"],
        "file_test_recall": file_row["recall"],
        "file_test_f1": file_row["f1"],
        "file_test_roc_auc": file_row["roc_auc"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the classical ML baseline pipeline")
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Scenario name, comma-separated list, or 'all'.",
    )
    parser.add_argument(
        "--list_scenarios",
        action="store_true",
        help="Print all supported scenarios and exit.",
    )
    parser.add_argument("--scoring", type=str, default="f1")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument(
        "--force_recompute_features",
        action="store_true",
        help="Recompute dataset-level feature caches even if they already exist.",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Train models and save metrics without rebuilding plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = get_scenario_catalog()

    if args.list_scenarios:
        print("Supported scenarios:")
        for name, spec in catalog.items():
            print(f"- {name}: {spec.description}")
        return

    scenarios = resolve_scenarios(args.scenario)
    dataset_names = [
        dataset_name
        for dataset_name in DATASET_ORDER
        if any(
            dataset_name in scenario.train_datasets
            or dataset_name in scenario.val_datasets
            or dataset_name in scenario.test_datasets
            for scenario in scenarios
        )
    ]

    logger.info("Selected scenarios: %s", ", ".join(scenario.name for scenario in scenarios))
    logger.info("Datasets involved: %s", ", ".join(dataset_names))

    cache_root = ML_ROOT / "data" / "cache" / "dataset_features"
    dataset_feature_tables = {
        dataset_name: _ensure_dataset_feature_cache(
            dataset_name,
            cache_root=cache_root,
            force_recompute=args.force_recompute_features,
        )
        for dataset_name in dataset_names
    }

    summary_rows: list[dict[str, object]] = []
    executed_scenarios = {
        "project_root": str(PROJECT_ROOT),
        "ml_root": str(ML_ROOT),
        "selected_scenarios": [scenario.name for scenario in scenarios],
        "datasets": dataset_names,
        "scenario_specs": {scenario.name: scenario.to_dict() for scenario in scenarios},
        "dataset_metadata_paths": {
            dataset_name: str(dataset_metadata_path(dataset_name))
            for dataset_name in dataset_names
        },
    }
    _save_json(ML_ROOT / "data" / "summaries" / "ml_scenario_catalog.json", executed_scenarios)

    for scenario in scenarios:
        logger.info("Running scenario: %s", scenario.name)
        subdirs = _scenario_subdirs(scenario.name)
        for directory in subdirs.values():
            directory.mkdir(parents=True, exist_ok=True)

        scenario_feature_df = build_scenario_feature_table(scenario, dataset_feature_tables)
        feature_csv = subdirs["features"] / "ml_feature_table.csv"
        scenario_feature_df.to_csv(feature_csv, index=False)

        feature_summary = summarize_scenario_feature_table(scenario, scenario_feature_df)
        _save_json(subdirs["features"] / "ml_feature_summary.json", feature_summary)
        _save_json(subdirs["root"] / "scenario_config.json", scenario.to_dict())

        training_summary = train_ml_baselines(
            feature_df=scenario_feature_df,
            model_dir=subdirs["models"],
            result_dir=subdirs["results"],
            scenario_name=scenario.name,
            final_fit_splits=scenario.final_fit_splits,
            scoring=args.scoring,
            n_jobs=args.n_jobs,
        )

        if not args.skip_plots:
            build_ml_plots(
                feature_table_path=feature_csv,
                predictions_path=subdirs["results"] / "ml_predictions_all.csv",
                results_dir=subdirs["results"],
                models_dir=subdirs["models"],
                output_dir=subdirs["plots"],
                scenario_label=scenario.name,
                n_jobs=args.n_jobs,
            )

        summary_rows.append(
            _extract_overview_row(
                scenario_name=scenario.name,
                training_summary=training_summary,
                final_metrics_path=subdirs["results"] / "ml_final_test_metrics.csv",
            )
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(by="scenario").reset_index(drop=True)
    summary_path = ML_ROOT / "data" / "summaries" / "ml_scenarios_overview.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved scenario overview: %s", summary_path)


if __name__ == "__main__":
    main()
