"""
ml_scenarios.py
===============
Dataset registry and scenario assembly helpers for the classical ML block.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path

import pandas as pd

ML_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ML_ROOT.parent

DATASET_ORDER = ("uams", "ipvs", "mdvr-kcl")
SCENARIO_REQUIRED_COLUMNS = {"filename", "patient_id", "label", "split"}
DATASET_NAME_ALIASES = {
    "uams": "uams",
    "ipvs": "ipvs",
    "irvs": "ipvs",
    "mdvr-kcl": "mdvr-kcl",
    "mdvr_kcl": "mdvr-kcl",
}
DATASET_STORAGE_CANDIDATES = {
    "uams": ("uams",),
    "ipvs": ("ipvs", "irvs"),
    "mdvr-kcl": ("mdvr-kcl",),
}
DATASET_DISPLAY_NAMES = {
    "uams": "UAMS",
    "ipvs": "IPVS",
    "mdvr-kcl": "MDVR-KCL",
}
LEGACY_SCENARIO_NAME_REPLACEMENTS = (
    ("single_ipvs", "single_irvs"),
    ("cross_ipvs_to_", "cross_irvs_to_"),
    ("_to_ipvs", "_to_irvs"),
    ("_ipvs_", "_irvs_"),
)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    kind: str
    description: str
    train_datasets: tuple[str, ...]
    val_datasets: tuple[str, ...]
    test_datasets: tuple[str, ...]
    final_fit_splits: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_dataset_name(dataset_name: str) -> str:
    """Normalizes dataset aliases such as ``irvs`` -> ``ipvs``."""
    normalized = dataset_name.strip().lower().replace("_", "-")
    if normalized not in DATASET_NAME_ALIASES:
        raise ValueError(
            f"Unknown dataset name: {dataset_name}. "
            f"Expected one of: {sorted(DATASET_NAME_ALIASES)}"
        )
    return DATASET_NAME_ALIASES[normalized]


def display_dataset_name(dataset_name: str) -> str:
    """Returns a human-readable dataset label."""
    return DATASET_DISPLAY_NAMES[normalize_dataset_name(dataset_name)]


def _resolve_existing_dir(root: Path, candidates: tuple[str, ...]) -> Path:
    for candidate in candidates:
        candidate_path = root / candidate
        if candidate_path.exists():
            return candidate_path
    return root / candidates[0]


def dataset_token(dataset_name: str) -> str:
    """Converts a dataset name to a filesystem-friendly token."""
    return normalize_dataset_name(dataset_name).replace("-", "_")


def dataset_storage_name(dataset_name: str) -> str:
    """Returns the preferred on-disk directory token for one dataset."""
    canonical = normalize_dataset_name(dataset_name)
    return _resolve_existing_dir(
        PROJECT_ROOT / "data",
        DATASET_STORAGE_CANDIDATES[canonical],
    ).name


def dataset_data_dir(dataset_name: str) -> Path:
    canonical = normalize_dataset_name(dataset_name)
    return _resolve_existing_dir(
        PROJECT_ROOT / "data",
        DATASET_STORAGE_CANDIDATES[canonical],
    )


def dataset_src_dir(dataset_name: str) -> Path:
    canonical = normalize_dataset_name(dataset_name)
    return _resolve_existing_dir(
        PROJECT_ROOT / "src",
        DATASET_STORAGE_CANDIDATES[canonical],
    )


def dataset_metadata_path(dataset_name: str) -> Path:
    return dataset_data_dir(dataset_name) / "metadata.csv"


def dataset_processed_dir(dataset_name: str) -> Path:
    return dataset_data_dir(dataset_name) / "processed"


def dataset_cache_dir(cache_root: str | Path, dataset_name: str) -> Path:
    canonical = normalize_dataset_name(dataset_name)
    cache_root = Path(cache_root)
    candidates = DATASET_STORAGE_CANDIDATES[canonical]
    return _resolve_existing_dir(cache_root, candidates)


def canonicalize_scenario_name(scenario_name: str) -> str:
    """Normalizes legacy scenario aliases such as ``single_irvs`` -> ``single_ipvs``."""
    return scenario_name.strip().lower().replace("irvs", "ipvs")


def legacy_scenario_name(scenario_name: str) -> str:
    """Maps the canonical IPVS-based scenario name to the legacy IRVS token."""
    legacy_name = scenario_name
    for canonical_text, legacy_text in LEGACY_SCENARIO_NAME_REPLACEMENTS:
        legacy_name = legacy_name.replace(canonical_text, legacy_text)
    return legacy_name


def scenario_dir_candidates(scenario_name: str) -> list[str]:
    """Returns canonical and legacy directory-name candidates for one scenario."""
    canonical_name = canonicalize_scenario_name(scenario_name)
    candidates = [canonical_name]
    legacy_name = legacy_scenario_name(canonical_name)
    if legacy_name != canonical_name:
        candidates.append(legacy_name)
    return candidates


def resolve_existing_scenario_dir(scenario_name: str) -> Path:
    """Finds an existing scenario directory using canonical and legacy tokens."""
    scenario_root = ML_ROOT / "data" / "scenarios"
    for candidate in scenario_dir_candidates(scenario_name):
        candidate_path = scenario_root / candidate
        if candidate_path.exists():
            return candidate_path
    return scenario_root / canonicalize_scenario_name(scenario_name)


def load_dataset_metadata(dataset_name: str) -> pd.DataFrame:
    """
    Loads one dataset metadata table and normalizes identifiers so datasets can
    be safely merged without patient-id collisions.
    """
    canonical_name = normalize_dataset_name(dataset_name)
    metadata_path = dataset_metadata_path(canonical_name)
    processed_dir = dataset_processed_dir(canonical_name)
    df = pd.read_csv(metadata_path)

    missing = SCENARIO_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{metadata_path} missing required columns: {sorted(missing)}")

    normalized = df.copy()
    normalized["source_dataset"] = canonical_name
    normalized["dataset"] = canonical_name
    normalized["original_patient_id"] = normalized["patient_id"].astype(str)
    normalized["patient_id"] = normalized["source_dataset"] + "::" + normalized["original_patient_id"]

    if "recording_id" in normalized.columns:
        normalized["recording_id"] = (
            normalized["source_dataset"]
            + "::"
            + normalized["recording_id"].astype(str)
        )
    else:
        normalized["recording_id"] = (
            normalized["source_dataset"] + "::" + normalized["filename"].astype(str)
        )

    normalized["audio_path"] = normalized["filename"].map(
        lambda name: str((processed_dir / str(name)).resolve())
    )
    normalized["processed_path"] = normalized["audio_path"]
    normalized["dataset_root"] = str(dataset_data_dir(canonical_name).resolve())
    return normalized


def get_scenario_catalog() -> dict[str, ScenarioSpec]:
    """Returns all supported experiment scenarios."""
    scenarios: dict[str, ScenarioSpec] = {}

    for dataset_name in DATASET_ORDER:
        token = dataset_token(dataset_name)
        scenarios[f"single_{token}"] = ScenarioSpec(
            name=f"single_{token}",
            kind="single",
            description=(
                f"Train/validate/test only on the {display_dataset_name(dataset_name)} dataset."
            ),
            train_datasets=(dataset_name,),
            val_datasets=(dataset_name,),
            test_datasets=(dataset_name,),
            final_fit_splits=("train", "val"),
        )

    scenarios["combined_all"] = ScenarioSpec(
        name="combined_all",
        kind="combined",
        description="Train/validate/test on the union of UAMS, IPVS and MDVR-KCL.",
        train_datasets=DATASET_ORDER,
        val_datasets=DATASET_ORDER,
        test_datasets=DATASET_ORDER,
        final_fit_splits=("train", "val"),
    )

    for train_dataset, eval_dataset in permutations(DATASET_ORDER, 2):
        train_token = dataset_token(train_dataset)
        eval_token = dataset_token(eval_dataset)
        name = f"cross_{train_token}_to_{eval_token}"
        scenarios[name] = ScenarioSpec(
            name=name,
            kind="cross",
            description=(
                f"Train on {display_dataset_name(train_dataset)} (train split), "
                f"select on {display_dataset_name(eval_dataset)} (val split), "
                f"and test on {display_dataset_name(eval_dataset)} (test split)."
            ),
            train_datasets=(train_dataset,),
            val_datasets=(eval_dataset,),
            test_datasets=(eval_dataset,),
            final_fit_splits=("train",),
        )

    return scenarios


def resolve_scenarios(selection: str) -> list[ScenarioSpec]:
    """
    Resolves ``all`` or a comma-separated list of scenario names to specs.
    """
    catalog = get_scenario_catalog()
    if selection == "all":
        return [catalog[name] for name in catalog]

    names = [canonicalize_scenario_name(item) for item in selection.split(",") if item.strip()]
    unknown = [name for name in names if name not in catalog]
    if unknown:
        raise ValueError(
            "Unknown scenario(s): "
            + ", ".join(unknown)
            + ". Use --list_scenarios to inspect supported values."
        )
    return [catalog[name] for name in names]


def build_scenario_feature_table(
    scenario: ScenarioSpec,
    dataset_feature_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Selects and combines cached dataset feature tables according to a scenario.
    """
    role_map = [
        ("train", scenario.train_datasets, "train_source"),
        ("val", scenario.val_datasets, "evaluation_source"),
        ("test", scenario.test_datasets, "evaluation_source"),
    ]

    parts: list[pd.DataFrame] = []
    for split_name, dataset_names, dataset_role in role_map:
        for dataset_name in dataset_names:
            source_df = dataset_feature_tables[dataset_name]
            subset = source_df[source_df["split"] == split_name].copy()
            subset["dataset"] = dataset_name
            if "source_dataset" in subset.columns:
                subset["source_dataset"] = dataset_name
            if dataset_name == "ipvs":
                for column_name in ["patient_id", "recording_id"]:
                    if column_name in subset.columns:
                        subset[column_name] = subset[column_name].astype(str).str.replace(
                            "irvs::",
                            "ipvs::",
                            regex=False,
                        )
            subset["split_origin"] = subset["split"]
            subset["split"] = split_name
            subset["dataset_role"] = dataset_role
            subset["scenario_name"] = scenario.name
            subset["scenario_kind"] = scenario.kind
            parts.append(subset)

    if not parts:
        raise ValueError(f"Scenario {scenario.name} produced an empty feature table.")

    scenario_df = pd.concat(parts, ignore_index=True)
    scenario_df = scenario_df.sort_values(
        by=["split", "dataset", "patient_id", "filename"],
        kind="stable",
    ).reset_index(drop=True)
    return scenario_df


def summarize_scenario_feature_table(
    scenario: ScenarioSpec,
    feature_df: pd.DataFrame,
) -> dict[str, object]:
    """Builds a compact JSON-serializable summary for one scenario."""
    return {
        "scenario": scenario.name,
        "kind": scenario.kind,
        "description": scenario.description,
        "datasets": {
            "train": list(scenario.train_datasets),
            "val": list(scenario.val_datasets),
            "test": list(scenario.test_datasets),
        },
        "final_fit_splits": list(scenario.final_fit_splits),
        "n_rows": int(len(feature_df)),
        "n_patients": int(feature_df["patient_id"].nunique()),
        "split_counts": feature_df["split"].value_counts().sort_index().to_dict(),
        "dataset_counts": feature_df["dataset"].value_counts().sort_index().to_dict(),
        "class_counts": feature_df["label"].value_counts().sort_index().to_dict(),
    }
