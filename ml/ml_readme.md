# ML Baseline Block

This folder contains the classical machine-learning branch for the Parkinson voice project.

Input datasets are expected in the parent project folder:

- `../data/uams`
- `../data/ipvs`
- `../data/mdvr-kcl`

The corrected dataset token is `ipvs`. For backward compatibility, the scripts also accept the legacy token `irvs` in existing folders, caches and scenario names.

All ML-specific caches, scenario runs, models, tables and plots are stored inside `ml/data/`.

## Structure

### Source files

| Path | Purpose |
| --- | --- |
| `src/ml_scenarios.py` | Dataset registry, scenario catalog, scenario assembly |
| `src/ml_features.py` | Acoustic feature extraction and dataset-level feature caching |
| `src/ml_models.py` | Training, validation, model selection, final test evaluation |
| `src/ml_plots.py` | Plot rebuilding for one or many scenarios |
| `src/ml_pipeline.py` | End-to-end scenario runner |

### Data folders

| Path | Purpose |
| --- | --- |
| `data/cache/dataset_features/` | Per-dataset cached acoustic features |
| `data/scenarios/<scenario>/` | Features, models, results and plots for one scenario |
| `data/summaries/` | Scenario-level overview tables and run metadata |

## Supported scenarios

The pipeline supports 10 scenarios:

1. `single_uams`
2. `single_ipvs`
3. `single_mdvr_kcl`
4. `combined_all`
5. `cross_uams_to_ipvs`
6. `cross_uams_to_mdvr_kcl`
7. `cross_ipvs_to_uams`
8. `cross_ipvs_to_mdvr_kcl`
9. `cross_mdvr_kcl_to_uams`
10. `cross_mdvr_kcl_to_ipvs`

Interpretation:

- `single_*`: train, validation and test are taken from one dataset only
- `combined_all`: train, validation and test are taken from the union of all three datasets
- `cross_A_to_B`: train on dataset `A` (train split), select on dataset `B` (val split), test on dataset `B` (test split)

## Dependencies

Required Python packages:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `joblib`
- `librosa`
- `soundfile`
- `tqdm`

Example install command:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib librosa soundfile tqdm
```

## How to run

### Show all scenarios

```bash
python ml/src/ml_pipeline.py --list_scenarios
```

### Run everything

```bash
python ml/src/ml_pipeline.py
```

### Run one scenario

```bash
python ml/src/ml_pipeline.py --scenario single_uams
python ml/src/ml_pipeline.py --scenario cross_uams_to_mdvr_kcl
```

### Run several scenarios

```bash
python ml/src/ml_pipeline.py --scenario single_uams,combined_all,cross_ipvs_to_uams
```

### Recompute dataset-level acoustic features

```bash
python ml/src/ml_pipeline.py --scenario combined_all --force_recompute_features
```

### Rebuild plots only

This does not retrain models if the scenario results already exist.

```bash
python ml/src/ml_plots.py
python ml/src/ml_plots.py --scenario single_uams
python ml/src/ml_plots.py --scenario cross_uams_to_mdvr_kcl
```

## What the pipeline does

1. Loads normalized metadata for the selected datasets.
2. Makes dataset-level feature caches so the same audio is not processed repeatedly across scenarios.
3. Builds a scenario feature table.
4. Extracts acoustic features:
   - duration and signal statistics
   - zero-crossing rate
   - RMS energy
   - spectral centroid, bandwidth, rolloff, flatness, contrast
   - MFCC, delta MFCC, delta-delta MFCC
   - chroma
   - pitch statistics
   - approximate jitter and shimmer
   - approximate harmonic-to-noise ratio
5. Trains three baseline models:
   - Logistic Regression
   - SVM with RBF kernel
   - Random Forest
6. Selects the best model by validation patient-level `F1`.
7. Refits the selected model:
   - on `train + val` for `single_*` and `combined_all`
   - on `train` only for `cross_*`
8. Evaluates the final model on `test`.
9. Saves metrics, predictions, trained models and plots per scenario.

## Important implementation notes

- Dataset names are normalized to `uams`, `ipvs`, `mdvr-kcl`. The legacy token `irvs` is still accepted for backward compatibility.
- Patient identifiers are prefixed with the dataset name, so combined scenarios do not mix patients from different datasets.
- The current `jitter` and `shimmer` values are lightweight approximations based on pitch and energy contours, not clinical Praat-grade measurements.

## Main outputs

### Dataset feature cache

- `data/cache/dataset_features/uams/ml_feature_table.csv`
- `data/cache/dataset_features/ipvs/ml_feature_table.csv`
- `data/cache/dataset_features/mdvr-kcl/ml_feature_table.csv`

### Scenario outputs

For each scenario `data/scenarios/<scenario>/` contains:

- `features/ml_feature_table.csv`
- `features/ml_feature_summary.json`
- `models/ml_logreg.joblib`
- `models/ml_svm.joblib`
- `models/ml_random_forest.joblib`
- `models/ml_selected_final_model.joblib`
- `results/ml_metrics_summary.csv`
- `results/ml_model_selection.csv`
- `results/ml_final_test_metrics.csv`
- `results/ml_final_test_predictions.csv`
- `results/ml_predictions_all.csv`
- `results/ml_permutation_importance.csv`
- `results/ml_training_summary.json`
- `plots/*.png`

### Plot files

The `data/scenarios/<scenario>/plots/` folder stores the following plot files:

- `ml_confusion_test_logreg.png`
- `ml_confusion_test_random_forest.png`
- `ml_confusion_test_svm.png`
- `ml_confusion_val_logreg.png`
- `ml_confusion_val_random_forest.png`
- `ml_confusion_val_svm.png`
- `ml_logreg_cv_heatmap.png`
- `ml_logreg_top20_coefficients.png`
- `ml_roc_test_all_models.png`
- `ml_rf_cv_heatmap.png`
- `ml_rf_top20_feature_importance.png`
- `ml_rf_tree_visualization.png`
- `ml_svm_cv_heatmap.png`
- `ml_svm_top20_permutation_importance.png`

### Global summaries

- `data/summaries/ml_scenario_catalog.json`
- `data/summaries/ml_scenarios_overview.csv`

## Supplementary analysis

The following files are auxiliary and are not part of the main training pipeline. They are used only for a compact descriptive analysis that can be inserted into the report as a separate paragraph.

- `src/ml_collect_feature_stats.py` computes patient-level feature statistics for the combined dataset, filters features with significant Healthy vs Parkinson differences, and builds `data/summaries/plots/ml_feature_boxplots_combined_all_significant.png`.
- `src/ml_interpret_feature_groups.py` averages grouped permutation importance across all scenarios separately for Logistic Regression, SVM and Random Forest, and builds:
  - `data/summaries/plots/ml_feature_group_importance_logreg_average.png`
  - `data/summaries/plots/ml_feature_group_importance_svm_average.png`
  - `data/summaries/plots/ml_feature_group_importance_random_forest_average.png`

Related summary tables:

- `data/summaries/ml_feature_class_stats_combined_all.csv`
- `data/summaries/ml_feature_class_stats_combined_all_significant.csv`
- `data/summaries/ml_model_feature_group_summary.csv`
