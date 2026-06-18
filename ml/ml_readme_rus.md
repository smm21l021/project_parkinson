# ML Baseline Block

В этой папке находится classical machine-learning ветка проекта по анализу голоса при болезни Паркинсона.

Входные датасеты ожидаются в родительской папке проекта:

- `../data/uams`
- `../data/ipvs`
- `../data/mdvr-kcl`

Исправленный dataset-token: `ipvs`. Для обратной совместимости скрипты также принимают legacy-token `irvs` в уже существующих папках, кэшах и именах сценариев.

Все ML-specific кэши, запуски сценариев, модели, таблицы и графики сохраняются внутри `ml/data/`.

## Structure

### Source files

| Path | Назначение |
| --- | --- |
| `src/ml_scenarios.py` | Реестр датасетов, каталог сценариев, сборка сценария |
| `src/ml_features.py` | Извлечение acoustic features и кэширование признаков на уровне датасета |
| `src/ml_models.py` | Обучение, validation, выбор модели и финальная test-оценка |
| `src/ml_plots.py` | Перестроение графиков для одного или нескольких сценариев |
| `src/ml_pipeline.py` | End-to-end запуск пайплайна |

### Data folders

| Path | Назначение |
| --- | --- |
| `data/cache/dataset_features/` | Кэш acoustic features для каждого датасета |
| `data/scenarios/<scenario>/` | Признаки, модели, результаты и графики для одного сценария |
| `data/summaries/` | Сводные таблицы по сценариям и метаданные запусков |

## Supported scenarios

Пайплайн поддерживает 10 сценариев:

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

Интерпретация:

- `single_*`: train, validation и test берутся только из одного датасета
- `combined_all`: train, validation и test берутся из объединения всех трех датасетов
- `cross_A_to_B`: обучение на датасете `A` (`train` split), выбор модели на датасете `B` (`val` split), тестирование на датасете `B` (`test` split)

## Dependencies

Необходимые Python-пакеты:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `joblib`
- `librosa`
- `soundfile`
- `tqdm`

Пример команды установки:

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

Это не переобучает модели, если результаты сценария уже существуют.

```bash
python ml/src/ml_plots.py
python ml/src/ml_plots.py --scenario single_uams
python ml/src/ml_plots.py --scenario cross_uams_to_mdvr_kcl
```

## What the pipeline does

1. Загружает нормализованные metadata для выбранных датасетов.
2. Создает dataset-level feature cache, чтобы не извлекать признаки из одного и того же аудио повторно в разных сценариях.
3. Собирает feature table для сценария.
4. Извлекает acoustic features:
   - duration и статистики сигнала
   - zero-crossing rate
   - RMS energy
   - spectral centroid, bandwidth, rolloff, flatness, contrast
   - MFCC, delta MFCC, delta-delta MFCC
   - chroma
   - pitch statistics
   - approximate jitter и shimmer
   - approximate harmonic-to-noise ratio
5. Обучает три baseline-модели:
   - Logistic Regression
   - SVM with RBF kernel
   - Random Forest
6. Выбирает лучшую модель по validation patient-level `F1`.
7. Переобучает выбранную модель:
   - на `train + val` для `single_*` и `combined_all`
   - только на `train` для `cross_*`
8. Оценивает финальную модель на `test`.
9. Сохраняет метрики, предсказания, обученные модели и графики для каждого сценария.

## Important implementation notes

- Имена датасетов нормализуются к `uams`, `ipvs`, `mdvr-kcl`. Legacy-token `irvs` по-прежнему поддерживается для обратной совместимости.
- Идентификаторы пациентов получают префикс имени датасета, поэтому в объединенных сценариях пациенты из разных датасетов не смешиваются.
- Текущие значения `jitter` и `shimmer` являются легковесными приближениями на основе pitch и energy contours, а не клиническими Praat-grade измерениями.

## Main outputs

### Dataset feature cache

- `data/cache/dataset_features/uams/ml_feature_table.csv`
- `data/cache/dataset_features/ipvs/ml_feature_table.csv`
- `data/cache/dataset_features/mdvr-kcl/ml_feature_table.csv`

### Scenario outputs

Для каждого сценария `data/scenarios/<scenario>/` содержит:

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

Папка `data/scenarios/<scenario>/plots/` содержит следующие графики:

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

## Дополнительный анализ

Следующие файлы являются вспомогательными и не входят напрямую в основной пайплайн обучения. Они используются только для компактного описательного анализа, который можно отдельным абзацем включить в отчет.

- `src/ml_collect_feature_stats.py` считает patient-level статистики признаков для объединенного датасета, отбирает признаки со значимыми различиями между Healthy и Parkinson и строит график `data/summaries/plots/ml_feature_boxplots_combined_all_significant.png`.
- `src/ml_interpret_feature_groups.py` усредняет grouped permutation importance по всем сценариям отдельно для Logistic Regression, SVM и Random Forest и строит графики:
  - `data/summaries/plots/ml_feature_group_importance_logreg_average.png`
  - `data/summaries/plots/ml_feature_group_importance_svm_average.png`
  - `data/summaries/plots/ml_feature_group_importance_random_forest_average.png`

Связанные сводные таблицы:

- `data/summaries/ml_feature_class_stats_combined_all.csv`
- `data/summaries/ml_feature_class_stats_combined_all_significant.csv`
- `data/summaries/ml_model_feature_group_summary.csv`
