# Neural Network Block

В этой папке находится deep learning ветка проекта по анализу голоса при болезни Паркинсона.

Входные датасеты ожидаются в родительской папке проекта:

- `../data/UAMS`
- `../data/MDVR-KCL`
- `../data/IPVS`

Все результаты (веса моделей, графики, метрики) сохраняются внутри `nn/data/`.

## Structure

### Source files

| Path | Назначение |
| --- | --- |
| `src/config.py` | Конфигурация: пути, гиперпараметры, список сценариев |
| `src/dataset.py` | Загрузка данных, PyTorch Dataset, patient-level сплиты |
| `src/models.py` | Архитектуры: CNN Baseline, ResNet18, DenseNet |
| `src/train.py` | Обучение, валидация, тестирование, сохранение результатов |
| `src/gradcam.py` | Grad-CAM визуализация внимания модели |
| `src/pipeline.py` | Запуск всех экспериментов |
| `src/diagnostics.py` | Проверка данных на утечки и корректность сплитов |
| `src/baseline_diagnostics.py` | Быстрое сравнение базовых моделей (5 эпох) |

### Data folders

| Path | Назначение |
| --- | --- |
| `data/cnn_baseline/scenarios/<scenario>/` | Результаты для CNN Baseline |
| `data/transfer_learning/scenarios/<scenario>/` | Результаты для ResNet |
| `data/densenet/scenarios/<scenario>/` | Результаты для DenseNet |
| `data/summaries/` | Сводные таблицы и графики сравнения |

## Supported scenarios

Пайплайн поддерживает 10 сценариев:

1. `single_uams`
2. `single_mdvr_kcl`
3. `single_ipvs`
4. `combined_all`
5. `cross_uams_to_mdvr_kcl`
6. `cross_uams_to_ipvs`
7. `cross_mdvr_kcl_to_uams`
8. `cross_mdvr_kcl_to_ipvs`
9. `cross_ipvs_to_mdvr_kcl`
10. `cross_ipvs_to_uams`

Интерпретация:

- `single_*`: train, validation и test берутся только из одного датасета
- `combined_all`: объединение всех трёх датасетов
- `cross_A_to_B`: обучение на датасете `A` (`train` split, `val` split), тестирование на датасете `B` (`test` split)

## Models

| Модель | Архитектура | Параметры | Особенности |
|--------|-------------|-----------|-------------|
| **CNN Baseline** | 4 свёрточных блока + 2 FC | ~2.5M | Обучение с нуля |
| **Transfer Learning** | ResNet18 | ~27M | Предобучена на ImageNet |
| **DenseNet** | DenseNet121 | ~7M | Предобучена на ImageNet |

## Dependencies

Необходимые Python-пакеты:

- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `librosa>=0.10.0`
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `scikit-learn>=1.3.0`
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`
- `tqdm>=4.65.0`
- `scipy>=1.10.0`

Установка:

```bash
pip install -r requirements.txt
```

## How to run

### Запустить всё

```bash
python nn/src/pipeline.py
```

### Сравнение базовых моделей

```bash
python src/baseline_diagnostics.py
```

### Проверка данных

```bash
python src/diagnostics.py
```

## Что делает пайплайн

1. Загружает данные из готовых split файлов (train.csv, val.csv, test.csv)
2. Проверяет пересечения пациентов между сплитами (patient-level split)
3. Балансирует валидацию, если в ней только один класс
4. Для каждого сценария:
   - Обучает модель с early stopping
   - Сохраняет лучшие веса
   - Тестирует на отложенной выборке
5. Сохраняет графики и метрики (file-level и patient-level)
6. Сохраняет Grad-CAM визуализации для всех сценариев
7. Агрегирует результаты в сводные таблицы

## Main outputs

### Для каждого сценария

| Файл | Описание |
| --- | --- |
| `model_weights.pth` | Веса модели |
| `training_curves.png` | Графики loss и accuracy |
| `confusion_matrix.png` | Матрица ошибок |
| `roc_curve.png` | ROC-кривая |
| `metrics_file_level.csv` | Метрики на уровне файлов |
| `metrics_patient_level.csv` | Метрики на уровне пациентов |
| `training_history.csv` | История обучения |
| `gradcam_sample_*.png` | Grad-CAM визуализация (3 примера из теста) |

### Сводные таблицы

- `data/summaries/cnn_baseline_summary.csv`
- `data/summaries/transfer_learning_summary.csv`
- `data/summaries/densenet_summary.csv`
- `data/summaries/models_comparison_all.csv`
- `data/summaries/models_comparison_all.png`

## Grad-CAM visualization

Grad-CAM показывает, на какие области спектрограммы модель обращает внимание при принятии решения.

- **Красные области** — высокое внимание
- **Синие области** — низкое внимание

Сохраняется для каждого сценария (3 примера из тестовой выборки).

## Hyperparameters

| Параметр | Значение | Описание |
|----------|----------|----------|
| `SAMPLE_RATE` | 16000 | Частота дискретизации |
| `DURATION` | 3.0 | Длительность аудио (сек) |
| `N_MELS` | 128 | Количество mel-фильтров |
| `LEARNING_RATE` | 5e-4 | Скорость обучения |
| `BATCH_SIZE` | 16 | Размер батча |
| `NUM_EPOCHS` | 15 | Максимум эпох |
| `EARLY_STOPPING_PATIENCE` | 5 | Эпох без улучшения |
| `DROPOUT` | 0.4 | Вероятность отключения нейронов |