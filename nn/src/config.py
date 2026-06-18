"""
config.py
=========
Конфигурация проекта: пути к данным, гиперпараметры, список сценариев.
"""

from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    """
    Конфигурационный класс для проекта.
    
    Содержит все настройки: пути к данным, параметры аудио,
    гиперпараметры обучения и настройки сценариев.
    
    Атрибуты:
        DATA_PATH (Path): Путь к папке с данными.
        OUTPUT_PATH (Path): Путь для сохранения результатов.
        
        SAMPLE_RATE (int): Частота дискретизации аудио.
        DURATION (float): Длительность обрезки аудио в секундах.
        N_MELS (int): Количество mel-фильтров.
        N_FFT (int): Размер окна STFT.
        HOP_LENGTH (int): Шаг между окнами.
        TOP_DB (int): Порог для обрезки спектрограммы (дБ).
        
        NUM_CLASSES (int): Количество классов (2: healthy, parkinson).
        LEARNING_RATE (float): Скорость обучения (Adam).
        BATCH_SIZE (int): Размер батча.
        NUM_EPOCHS (int): Максимальное количество эпох.
        EARLY_STOPPING_PATIENCE (int): Количество эпох без улучшения для остановки.
        DROPOUT (float): Вероятность отключения нейронов.
        AUGMENTATIONS (bool): Применять ли аугментации.
        NUM_WORKERS (int): Количество потоков для загрузки данных.
        
        SEED (int): Seed для воспроизводимости.
        SCENARIOS (list): Список всех сценариев экспериментов.
        
        PATIENT_ID_COLUMN (str): Название колонки с ID пациента в CSV.
        AGGREGATE_BY_PATIENT (bool): Считать метрики по пациентам (усредняя предсказания).
        STRICT_PATIENT_SPLIT (bool): Запрещать пересечение пациентов между сплитами.
        PATIENT_LEVEL_WEIGHTS (bool): Считать веса классов на уровне пациентов.
        AUTO_RESOLVE_DUPLICATES (bool): Автоматически удалять дубликаты файлов из сплитов.
        AUTO_FIX_UNBALANCED_VALIDATION (bool): Балансировать валидацию, если в ней один класс.
        VALIDATION_BALANCE_STRATEGY (str): Стратегия балансировки валидации.
    """
    # Пути
    DATA_PATH = Path("C:\\Users\\Anastasiya\\Desktop\\parkinson_project\\data")
    OUTPUT_PATH = Path("data")
    
    # Параметры аудио
    SAMPLE_RATE = 16000
    DURATION = 3.0
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 512
    TOP_DB = 80
    
    # Параметры обучения
    NUM_CLASSES = 2
    LEARNING_RATE = 5e-4
    BATCH_SIZE = 16
    NUM_EPOCHS = 15
    DROPOUT = 0.4
    EARLY_STOPPING_PATIENCE = 5
    AUGMENTATIONS = True
    NUM_WORKERS = 2
    
    # Воспроизводимость
    SEED = 42
    
    # Сценарии
    SCENARIOS = [
        "single_uams",
        "single_mdvr_kcl",
        "single_ipvs",
        "cross_uams_to_mdvr_kcl",
        "cross_uams_to_ipvs",
        "cross_mdvr_kcl_to_uams",
        "cross_mdvr_kcl_to_ipvs",
        "cross_ipvs_to_mdvr_kcl",
        "cross_ipvs_to_uams",
        "combined_all"
    ]
    
    # Настройки работы с пациентами
    PATIENT_ID_COLUMN = 'patient_id'
    AGGREGATE_BY_PATIENT = True
    STRICT_PATIENT_SPLIT = True
    PATIENT_LEVEL_WEIGHTS = True
    
    AUTO_RESOLVE_DUPLICATES = True
    AUTO_FIX_UNBALANCED_VALIDATION = True
    VALIDATION_BALANCE_STRATEGY = 'balance_equal'

config = Config()