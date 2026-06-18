"""
data_loader.py
==============
Единые функции загрузки аудио и метаданных.
Адаптировано для датасета с иерархической структурой:

raw_dir/
├── 15 Young Healthy Control/        # здоровые
│   ├── Имя пациента 1/
│   │   ├── файл1.wav
│   │   └── ...
│   └── Имя пациента 2/
│       └── ...
├── 22 Elderly Healthy Control/      # здоровые
│   └── ...
└── 28 People with Parkinson's disease/  # больные
    └── ...
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa

logger = logging.getLogger(__name__)

# Стандартные метки
LABEL_MAP = {0: "healthy", 1: "parkinson"}
LABEL_INVERSE = {"healthy": 0, "parkinson": 1, "pd": 1, "hc": 0, "h": 0, "p": 1}

TARGET_SR = 16_000


# ─── Загрузка метаданных ──────────────────────────────────────────────────────

def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    """Загружает metadata.csv."""
    df = pd.read_csv(str(csv_path))
    required = {"file_path", "label", "patient_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing columns: {missing}")
    if df["label"].dtype == object:
        df["label"] = df["label"].map(LABEL_INVERSE).fillna(df["label"]).astype(int)
    logger.info(f"Loaded metadata: {len(df)} records")
    return df


def load_split(splits_dir: str | Path, split_name: str) -> pd.DataFrame:
    """Загружает один из сплитов train/val/test."""
    splits_dir = Path(splits_dir)
    path = splits_dir / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    df = pd.read_csv(str(path))
    logger.info(f"Loaded split '{split_name}': {len(df)} records")
    return df


# ─── Парсер для иерархической структуры ──────────────────────────────────────

def parse_sakar_dataset(
    raw_dir: str | Path,
    dataset_name: str = "sakar",
) -> pd.DataFrame:
    """
    Парсит иерархическую структуру датасета:
    
    raw_dir/
    ├── 15 Young Healthy Control/        # здоровые (label=0)
    │   ├── Patient1/
    │   │   ├── file1.wav
    │   │   └── ...
    │   └── Patient2/
    │       └── ...
    ├── 22 Elderly Healthy Control/      # здоровые (label=0)
    │   └── ...
    └── 28 People with Parkinson's disease/  # больные (label=1)
        └── ...
    """
    raw_dir = Path(raw_dir)
    records = []
    
    # Определяем папки верхнего уровня и их метки
    # Ключевые слова для определения класса по имени папки
    healthy_keywords = ["healthy", "control", "young", "elderly"]
    parkinson_keywords = ["parkinson", "pd", "disease"]
    
    # Рекурсивно обходим все wav файлы
    for wav_file in raw_dir.rglob("*.wav"):
        # Получаем относительный путь для анализа
        rel_path = wav_file.relative_to(raw_dir)
        parts = rel_path.parts
        
        # Определяем класс по пути
        label = None
        label_name = None
        
        # Проверяем каждый уровень пути на ключевые слова
        for part in parts:
            part_lower = part.lower()
            
            # Проверка на здоровых
            if any(kw in part_lower for kw in healthy_keywords):
                label = 0
                label_name = "healthy"
                break
            # Проверка на больных
            elif any(kw in part_lower for kw in parkinson_keywords):
                label = 1
                label_name = "parkinson"
                break
        
        # Если не определили по ключевым словам, используем имя родительской папки
        if label is None:
            # Берем имя первой папки верхнего уровня
            top_folder = parts[0] if parts else ""
            if "young" in top_folder.lower() or "elderly" in top_folder.lower():
                label = 0
                label_name = "healthy"
            elif "parkinson" in top_folder.lower():
                label = 1
                label_name = "parkinson"
            else:
                logger.warning(f"Cannot determine label for: {wav_file}")
                continue
        
        # Формируем ID пациента (имя папки второго уровня или имя файла)
        if len(parts) >= 2:
            patient_name = parts[1]  # папка с именем пациента
        else:
            patient_name = wav_file.stem
        
        patient_id = f"{dataset_name}_{label_name}_{patient_name}"
        
        records.append({
            "file_path": str(wav_file),
            "label": label,
            "label_name": label_name,
            "patient_id": patient_id,
            "recording_id": 1,
            "dataset": dataset_name,
            "filename": wav_file.name,
            "patient_folder": patient_name,
            "group_folder": parts[0] if parts else "",
        })
    
    df = pd.DataFrame(records)
    
    if len(df) == 0:
        logger.error(f"No .wav files found in {raw_dir}")
        logger.info("Expected structure: raw_dir/[group_folder]/[patient_folder]/*.wav")
    else:
        logger.info(
            f"[{dataset_name}] Found {len(df)} files, "
            f"{df['label'].sum()} PD, {(df['label']==0).sum()} healthy"
        )
        logger.info(f"  Groups found: {df['group_folder'].unique().tolist()}")
        logger.info(f"  Patients: {df['patient_id'].nunique()}")
    
    return df


def parse_generic_dataset(
    raw_dir: str | Path,
    dataset_name: str = "generic",
    pd_folder: str = "parkinson",
    healthy_folder: str = "healthy",
) -> pd.DataFrame:
    """
    Парсит папку со структурой:
        raw_dir/
          parkinson/  (или указанная папка)
            *.wav
          healthy/    (или указанная папка)
            *.wav
    """
    raw_dir = Path(raw_dir)
    records = []
    folder_label = {pd_folder: 1, healthy_folder: 0}
    
    for folder, label in folder_label.items():
        folder_path = raw_dir / folder
        if not folder_path.exists():
            logger.warning(f"Folder not found: {folder_path}")
            continue
        
        for wav_file in folder_path.rglob("*.wav"):
            records.append({
                "file_path": str(wav_file),
                "label": label,
                "label_name": LABEL_MAP[label],
                "patient_id": f"{dataset_name}_{wav_file.stem}",
                "recording_id": 1,
                "dataset": dataset_name,
                "filename": wav_file.name,
            })
    
    df = pd.DataFrame(records)
    logger.info(
        f"[{dataset_name}] Found {len(df)} files, "
        f"{df['label'].sum()} PD, {(df['label']==0).sum()} healthy"
    )
    return df


def combine_datasets(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Объединяет несколько датафреймов метаданных."""
    combined = pd.concat(dfs, ignore_index=True)
    dup = combined[combined.duplicated(["patient_id", "filename"], keep=False)]
    if len(dup) > 0:
        logger.warning(f"Found {len(dup)} duplicate records by patient_id+filename")
    logger.info(
        f"Combined dataset: {len(combined)} records, "
        f"{combined['patient_id'].nunique()} unique patients"
    )
    return combined


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class ParkinsonsDataset(Dataset):
    """PyTorch Dataset для аудиофайлов болезни Паркинсона."""
    
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        sr: int = TARGET_SR,
        max_duration: Optional[float] = 3.0,
        processed_dir: Optional[str | Path] = None,
        transform=None,
    ):
        self.meta = metadata_df.reset_index(drop=True)
        self.sr = sr
        self.max_len = int(max_duration * sr) if max_duration else None
        self.processed_dir = Path(processed_dir) if processed_dir else None
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.meta)
    
    def __getitem__(self, idx: int):
        row = self.meta.iloc[idx]
        
        if self.processed_dir is not None:
            file_path = self.processed_dir / row["filename"]
        else:
            file_path = Path(row["file_path"])
        
        y, _ = librosa.load(str(file_path), sr=self.sr, mono=True)
        
        if self.max_len is not None:
            if len(y) < self.max_len:
                y = np.pad(y, (0, self.max_len - len(y)), mode="constant")
            else:
                y = y[:self.max_len]
        
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        
        if self.transform:
            y_tensor = self.transform(y_tensor)
        
        label = int(row["label"])
        return y_tensor, label
    
    def get_class_weights(self) -> torch.Tensor:
        counts = self.meta["label"].value_counts().sort_index()
        weights = 1.0 / counts.values
        weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float32)