"""
data_loader.py
==============
Единые функции загрузки аудио и метаданных.
Адаптировано для датасета с файлами в папках:
  - data/raw/parkinson/  (файлы *.wav)
  - data/raw/healthy/    (файлы *.wav)
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


# ─── Парсер для структуры с папками parkinson/ и healthy/ ───────────────────

def parse_sakar_dataset(
    raw_dir: str | Path,
    dataset_name: str = "sakar",
) -> pd.DataFrame:
    """
    Парсит структуру датасета, где класс определяется по папке:
        raw_dir/
          parkinson/   (или PD, parkinsons)
            *.wav
          healthy/     (или HC, control, healthy)
            *.wav
    
    Для вашего датасета:
      - Больные: raw_dir/parkinson/*.wav
      - Здоровые: raw_dir/healthy/*.wav
    """
    raw_dir = Path(raw_dir)
    records = []
    
    # Определяем возможные названия папок для каждого класса
    pd_folders = ["parkinson", "parkinsons", "pd", "PD", "Parkinson"]
    healthy_folders = ["healthy", "hc", "HC", "control", "Healthy", "health"]
    
    # Сканируем папки
    for folder in raw_dir.iterdir():
        if not folder.is_dir():
            continue
        
        folder_name = folder.name
        
        # Определяем метку по имени папки
        if folder_name in pd_folders or folder_name.lower() == "parkinson":
            label = 1
            label_name = "parkinson"
        elif folder_name in healthy_folders or folder_name.lower() == "healthy":
            label = 0
            label_name = "healthy"
        else:
            logger.warning(f"Unknown folder: {folder_name}, skipping")
            continue
        
        # Обрабатываем все wav файлы в папке
        wav_files = list(folder.glob("*.wav")) + list(folder.glob("*.WAV"))
        for wav_file in sorted(wav_files):
            # Формируем patient_id из имени файла (без расширения)
            stem = wav_file.stem
            
            # Используем имя файла как ID пациента (каждый файл = отдельный пациент)
            patient_id = f"{dataset_name}_{label_name}_{stem}"
            
            records.append({
                "file_path": str(wav_file),
                "label": label,
                "label_name": label_name,
                "patient_id": patient_id,
                "recording_id": 1,
                "dataset": dataset_name,
                "filename": wav_file.name,
            })
    
    df = pd.DataFrame(records)
    
    if len(df) == 0:
        logger.error(f"No files found in {raw_dir}")
        logger.info(f"Expected structure: {raw_dir}/parkinson/*.wav and {raw_dir}/healthy/*.wav")
    else:
        logger.info(
            f"[{dataset_name}] Found {len(df)} files, "
            f"{df['label'].sum()} PD, {(df['label']==0).sum()} healthy"
        )
    
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
        
        wav_files = list(folder_path.glob("*.wav")) + list(folder_path.glob("*.WAV"))
        for wav_file in sorted(wav_files):
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