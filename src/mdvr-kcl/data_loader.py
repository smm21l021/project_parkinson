"""
data_loader.py
==============
Единые функции загрузки аудио и метаданных.
Адаптировано для датасета со структурой:

raw_dir/
├── ReadText/
│   ├── HC/          # здоровые (label=0)
│   │   └── *.wav
│   └── PD/          # больные (label=1)
│       └── *.wav
└── SpontaneousDialogue/
    ├── HC/          # здоровые (label=0)
    │   └── *.wav
    └── PD/          # больные (label=1)
        └── *.wav
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


# ─── Парсер для вашей структуры ──────────────────────────────────────────────

def parse_sakar_dataset(
    raw_dir: str | Path,
    dataset_name: str = "sakar",
) -> pd.DataFrame:
    """
    Парсит структуру датасета:
    
    raw_dir/
    ├── ReadText/
    │   ├── HC/          # здоровые
    │   │   └── *.wav
    │   └── PD/          # больные
    │       └── *.wav
    └── SpontaneousDialogue/
        ├── HC/
        │   └── *.wav
        └── PD/
            └── *.wav
    """
    raw_dir = Path(raw_dir)
    records = []
    
    if not raw_dir.exists():
        logger.error(f"Raw directory not found: {raw_dir}")
        return pd.DataFrame()
    
    # Обходим все wav файлы рекурсивно
    wav_files = list(raw_dir.rglob("*.wav")) + list(raw_dir.rglob("*.WAV"))
    
    if len(wav_files) == 0:
        logger.error(f"No .wav files found in {raw_dir}")
        return pd.DataFrame()
    
    for wav_file in sorted(wav_files):
        # Получаем относительный путь
        rel_path = wav_file.relative_to(raw_dir)
        parts = rel_path.parts
        
        # Определяем тип речи (ReadText / SpontaneousDialogue)
        if len(parts) >= 1:
            speech_type = parts[0]
        else:
            speech_type = "unknown"
        
        # Определяем класс по имени папки (HC или PD)
        label = None
        label_name = None
        
        if len(parts) >= 2:
            folder_name = parts[1].upper()
            if folder_name == "HC":
                label = 0
                label_name = "healthy"
            elif folder_name == "PD":
                label = 1
                label_name = "parkinson"
        
        # Если не определили по папке, пробуем по имени файла
        if label is None:
            filename = wav_file.name.lower()
            if "_hc_" in filename:
                label = 0
                label_name = "healthy"
            elif "_pd_" in filename:
                label = 1
                label_name = "parkinson"
            else:
                logger.warning(f"Cannot determine label for: {wav_file}")
                continue
        
        # Извлекаем ID пациента из имени файла
        # Формат: ID00_hc_0_0_0.wav
        patient_id_raw = wav_file.stem.split("_")[0]  # ID00
        # Добавляем тип речи, чтобы различать одинаковые ID из разных групп
        patient_id = f"{dataset_name}_{speech_type}_{patient_id_raw}"
        
        records.append({
            "file_path": str(wav_file),
            "label": label,
            "label_name": label_name,
            "patient_id": patient_id,
            "recording_id": 1,
            "dataset": dataset_name,
            "filename": wav_file.name,
            "speech_type": speech_type,  # ReadText или SpontaneousDialogue
            "original_patient_id": patient_id_raw,
        })
    
    df = pd.DataFrame(records)
    
    if len(df) == 0:
        logger.error(f"No valid records created from {raw_dir}")
    else:
        logger.info(f"[{dataset_name}] Found {len(df)} files")
        logger.info(f"  Parkinson (label=1): {(df['label']==1).sum()}")
        logger.info(f"  Healthy (label=0): {(df['label']==0).sum()}")
        logger.info(f"  Unique patients: {df['patient_id'].nunique()}")
        logger.info(f"  Speech types: {df['speech_type'].unique().tolist()}")
    
    return df


def parse_generic_dataset(
    raw_dir: str | Path,
    dataset_name: str = "generic",
    pd_folder: str = "parkinson",
    healthy_folder: str = "healthy",
) -> pd.DataFrame:
    """
    Парсит папку с простой структурой:
        raw_dir/
          parkinson/    (папка с больными)
            *.wav
          healthy/      (папка со здоровыми)
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
    
    if len(df) > 0:
        logger.info(f"[{dataset_name}] Found {len(df)} files")
        logger.info(f"  Parkinson (label=1): {(df['label']==1).sum()}")
        logger.info(f"  Healthy (label=0): {(df['label']==0).sum()}")
    else:
        logger.warning(f"No files found in {raw_dir} using generic parser")
    
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