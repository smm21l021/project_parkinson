"""
dataset.py
==========
Загрузка данных, создание PyTorch Dataset, разбиение по сценариям.
"""

import torch
import numpy as np
import pandas as pd
import librosa
from torch.utils.data import Dataset
from pathlib import Path
from config import config
from collections import Counter

class ParkinsonDataset(Dataset):
    """
    PyTorch Dataset для загрузки аудиофайлов и преобразования в спектрограммы.
    
    Загружает аудио, обрезает/дополняет до 3 секунд, строит mel-спектрограмму,
    нормализует в [0, 1] и возвращает тензор.
    
    Параметры:
        audio_paths (list): Список путей к аудиофайлам.
        labels (list): Список меток (0 - healthy, 1 - parkinson).
        patient_ids (list, optional): Список ID пациентов. По умолчанию None.
            Используется для patient-level валидации.
        augment (bool, optional): Применять ли аугментации. По умолчанию False.
    
    Возвращает:
        tuple: (spectrogram, label)
            - spectrogram: torch.Tensor формы (1, N_MELS, time_frames)
            - label: torch.Tensor формы (1,)
    """
    def __init__(self, audio_paths, labels, patient_ids=None, augment=False):
        self.audio_paths = audio_paths
        self.labels = labels
        self.patient_ids = patient_ids
        self.augment = augment
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio, _ = librosa.load(self.audio_paths[idx], sr=16000)
        target_len = int(16000 * 3.0)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        
        if self.augment:
            audio = self._apply_augmentations(audio)

        mel = librosa.feature.melspectrogram(y=audio, sr=16000,
                                              n_fft=1024, hop_length=512, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        
        spec = torch.FloatTensor(mel_db).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.patient_ids is not None:
            return spec, label, self.patient_ids[idx]
        return spec, label

    def _apply_augmentations(self, audio):
        if np.random.rand() < 0.5:
            noise_amp = 0.005 * np.random.uniform() * np.max(audio)
            audio = audio + noise_amp * np.random.normal(size=audio.shape)
        if np.random.rand() < 0.5:
            shift = int(0.1 * len(audio) * (np.random.rand() - 0.5))
            audio = np.roll(audio, shift)
        if np.random.rand() < 0.3:
            rate = np.random.uniform(0.9, 1.1)
            try:
                orig_sr = 16000
                new_sr = int(round(orig_sr * rate))
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=new_sr)
            except Exception:
                try:
                    audio = librosa.effects.time_stretch(audio, rate)
                except Exception:
                    pass
            target_len = int(16000 * 3.0)
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]
        return audio


def print_split_stats(name, split_name, paths, labels, pids=None):
    """
    Печатает статистику по сплиту: количество файлов, пациентов, соотношение классов.
    """
    n_files = len(paths)
    n_patients = len(set(pids)) if pids else 'N/A'
    healthy = labels.count(0)
    parkinson = labels.count(1)
    total = healthy + parkinson
    healthy_pct = (healthy / total * 100) if total > 0 else 0
    parkinson_pct = (parkinson / total * 100) if total > 0 else 0
    
    print(f"    {split_name:5s}: files={n_files:3d}, patients={n_patients}, "
          f"Healthy={healthy:3d} ({healthy_pct:.1f}%), "
          f"Parkinson={parkinson:3d} ({parkinson_pct:.1f}%)")


def load_datasets(data_path, seed=42):
    """
    Загружает датасеты из готовых split файлов.
    Минимальная балансировка: если в сплите один класс.
    """
    datasets = {}
    
    for name in ["UAMS", "MDVR-KCL", "IPVS"]:
        splits_dir = data_path / name / "splits"
        processed_dir = data_path / name / "processed"
        
        if not processed_dir.exists():
            print(f"{name}: папка processed не найдена")
            continue
        
        available_files = {f.name: str(f) for f in processed_dir.glob("*.wav")}
        print(f"\n{name}: {len(available_files)} файлов в processed")
        
        train_df = pd.read_csv(splits_dir / "train.csv")
        val_df = pd.read_csv(splits_dir / "val.csv")
        test_df = pd.read_csv(splits_dir / "test.csv")

        pid_col = config.PATIENT_ID_COLUMN
        
        def extract_from_split(df, split_name):
            paths, labels, pids = [], [], []
            for _, row in df.iterrows():
                if 'processed_path' in df.columns and pd.notna(row['processed_path']):
                    filename = Path(row['processed_path']).name
                elif 'file_path' in df.columns and pd.notna(row['file_path']):
                    filename = Path(row['file_path']).name
                else:
                    continue
                if filename in available_files:
                    paths.append(available_files[filename])
                    labels.append(row['label'])
                    pids.append(row[pid_col])
            print(f"{split_name}: найдено {len(paths)}/{len(df)} файлов")
            return paths, labels, pids
        
        train_paths, train_labels, train_pids = extract_from_split(train_df, "train")
        val_paths, val_labels, val_pids = extract_from_split(val_df, "val")
        test_paths, test_labels, test_pids = extract_from_split(test_df, "test")
        
        if len(train_paths) == 0:
            print(f"Нет данных для {name}!")
            continue

        unique_val = set(val_labels)
        if len(unique_val) <= 1 and config.AUTO_FIX_UNBALANCED_VALIDATION:
            print(f"  Валидация содержит один класс: {unique_val}. Исправляем...")
            target_class = 0 if 1 in unique_val else 1
            candidates = []
            for pid, lbl in zip(train_pids, train_labels):
                if lbl == target_class and pid not in val_pids:
                    candidates.append(pid)
            if candidates:
                moved_pid = candidates[0]
                indices = [i for i, p in enumerate(train_pids) if p == moved_pid]
                for idx in sorted(indices, reverse=True):
                    val_paths.append(train_paths.pop(idx))
                    val_labels.append(train_labels.pop(idx))
                    val_pids.append(train_pids.pop(idx))
                print(f"    Перемещён пациент {moved_pid} в val")
            else:
                print(f"    Не найден пациент класса {target_class} в train")

        print(f"\n{name}:")
        print_split_stats(name, "train", train_paths, train_labels, train_pids)
        print_split_stats(name, "val", val_paths, val_labels, val_pids)
        print_split_stats(name, "test", test_paths, test_labels, test_pids)
        
        datasets[name] = {
            'train_paths': train_paths,
            'train_labels': train_labels,
            'train_pids': train_pids,
            'val_paths': val_paths,
            'val_labels': val_labels,
            'val_pids': val_pids,
            'test_paths': test_paths,
            'test_labels': test_labels,
            'test_pids': test_pids
        }

    # COMBINED
    combined = {
        'train_paths': [], 'train_labels': [],
        'val_paths': [], 'val_labels': [],
        'test_paths': [], 'test_labels': []
    }
    for data in datasets.values():
        for key in ['train', 'val', 'test']:
            combined[f'{key}_paths'].extend(data[f'{key}_paths'])
            combined[f'{key}_labels'].extend(data[f'{key}_labels'])
    
    datasets['COMBINED'] = combined
    
    print(f"\nCOMBINED:")
    print_split_stats("COMBINED", "train", combined['train_paths'], combined['train_labels'])
    print_split_stats("COMBINED", "val", combined['val_paths'], combined['val_labels'])
    print_split_stats("COMBINED", "test", combined['test_paths'], combined['test_labels'])
    
    return datasets


def get_data_for_scenario(scenario, datasets):
    if scenario == "combined_all":
        return datasets['COMBINED'], datasets['COMBINED']
    
    if scenario.startswith("cross_"):
        parts = scenario.replace("cross_", "").split("_to_")
        train_name = parts[0].upper()
        test_name = parts[1].upper()
        name_map = {"MDVR_KCL": "MDVR-KCL", "UAMS": "UAMS", "IPVS": "IPVS", "IRVS": "IPVS"}
        train_name = name_map.get(train_name, train_name)
        test_name = name_map.get(test_name, test_name)
        return datasets.get(train_name), datasets.get(test_name)
    
    if scenario.startswith("single_"):
        name = scenario.replace("single_", "").upper()
        name_map = {"MDVR_KCL": "MDVR-KCL", "UAMS": "UAMS", "IPVS": "IPVS", "IRVS": "IPVS"}
        name = name_map.get(name, name)
        return datasets.get(name), datasets.get(name)
    
    return None, None