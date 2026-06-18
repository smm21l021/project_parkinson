"""
baseline_diagnostics.py
=======================
Автономный скрипт для быстрого сравнения базовых моделей на IPVS (5 эпох).

НЕ зависит от основного кода пайплайна.
Использует собственную загрузку данных с patient-level разбиением.
Помогает выбрать лучшую модель для полного обучения.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import re
import random
import warnings
warnings.filterwarnings('ignore')

class Config:
    """
    Локальная конфигурация для диагностики.
    
    Используется только в этом скрипте.
    Параметры обучения упрощены для быстрого запуска (5 эпох).
    """
    DATA_PATH = Path("C:\\Users\\Anastasiya\\Desktop\\parkinson_project\\data")
    SAMPLE_RATE = 16000
    DURATION = 3.0
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 512
    TOP_DB = 80
    NUM_CLASSES = 2
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 16
    NUM_EPOCHS = 5
    EARLY_STOPPING_PATIENCE = 3
    SEED = 42

config = Config()

class ParkinsonDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset для загрузки аудио и преобразования в спектрограммы.
    
    Упрощённая версия для диагностики.
    Не использует аугментации по умолчанию.
    
    Параметры:
        audio_paths (list): Список путей к аудиофайлам.
        labels (list): Список меток (0 - healthy, 1 - parkinson).
        patient_ids (list, optional): Список ID пациентов.
        augment (bool, optional): Применять ли аугментации.
    """
    def __init__(self, audio_paths, labels, patient_ids=None, augment=False):
        import librosa
        self.audio_paths = audio_paths
        self.labels = labels
        self.patient_ids = patient_ids
        self.augment = augment
    def __len__(self):
        return len(self.audio_paths)
    def __getitem__(self, idx):
        import librosa
        import numpy as np
        audio, _ = librosa.load(self.audio_paths[idx], sr=16000)
        target_len = int(16000 * 3.0)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        if self.augment:
            if np.random.rand() < 0.5:
                noise_amp = 0.005 * np.random.uniform() * np.max(audio)
                audio = audio + noise_amp * np.random.normal(size=audio.shape)
        mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=1024, hop_length=512, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        return torch.FloatTensor(mel_db).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.long)

def load_ipvs_patient_split(data_path, seed=42):
    """
    Загружает только датасет IPVS с patient-level разбиением.
    
    Использует колонку 'split' из metadata.csv.
    Гарантирует, что один пациент не попадёт в разные сплиты.
    
    Параметры:
        data_path (Path): Путь к папке с данными.
        seed (int): Seed для воспроизводимости.
    
    Возвращает:
        dict: Словарь с train/val/test путями и метками.
    """
    import pandas as pd
    from pathlib import Path
    
    name = "IPVS"
    metadata_path = data_path / name / "metadata.csv"
    processed_dir = data_path / name / "processed"
    
    if not processed_dir.exists():
        print(f"ОШИБКА: папка processed не найдена для {name}")
        return None
    
    available_files = {f.name: str(f) for f in processed_dir.glob("*.wav")}
    print(f"\n{name}: {len(available_files)} файлов в processed")
    
    df = pd.read_csv(metadata_path)
    print(f"Строк в metadata: {len(df)}")
    
    # Группируем файлы по пациентам
    patient_files = {}
    patient_labels = {}
    patient_splits = {}
    
    for _, row in df.iterrows():
        raw_path = row['processed_path']
        filename = re.sub(r'^.*[\\/]', '', raw_path)
        
        if filename in available_files:
            pid = row['patient_id']
            split = row['split']
            
            if pid not in patient_files:
                patient_files[pid] = []
                patient_labels[pid] = row['label']
                patient_splits[pid] = split
            
            if patient_splits[pid] == split:
                patient_files[pid].append(available_files[filename])
    
    # Группируем по сплитам
    split_data = {
        'train': {'paths': [], 'labels': []},
        'val': {'paths': [], 'labels': []},
        'test': {'paths': [], 'labels': []}
    }
    
    for pid, files in patient_files.items():
        split = patient_splits.get(pid, 'train')
        label = patient_labels[pid]
        for f in files:
            split_data[split]['paths'].append(f)
            split_data[split]['labels'].append(label)
    
    # Проверка: нет ли валидации с одним классом
    val_labels = split_data['val']['labels']
    if len(set(val_labels)) < 2:
        print(f"\nПРЕДУПРЕЖДЕНИЕ: В валидации только один класс: {set(val_labels)}")
        print("Исправление путем перемещения пациентов из train в val...")
        
        train_patients = {}
        for pid, files in patient_files.items():
            if patient_splits.get(pid) == 'train':
                train_patients[pid] = patient_labels[pid]
        
        missing_class = 1 if 0 in set(val_labels) else 0
        candidates = [pid for pid, lbl in train_patients.items() if lbl == missing_class]
        
        if candidates:
            moved_pid = candidates[0]
            moved_files = patient_files[moved_pid]
            
            for f in moved_files:
                if f in split_data['train']['paths']:
                    idx = split_data['train']['paths'].index(f)
                    split_data['train']['paths'].pop(idx)
                    split_data['train']['labels'].pop(idx)
            
            for f in moved_files:
                split_data['val']['paths'].append(f)
                split_data['val']['labels'].append(patient_labels[moved_pid])
            
            print(f"  Перемещен пациент {moved_pid} из train в val")
    
    train_paths = split_data['train']['paths']
    train_labels = split_data['train']['labels']
    val_paths = split_data['val']['paths']
    val_labels = split_data['val']['labels']
    test_paths = split_data['test']['paths']
    test_labels = split_data['test']['labels']
    
    print(f"\nИТОГОВЫЕ СПЛИТЫ:")
    print(f"Train: {len(train_paths)} файлов")
    print(f"  Healthy: {train_labels.count(0)}, Parkinson: {train_labels.count(1)}")
    print(f"Val: {len(val_paths)} файлов")
    print(f"  Healthy: {val_labels.count(0)}, Parkinson: {val_labels.count(1)}")
    print(f"Test: {len(test_paths)} файлов")
    print(f"  Healthy: {test_labels.count(0)}, Parkinson: {test_labels.count(1)}")
    
    return {
        'train_paths': train_paths,
        'train_labels': train_labels,
        'val_paths': val_paths,
        'val_labels': val_labels,
        'test_paths': test_paths,
        'test_labels': test_labels
    }

class CNNBaseline(nn.Module):
    """
    Базовая свёрточная нейросеть.
    Та же архитектура, что и в основном коде.
    """
    def __init__(self, num_classes=2):
        super(CNNBaseline, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(256 * 4 * 4, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class TransferLearningModel(nn.Module):
    """
    Модель с Transfer Learning на базе ResNet18.
    """
    def __init__(self, num_classes=2):
        super(TransferLearningModel, self).__init__()
        import torchvision.models as models
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(num_features, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
        for name, param in self.backbone.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
    def forward(self, x):
        return self.backbone(x)

class DenseNetModel(nn.Module):
    """
    Модель с Transfer Learning на базе DenseNet121.
    """
    def __init__(self, num_classes=2):
        super(DenseNetModel, self).__init__()
        import torchvision.models as models
        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.backbone.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(num_features, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
        for name, param in self.backbone.named_parameters():
            if 'denseblock4' not in name and 'classifier' not in name:
                param.requires_grad = False
    def forward(self, x):
        return self.backbone(x)

def get_model_by_name(model_name, num_classes=2):
    if model_name == "cnn_baseline":
        return CNNBaseline(num_classes)
    elif model_name == "transfer_learning":
        return TransferLearningModel(num_classes)
    elif model_name == "densenet":
        return DenseNetModel(num_classes)
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

def evaluate_model(model, val_loader, device):
    """
    Оценивает модель на валидационной выборке.
    
    Параметры:
        model (nn.Module): Модель для оценки.
        val_loader (DataLoader): Загрузчик валидационных данных.
        device (torch.device): Устройство для вычислений.
    
    Возвращает:
        dict: Словарь с метриками (f1, roc_auc, accuracy).
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for specs, labels in val_loader:
            specs, labels = specs.to(device), labels.to(device)
            outputs = model(specs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    roc_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    accuracy = accuracy_score(all_labels, all_preds)
    return {'f1': f1, 'roc_auc': roc_auc, 'accuracy': accuracy}

def train_model(model_name, train_data, val_data, device):
    """
    Обучает одну модель на IPVS (5 эпох) и возвращает метрики.
    
    Параметры:
        model_name (str): Название модели.
        train_data (dict): Данные для обучения.
        val_data (dict): Данные для валидации.
        device (torch.device): Устройство для вычислений.
    
    Возвращает:
        dict: Результаты обучения (метрики, время).
    """
    print(f"\n{'='*30}")
    print(f"ОБУЧЕНИЕ: {model_name.upper()}")
    print(f"{'='*30}")
    
    train_dataset = ParkinsonDataset(train_data['train_paths'], train_data['train_labels'], augment=False)
    val_dataset = ParkinsonDataset(val_data['val_paths'], val_data['val_labels'], augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = get_model_by_name(model_name, config.NUM_CLASSES)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_f1 = 0
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        metrics = evaluate_model(model, val_loader, device)
        print(f"  Epoch {epoch+1}: Val F1={metrics['f1']:.4f}, Val Acc={metrics['accuracy']:.4f}")
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping после {epoch+1} эпох")
                break
    
    elapsed = time.time() - start_time
    final_metrics = evaluate_model(model, val_loader, device)
    
    print(f"\nРезультаты для {model_name.upper()}:")
    print(f"  F1: {final_metrics['f1']:.4f}")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  ROC-AUC: {final_metrics['roc_auc']:.4f}")
    print(f"  Время: {elapsed/60:.1f} мин")
    
    return {
        'model': model_name,
        'f1': final_metrics['f1'],
        'accuracy': final_metrics['accuracy'],
        'roc_auc': final_metrics['roc_auc'],
        'time_min': elapsed/60
    }

def run_baseline_diagnostics():
    """
    Запускает диагностику базовых моделей.
    
    Этапы:
        1. Загрузка IPVS с patient-level разбиением
        2. Обучение трёх моделей (5 эпох)
        3. Сравнение результатов
        4. Выбор лучшей модели
        5. Сохранение результатов и графиков
    
    Возвращает:
        tuple: (best_model, results_df) — лучшая модель и таблица результатов.
    """
    print("="*30)
    print("ДИАГНОСТИКА БАЗОВЫХ МОДЕЛЕЙ (5 эпох)")
    print("="*30)
    
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}\n")
    
    data = load_ipvs_patient_split(config.DATA_PATH, config.SEED)
    if not data:
        print("ОШИБКА: Нет данных!")
        return None
    
    print(f"\nДАННЫЕ ЗАГРУЖЕНЫ УСПЕШНО")
    
    models_to_test = ["cnn_baseline", "transfer_learning", "densenet"]
    results = []
    
    for model_name in models_to_test:
        result = train_model(model_name, data, data, device)
        results.append(result)
    
    # Compare
    print("\n" + "="*30)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("="*30)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1', ascending=False)
    print("\n" + results_df.to_string(index=False))
    
    best_model = results_df.iloc[0]['model']
    best_f1 = results_df.iloc[0]['f1']
    
    print("\n" + "="*30)
    print(f"ЛУЧШАЯ МОДЕЛЬ: {best_model.upper()}")
    print(f"   F1 Score: {best_f1:.4f}")
    print("="*30)
    
    Path("data/summaries").mkdir(parents=True, exist_ok=True)
    results_df.to_csv("data/summaries/baseline_diagnostics.csv", index=False)
    print(f"\nРезультаты сохранены в: data/summaries/baseline_diagnostics.csv")
    

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    bars = plt.bar(results_df['model'], results_df['f1'], color=['#9b59b6', '#8e44ad', '#6c3483'])
    plt.ylim(0, 1)
    plt.ylabel('F1 Score')
    plt.title('Сравнение F1 Score')
    for bar, val in zip(bars, results_df['f1']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', fontsize=11)
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Случайно (0.5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    x = np.arange(len(results_df))
    width = 0.35
    plt.bar(x - width/2, results_df['accuracy'], width, label='Accuracy', color='#3498db')
    plt.bar(x + width/2, results_df['roc_auc'], width, label='ROC-AUC', color='#2ecc71')
    plt.xticks(x, results_df['model'])
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Accuracy vs ROC-AUC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("data/summaries/baseline_diagnostics.png", dpi=150)
    plt.show()
    
    print("\n" + "="*30)
    print("РЕКОМЕНДАЦИЯ")
    print("="*30)
    print(f"Используйте {best_model} для полного обучения")
    print(f"Обновите config.py: MODEL_TO_USE = \"{best_model}\"")
    print("="*30)
    
    return best_model, results_df

if __name__ == "__main__":
    import time
    best_model, df = run_baseline_diagnostics()