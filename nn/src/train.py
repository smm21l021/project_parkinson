"""
train.py
========
Функции обучения, валидации, тестирования и сохранения результатов.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
import random

from dataset import ParkinsonDataset
from config import config


def seed_worker(worker_id):
    """
    Функция для инициализации seed в каждом воркере DataLoader.
    
    Используется для воспроизводимости при многопоточной загрузке данных.
    
    Параметры:
        worker_id (int): ID воркера.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_and_evaluate(model, model_name, train_data, test_data, scenario_name, device, config):
    """
    Полный цикл обучения, валидации и тестирования модели.
    
    Этапы:
        1. Создание DataLoader'ов для train/val/test
        2. Цикл обучения с early stopping
        3. Сохранение лучшей модели (по validation loss)
        4. Тестирование на отложенной выборке
        5. Построение графиков (обучение, confusion matrix, ROC)
        6. Grad-CAM визуализация
        7. Сохранение метрик и истории обучения
    
    Параметры:
        model (nn.Module): Модель для обучения.
        model_name (str): Название модели ("cnn_baseline", "transfer_learning", "densenet").
        train_data (dict): Словарь с ключами 'train_paths', 'train_labels', 
            'val_paths', 'val_labels', 'train_pids', 'val_pids'.
        test_data (dict): Словарь с ключами 'test_paths', 'test_labels', 'test_pids'.
        scenario_name (str): Название сценария (например, "single_uams").
        device (torch.device): Устройство для вычислений (cuda/cpu).
        config (Config): Объект с гиперпараметрами.
    
    Возвращает:
        dict: Словарь с метриками:
            - accuracy (float)
            - precision (float)
            - recall (float)
            - f1_score (float)
            - roc_auc (float)
            (если включена patient-level агрегация, возвращаются patient-level метрики)
    """
    folder = Path(f"data/{model_name}/scenarios/{scenario_name}")
    folder.mkdir(parents=True, exist_ok=True)
    
    # Создание датасетов и загрузчиков данных
    train_dataset = ParkinsonDataset(train_data['train_paths'], train_data['train_labels'], patient_ids=train_data.get('train_pids', None), augment=config.AUGMENTATIONS)
    val_dataset = ParkinsonDataset(train_data['val_paths'], train_data['val_labels'], patient_ids=train_data.get('val_pids', None), augment=False)
    test_dataset = ParkinsonDataset(test_data['test_paths'], test_data['test_labels'], patient_ids=test_data.get('test_pids', None), augment=False)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=False, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=False, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=False, worker_init_fn=seed_worker)
    
    model = model.to(device)
    # Вычисление весов классов для учета дисбаланса
    if config.PATIENT_LEVEL_WEIGHTS and train_data.get('train_pids') is not None:
        pid_to_labels = {}
        for pid, lbl in zip(train_data['train_pids'], train_data['train_labels']):
            pid_to_labels.setdefault(pid, []).append(lbl)
        pid_label = {pid: int(round(np.mean(lbls))) for pid, lbls in pid_to_labels.items()}
        unique, counts = np.unique(list(pid_label.values()), return_counts=True)
        class_counts = dict(zip(unique.tolist(), counts.tolist()))
    else:
        labels = train_data['train_labels']
        unique, counts = np.unique(labels, return_counts=True)
        class_counts = dict(zip(unique.tolist(), counts.tolist()))

    total = sum(counts)
    weights = [0.0] * config.NUM_CLASSES
    for cls in range(config.NUM_CLASSES):
        if cls in class_counts:
            weights[cls] = total / (class_counts[cls] + 1e-8)
        else:
            weights[cls] = 0.0
    weights = torch.tensor(weights, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE)
    
    best_val_loss = float('inf')
    patience = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Цикл обучения
    for epoch in range(config.NUM_EPOCHS):
        # Обучение
        model.train()
        train_loss = 0
        train_correct = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} training"):
            if len(batch) == 3:
                specs, labels, _ = batch
            else:
                specs, labels = batch
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / len(train_loader.dataset)
        
        # Валидация
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    specs, labels, _ = batch
                else:
                    specs, labels = batch
                specs, labels = specs.to(device), labels.to(device)
                outputs = model(specs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / len(val_loader.dataset)
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"  Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}")
        print(f"  Val Loss={avg_val_loss:.4f},   Val Acc={val_acc:.4f}")

        # Early stopping и сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            torch.save(model.state_dict(), folder / "model_weights.pth")
            print(f"    Лучшая модель сохранена (Val Loss={avg_val_loss:.4f})")
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                print(f"    Early stopping после {epoch+1} эпох")
                break
    
    # Загрузка лучшей модели и тестирование
    model.load_state_dict(torch.load(folder / "model_weights.pth"))
    model.eval()
    preds, trues, probs, pids = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                specs, labels, batch_pids = batch
            else:
                specs, labels = batch
                batch_pids = [None] * specs.size(0)
            specs = specs.to(device)
            outputs = model(specs)
            probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            preds.extend(outputs.argmax(1).cpu().numpy())
            trues.extend(labels.numpy())
            pids.extend(batch_pids)
    
    # Метрики на уровне файлов
    metrics_file_level = {
        'accuracy': accuracy_score(trues, preds),
        'precision': precision_score(trues, preds, average='weighted', zero_division=0),
        'recall': recall_score(trues, preds, average='weighted', zero_division=0),
        'f1_score': f1_score(trues, preds, average='weighted', zero_division=0),
        'roc_auc': roc_auc_score(trues, probs) if len(set(trues)) > 1 else 0.5
    }

    # Метрики на уровне пациентов
    metrics_patient_level = None
    if config.AGGREGATE_BY_PATIENT and any(pid is not None for pid in pids):
        df = pd.DataFrame({'patient_id': pids, 'true': trues, 'pred': preds, 'prob': probs})
        agg = df.groupby('patient_id').agg({'prob':'mean', 'pred':lambda x: int(np.round(x.mean())), 'true':lambda x: int(round(x.mean()))}).reset_index()
        patient_trues = agg['true'].tolist()
        patient_preds = (agg['prob'] >= 0.5).astype(int).tolist()
        patient_probs = agg['prob'].tolist()
        metrics_patient_level = {
            'accuracy': accuracy_score(patient_trues, patient_preds),
            'precision': precision_score(patient_trues, patient_preds, average='weighted', zero_division=0),
            'recall': recall_score(patient_trues, patient_preds, average='weighted', zero_division=0),
            'f1_score': f1_score(patient_trues, patient_preds, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(patient_trues, patient_probs) if len(set(patient_trues)) > 1 else 0.5
        }
    
    # Основные метрики для отображения
    metrics = metrics_patient_level if metrics_patient_level is not None else metrics_file_level

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blueviolet')
    plt.plot(history['val_loss'], label='Val Loss', color='hotpink')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', color='blueviolet')
    plt.plot(history['val_acc'], label='Val Acc', color='hotpink')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(folder / "training_curves.png", dpi=150)
    plt.close()
    

    cm = confusion_matrix(trues, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='winter', 
                xticklabels=['Healthy', 'Parkinson'],
                yticklabels=['Healthy', 'Parkinson'])
    plt.title(f'Confusion Matrix\nAcc={metrics["accuracy"]:.3f}')
    plt.tight_layout()
    plt.savefig(folder / "confusion_matrix.png", dpi=150)
    plt.close()
    

    if len(set(trues)) > 1:
        fpr, tpr, _ = roc_curve(trues, probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, 'c-', label=f'AUC={metrics["roc_auc"]:.3f}')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(folder / "roc_curve.png", dpi=150)
        plt.close()

    
    try:
        from gradcam import visualize_gradcam
        from dataset import ParkinsonDataset as GradCamDataset

        if test_data and test_data.get('test_paths'):
            test_dataset = GradCamDataset(test_data['test_paths'], test_data['test_labels'])
            indices = [0, len(test_dataset)//2, len(test_dataset)-1]
            for idx in indices:
                if 0 <= idx < len(test_dataset):
                    item = test_dataset[idx]
                    if len(item) == 3:
                        spec, label, _ = item
                    else:
                        spec, label = item
                    save_path = folder / f"gradcam_sample_{idx}.png"
                    visualize_gradcam(model, model_name, spec.squeeze().numpy(), label.item(), save_path)
                    print(f"Grad-CAM сохранён: gradcam_sample_{idx}.png")
        else:
            print("Grad-CAM: нет тестовых данных для визуализации")
    except Exception as e:
        print(f"Grad-CAM не сработал: {e}")
    
    pd.DataFrame([metrics_file_level]).to_csv(folder / "metrics_file_level.csv", index=False)
    if metrics_patient_level is not None:
        pd.DataFrame([metrics_patient_level]).to_csv(folder / "metrics_patient_level.csv", index=False)
    pd.DataFrame(history).to_csv(folder / "training_history.csv", index=False)
    
    print(f"{scenario_name}: Acc={metrics['accuracy']:.4f}, AUC={metrics['roc_auc']:.4f}")
    
    return metrics