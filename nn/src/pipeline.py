"""
pipeline.py
===========
Главный скрипт для запуска всех экспериментов.

Координирует загрузку данных, обучение трёх моделей (CNN, ResNet18, DenseNet)
для всех 10 сценариев, агрегирует результаты и строит графики сравнения.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import config
from dataset import load_datasets, get_data_for_scenario
from models import CNNBaseline, TransferLearningModel, DenseNetModel
from train import train_and_evaluate

def set_seed(seed):
    """
    Фиксирует seed для воспроизводимости результатов.
    
    Параметры:
        seed (int): Seed для random, numpy, torch и CUDA.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def main():
    """
    Основная функция запуска экспериментов.
    
    Этапы:
        1. Создание папки для результатов
        2. Загрузка всех датасетов
        3. Обучение CNN Baseline для всех 10 сценариев
        4. Обучение Transfer Learning (ResNet18) для всех 10 сценариев
        5. Обучение DenseNet для всех 10 сценариев
        6. Агрегация результатов в сводные таблицы
        7. Построение графиков сравнения моделей
        8. Создание ML-подобной сводки для совместимости
        9. Построение графиков сравнения loss
    """
    Path("data/summaries").mkdir(parents=True, exist_ok=True)
    
    set_seed(config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}\n")
    
    print("="*30)
    print("ЗАГРУЗКА ДАННЫХ")
    print("="*30)
    
    datasets = load_datasets(config.DATA_PATH, config.SEED)
    
    print("\n" + "="*30)
    print("ПРОВЕРКА ЗАГРУЖЕННЫХ ДАННЫХ")
    print("="*30)
    for name, data in datasets.items():
        print(f"{name}: Train={len(data['train_paths'])}, Val={len(data['val_paths'])}, Test={len(data['test_paths'])}")
    
    print("\n" + "="*30)
    print("ЗАПУСК CNN BASELINE")
    print("="*30)
    
    cnn_results = {}
    for scenario in config.SCENARIOS:
        train_data, test_data = get_data_for_scenario(scenario, datasets)
        if train_data and len(train_data['train_paths']) > 0:
            print(f"\n---| {scenario} |---")
            metrics = train_and_evaluate(CNNBaseline(), "cnn_baseline", train_data, test_data, scenario, device, config)
            cnn_results[scenario] = {'metrics': metrics, 'n_train': len(train_data['train_paths']), 'n_val': len(train_data['val_paths']), 'n_test': len(test_data['test_paths'])}
    
    print("\n" + "="*30)
    print("ЗАПУСК TRANSFER LEARNING")
    print("="*30)
    
    tl_results = {}
    for scenario in config.SCENARIOS:
        train_data, test_data = get_data_for_scenario(scenario, datasets)
        if train_data and len(train_data['train_paths']) > 0:
            print(f"\n---| {scenario} |---")
            metrics = train_and_evaluate(TransferLearningModel(), "transfer_learning", train_data, test_data, scenario, device, config)
            tl_results[scenario] = {'metrics': metrics, 'n_train': len(train_data['train_paths']), 'n_val': len(train_data['val_paths']), 'n_test': len(test_data['test_paths'])}
    
    print("\n" + "="*30)
    print("ЗАПУСК DENSENET")
    print("="*30)
    dn_results = {}
    for scenario in config.SCENARIOS:
        train_data, test_data = get_data_for_scenario(scenario, datasets)
        if train_data and len(train_data['train_paths']) > 0:
            print(f"\n---| {scenario} |---")
            metrics = train_and_evaluate(DenseNetModel(), "densenet", train_data, test_data, scenario, device, config)
            dn_results[scenario] = {'metrics': metrics, 'n_train': len(train_data['train_paths']), 'n_val': len(train_data['val_paths']), 'n_test': len(test_data['test_paths'])}
    
    print("\n" + "="*30)
    print("РЕЗУЛЬТАТЫ")
    print("="*30)
    
    if cnn_results:
        cnn_rows = {}
        for s, v in cnn_results.items():
            m = v['metrics']
            cnn_rows[s] = m
        cnn_df = pd.DataFrame(cnn_rows).T[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].round(4)
        cnn_df.to_csv("data/summaries/cnn_baseline_summary.csv")
        print("\nCNN BASELINE:")
        print(cnn_df)
    
    if tl_results:
        tl_rows = {}
        for s, v in tl_results.items():
            m = v['metrics']
            tl_rows[s] = m
        tl_df = pd.DataFrame(tl_rows).T[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].round(4)
        tl_df.to_csv("data/summaries/transfer_learning_summary.csv")
        print("\nTRANSFER LEARNING:")
        print(tl_df)

    if dn_results:
        dn_rows = {}
        for s, v in dn_results.items():
            dn_rows[s] = v['metrics']
        dn_df = pd.DataFrame(dn_rows).T[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].round(4)
        dn_df.to_csv("data/summaries/densenet_summary.csv")
        print("\nDENSENET:")
        print(dn_df)
    
    if cnn_results and tl_results and dn_results:
        comparison = []
        for s in config.SCENARIOS:
            comparison.append({
                'Scenario': s,
                'CNN': cnn_results.get(s, {}).get('metrics', {}).get('accuracy', 0),
                'TransferLearning': tl_results.get(s, {}).get('metrics', {}).get('accuracy', 0),
                'DenseNet': dn_results.get(s, {}).get('metrics', {}).get('accuracy', 0)
            })
        comp_df = pd.DataFrame(comparison)
        comp_df.to_csv("data/summaries/models_comparison_all.csv", index=False)

        plt.figure(figsize=(14, 6))
        x = np.arange(len(comp_df))
        width = 0.25
        plt.bar(x - width, comp_df['CNN'], width, label='CNN Baseline', color='violet')
        plt.bar(x, comp_df['TransferLearning'], width, label='Transfer Learning', color='darkviolet')
        plt.bar(x + width, comp_df['DenseNet'], width, label='DenseNet', color='purple')
        plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, label='Random (0.5)')
        plt.xticks(x, comp_df['Scenario'], rotation=45, ha='right')
        plt.ylabel('Accuracy')
        plt.title('Model Comparison (All)')
        plt.legend()
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig("data/summaries/models_comparison_all.png", dpi=150)
        plt.show()

    # Сводка результатов
    overview_rows = []
    def make_row(scenario, model_name, data_dict):
        m = data_dict['metrics']
        return {
            'scenario': scenario,
            'selected_model': model_name,
            'final_fit_splits': 'train,val',
            'n_train': data_dict['n_train'],
            'n_val': data_dict['n_val'],
            'n_test': data_dict['n_test'],
            'patient_test_accuracy': m.get('accuracy', 0),
            'patient_test_balanced_accuracy': m.get('accuracy', 0),
            'patient_test_precision': m.get('precision', 0),
            'patient_test_recall': m.get('recall', 0),
            'patient_test_f1': m.get('f1_score', 0),
            'patient_test_roc_auc': m.get('roc_auc', 0),
            'file_test_accuracy': m.get('accuracy', 0),
            'file_test_balanced_accuracy': m.get('accuracy', 0),
            'file_test_precision': m.get('precision', 0),
            'file_test_recall': m.get('recall', 0),
            'file_test_f1': m.get('f1_score', 0),
            'file_test_roc_auc': m.get('roc_auc', 0)
        }

    for s, v in cnn_results.items():
        overview_rows.append(make_row(s, 'cnn_baseline', v))
    for s, v in tl_results.items():
        overview_rows.append(make_row(s, 'transfer_learning', v))
    for s, v in dn_results.items():
        overview_rows.append(make_row(s, 'densenet', v))

    if overview_rows:
        overview_df = pd.DataFrame(overview_rows)
        overview_df.to_csv("data/summaries/nn_scenarios_overview.csv", index=False)

    # Графики сравнения loss для всех сценариев
    def plot_loss_compare(scenario):
        plt.figure(figsize=(10, 6))
        for model_name in ['cnn_baseline', 'transfer_learning', 'densenet']:
            hist_file = Path(f"data/{model_name}/scenarios/{scenario}/training_history.csv")
            if hist_file.exists():
                try:
                    df = pd.read_csv(hist_file)
                    plt.plot(df['val_loss'], label=f"{model_name} val_loss")
                except Exception:
                    continue
        plt.xlabel('Epoch')
        plt.ylabel('Val Loss')
        plt.title(f'Validation Loss Comparison - {scenario}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        out = Path('data/summaries') / f'loss_comparison_{scenario}.png'
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()

    for s in config.SCENARIOS:
        plot_loss_compare(s)

    print(f"\nРезультаты сохранены в: parkinson_project/nn/data/")

if __name__ == "__main__":
    main()