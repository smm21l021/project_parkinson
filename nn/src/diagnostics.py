"""
diagnostics.py
==============
Быстрая проверка данных на утечки и корректность сплитов.

Загружает датасеты через load_datasets, выводит:
    - Количество файлов и пациентов в каждом сплите
    - Баланс классов
    - Пересечения пациентов между сплитами
    - Дубликаты файлов
"""

from pathlib import Path
from dataset import load_datasets
from config import config


def main():
    """
    Запускает диагностику данных.
    
    Выводит для каждого датасета:
        - Количество файлов и пациентов в train/val/test
        - Распределение классов
        - Дубликаты файлов между сплитами
    """
    dp = config.DATA_PATH
    print(f"Используемый путь к данным: {dp}")
    datasets = load_datasets(dp, config.SEED)

    for name, d in datasets.items():
        print(f"\nДатасет: {name}")
        for split in ['train', 'val', 'test']:
            pths = d.get(f'{split}_paths', [])
            labs = d.get(f'{split}_labels', [])
            pids = d.get(f'{split}_pids', [])
            print(f"  {split}: files={len(pths)}, patients={len(set(pids)) if pids else 'N/A'}, labels={ {0: labs.count(0), 1: labs.count(1)} }")

        # Проверка дубликатов файлов между сплитами
        all_files = {}
        for split in ['train','val','test']:
            for p in d.get(f'{split}_paths', []):
                all_files.setdefault(Path(p).name, set()).add(split)
        dup_files = {fn: splits for fn, splits in all_files.items() if len(splits) > 1}
        print(f"  Дубликаты файлов между сплитами: {len(dup_files)}")
        if len(dup_files) > 0:
            for fn, splits in list(dup_files.items())[:10]:
                print(f"    {fn}: in {splits}")

if __name__ == '__main__':
    main()
