"""
prepare_dataset.py
==================
Главный скрипт подготовки датасета. Запускать из корня проекта:

    python src/prepare_dataset.py --dataset sakar --raw_dir data/raw --out_dir data

Что делает:
  1. Сканирует сырые аудиофайлы
  2. Предобрабатывает каждый (trim, normalize, resample → 16kHz WAV mono)
  3. Создаёт data/metadata.csv
  4. Делает train/val/test split по пациентам (patient-level split)
  5. Сохраняет splits/train.csv, splits/val.csv, splits/test.csv
  6. Выводит короткое EDA
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from preprocess import batch_preprocess, get_duration, TARGET_SR
from data_loader import (
    parse_sakar_dataset,
    parse_generic_dataset,
    combine_datasets,
    load_metadata,
    LABEL_MAP,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─── Split по пациентам ───────────────────────────────────────────────────────

def patient_level_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Разделяет данные так, чтобы один пациент попадал ровно в один из сплитов.
    Это предотвращает data leakage между train и test.

    Алгоритм:
      1. Получаем список уникальных пациентов
      2. Стратифицированно (по label) делим на train+val / test
      3. Затем train+val → train / val
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    patients = df.groupby("patient_id")["label"].first().reset_index()
    patients.columns = ["patient_id", "label"]

    # Шаг 1: (train+val) vs test
    splitter1 = GroupShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=random_state
    )
    trainval_idx, test_idx = next(
        splitter1.split(patients, patients["label"], groups=patients["patient_id"])
    )
    trainval_patients = patients.iloc[trainval_idx]
    test_patients = patients.iloc[test_idx]["patient_id"].tolist()

    # Шаг 2: train vs val из trainval
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    splitter2 = GroupShuffleSplit(
        n_splits=1, test_size=val_size_adjusted, random_state=random_state
    )
    train_idx, val_idx = next(
        splitter2.split(
            trainval_patients,
            trainval_patients["label"],
            groups=trainval_patients["patient_id"],
        )
    )
    train_patients = trainval_patients.iloc[train_idx]["patient_id"].tolist()
    val_patients = trainval_patients.iloc[val_idx]["patient_id"].tolist()

    # Присваиваем метки сплитов
    split_map = {}
    for pid in train_patients:
        split_map[pid] = "train"
    for pid in val_patients:
        split_map[pid] = "val"
    for pid in test_patients:
        split_map[pid] = "test"

    df = df.copy()
    df["split"] = df["patient_id"].map(split_map)

    # Проверка
    assert df["split"].isna().sum() == 0, "Some patients have no split assignment"

    logger.info("Split summary (patients):")
    for split in ["train", "val", "test"]:
        n_patients = df[df["split"] == split]["patient_id"].nunique()
        n_files = (df["split"] == split).sum()
        n_pd = df[(df["split"] == split) & (df["label"] == 1)]["patient_id"].nunique()
        n_hc = df[(df["split"] == split) & (df["label"] == 0)]["patient_id"].nunique()
        logger.info(
            f"  {split:5s}: {n_patients:3d} patients "
            f"(PD={n_pd}, HC={n_hc}), {n_files} files"
        )

    return df


# ─── Cross-dataset сценарии ───────────────────────────────────────────────────

def prepare_cross_dataset_splits(
    combined_df: pd.DataFrame,
    train_dataset: str,
    test_dataset: str,
    out_dir: Path,
) -> None:
    """
    Сценарий cross-dataset:
      train = весь датасет A
      test  = весь датасет B
    Полезно для проверки обобщаемости модели.
    """
    train_df = combined_df[combined_df["dataset"] == train_dataset].copy()
    test_df = combined_df[combined_df["dataset"] == test_dataset].copy()

    cross_dir = out_dir / "splits_cross"
    cross_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(cross_dir / "train.csv", index=False)
    test_df.to_csv(cross_dir / "test.csv", index=False)

    logger.info(
        f"Cross-dataset split saved: "
        f"train={train_dataset} ({len(train_df)}), "
        f"test={test_dataset} ({len(test_df)})"
    )


# ─── Краткое EDA ─────────────────────────────────────────────────────────────

def print_eda(df: pd.DataFrame, processed_dir: Path) -> dict:
    """
    Выводит в консоль и возвращает словарь с EDA-статистиками:
      - Кол-во пациентов / файлов
      - Баланс классов
      - Статистика длин записей
    """
    print("\n" + "=" * 55)
    print("  EDA — КРАТКИЙ ОТЧЁТ")
    print("=" * 55)

    # Общее
    n_total = len(df)
    n_patients = df["patient_id"].nunique()
    print(f"\n📁 Всего файлов   : {n_total}")
    print(f"👤 Уникальных пациентов: {n_patients}")

    # Баланс классов
    print("\n📊 Баланс классов:")
    for label, name in LABEL_MAP.items():
        n = (df["label"] == label).sum()
        pct = 100 * n / n_total
        patients_n = df[df["label"] == label]["patient_id"].nunique()
        print(f"  {name:10s} (label={label}): {n:4d} файлов ({pct:.1f}%), "
              f"{patients_n} пациентов")

    # По датасетам
    if df["dataset"].nunique() > 1:
        print("\n🗂  По датасетам:")
        for ds, grp in df.groupby("dataset"):
            pd_n = (grp["label"] == 1).sum()
            hc_n = (grp["label"] == 0).sum()
            print(f"  {ds}: {len(grp)} файлов, PD={pd_n}, HC={hc_n}")

    # Длины записей
    print("\n⏱  Длины записей (сек):")
    durations = []
    for _, row in df.iterrows():
        fpath = processed_dir / row["filename"] \
            if (processed_dir / row["filename"]).exists() \
            else Path(row["file_path"])
        d = get_duration(fpath)
        if d > 0:
            durations.append(d)

    if durations:
        arr = np.array(durations)
        stats = {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
        }
        print(f"  min={stats['min']:.2f}  max={stats['max']:.2f}  "
              f"mean={stats['mean']:.2f}  median={stats['median']:.2f}  "
              f"std={stats['std']:.2f}")
    else:
        stats = {}
        print("  (processed files not found, skipping duration stats)")

    # По сплитам
    if "split" in df.columns:
        print("\n✂️  По сплитам:")
        for split in ["train", "val", "test"]:
            sub = df[df["split"] == split]
            pd_n = (sub["label"] == 1).sum()
            hc_n = (sub["label"] == 0).sum()
            print(f"  {split:5s}: {len(sub):4d} файлов | PD={pd_n}, HC={hc_n} | "
                  f"{sub['patient_id'].nunique()} пациентов")

    print("=" * 55 + "\n")

    return {
        "n_files": n_total,
        "n_patients": n_patients,
        "class_balance": df["label"].value_counts().to_dict(),
        "duration_stats": stats if durations else {},
    }


# ─── Главная функция ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parkinson's voice dataset preprocessing pipeline"
    )
    parser.add_argument(
        "--dataset",
        choices=["sakar", "generic", "combined"],
        default="sakar",
        help="Тип датасета",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/raw",
        help="Папка с исходными аудиофайлами",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data",
        help="Корневая папка для результатов",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=TARGET_SR,
        help=f"Целевая частота дискретизации (default={TARGET_SR})",
    )
    parser.add_argument(
        "--skip_preprocess",
        action="store_true",
        help="Пропустить предобработку (если уже сделана)",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.70
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.15
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    processed_dir = out_dir / "processed"
    splits_dir = out_dir / "splits"
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Парсинг датасета ──────────────────────────────────────────────────
    logger.info(f"Parsing dataset: {args.dataset}")

    if args.dataset == "sakar":
        df = parse_sakar_dataset(raw_dir, dataset_name="sakar")
    elif args.dataset == "generic":
        df = parse_generic_dataset(raw_dir, dataset_name="generic")
    elif args.dataset == "combined":
        # Пример: два датасета в raw/sakar/ и raw/generic/
        df_sakar = parse_sakar_dataset(raw_dir / "sakar", dataset_name="sakar")
        df_gen = parse_generic_dataset(raw_dir / "generic", dataset_name="generic")
        df = combine_datasets([df_sakar, df_gen])
        # Cross-dataset splits
        prepare_cross_dataset_splits(df, "sakar", "generic", out_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if len(df) == 0:
        logger.error("No files found! Check --raw_dir path.")
        return

    # ── 2. Предобработка аудио ───────────────────────────────────────────────
    if not args.skip_preprocess:
        logger.info("Starting audio preprocessing...")
        file_pairs = [
            (Path(row["file_path"]), processed_dir / row["filename"])
            for _, row in df.iterrows()
        ]
        results = batch_preprocess(file_pairs, sr=args.sr)

        # Убираем из метаданных битые файлы
        failed_set = set(results["failed"])
        df = df[~df["file_path"].isin(failed_set)].reset_index(drop=True)
        logger.info(
            f"After filtering failed files: {len(df)} records remain"
        )
    else:
        logger.info("Skipping preprocessing (--skip_preprocess set)")

    # Обновляем file_path на processed
    df["processed_path"] = df["filename"].apply(
        lambda fn: str(processed_dir / fn)
    )

    # ── 3. Split по пациентам ────────────────────────────────────────────────
    logger.info("Creating patient-level splits...")
    df = patient_level_split(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    # ── 4. Сохранение metadata.csv ───────────────────────────────────────────
    meta_path = out_dir / "metadata.csv"
    df.to_csv(meta_path, index=False)
    logger.info(f"Saved metadata: {meta_path}")

    # ── 5. Сохранение splits ─────────────────────────────────────────────────
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        split_path = splits_dir / f"{split}.csv"
        split_df.to_csv(split_path, index=False)
    logger.info(f"Saved splits to: {splits_dir}")

    # ── 6. EDA ───────────────────────────────────────────────────────────────
    eda_stats = print_eda(df, processed_dir)
    eda_path = out_dir / "eda_stats.json"
    with open(eda_path, "w", encoding="utf-8") as f:
        json.dump(eda_stats, f, ensure_ascii=False, indent=2)
    logger.info(f"EDA stats saved: {eda_path}")

    logger.info("✅ Pipeline complete!")
    logger.info(f"   metadata.csv  → {meta_path}")
    logger.info(f"   processed/    → {processed_dir}")
    logger.info(f"   splits/       → {splits_dir}")


if __name__ == "__main__":
    main()
