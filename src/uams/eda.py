"""
eda.py
======
Визуализация EDA для датасета Паркинсона.
Сохраняет графики в data/eda_plots/.

Запуск:
    python src/eda.py --metadata ../data/metadata.csv --out_dir ../data
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import librosa
import librosa.display

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

COLORS = {"parkinson": "#E05C5C", "healthy": "#5C9AE0"}


def plot_class_balance(df: pd.DataFrame, out_dir: Path) -> None:
    """
    График 1: Баланс классов
    - Слева: общий баланс классов
    - Справа: баланс по сплитам (train/val/test)
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # --- Общий баланс ---
    ax = axes[0]
    counts = df.groupby("label_name")["file_path"].count()
    colors = [COLORS.get(c, "#888") for c in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.45, edgecolor="white")
    ax.set_title("Баланс классов (всего)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Количество файлов")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                str(val), ha="center", fontsize=11)
    ax.set_ylim(0, counts.max() * 1.2)
    ax.spines[["top", "right"]].set_visible(False)

    # --- Баланс по сплитам ---
    ax = axes[1]
    if "split" in df.columns:
        split_counts = df.groupby(["split", "label_name"])["file_path"].count().unstack(fill_value=0)
        split_order = [s for s in ["train", "val", "test"] if s in split_counts.index]
        split_counts = split_counts.loc[split_order]
        x = np.arange(len(split_counts))
        w = 0.35
        for i, label_name in enumerate(["healthy", "parkinson"]):
            if label_name in split_counts.columns:
                ax.bar(x + i * w, split_counts[label_name],
                       width=w, label=label_name,
                       color=COLORS[label_name], edgecolor="white")
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels(split_counts.index)
        ax.set_title("Баланс классов по сплитам", fontsize=13, fontweight="bold")
        ax.set_ylabel("Количество файлов")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
    else:
        ax.text(0.5, 0.5, "splits not found", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    out_path = out_dir / "01_class_balance.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def plot_duration_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """
    График 2: Распределение длин записей
    - Слева: гистограмма распределения длин
    - Справа: boxplot по классам
    """
    if "duration_sec" not in df.columns:
        logger.warning("No 'duration_sec' column — skipping duration plot")
        return

    df_dur = df.dropna(subset=["duration_sec"])
    if len(df_dur) == 0:
        logger.warning("No valid duration data — skipping duration plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Гистограмма
    ax = axes[0]
    for label_name, color in COLORS.items():
        sub = df_dur[df_dur["label_name"] == label_name]["duration_sec"]
        if len(sub) > 0:
            ax.hist(sub, bins=30, alpha=0.65, color=color,
                    label=label_name, edgecolor="white")
    ax.set_title("Распределение длин записей", fontsize=13, fontweight="bold")
    ax.set_xlabel("Длина (сек)")
    ax.set_ylabel("Количество файлов")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    # Boxplot
    ax = axes[1]
    data_by_class = [
        df_dur[df_dur["label_name"] == lbl]["duration_sec"].values
        for lbl in ["healthy", "parkinson"]
    ]
    if len(data_by_class[0]) > 0 and len(data_by_class[1]) > 0:
        bp = ax.boxplot(data_by_class, labels=["healthy", "parkinson"],
                        patch_artist=True, widths=0.4,
                        medianprops={"color": "black", "linewidth": 2})
        for patch, color in zip(bp["boxes"], [COLORS["healthy"], COLORS["parkinson"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax.set_title("Длины записей (boxplot)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Длина (сек)")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = out_dir / "02_duration_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def plot_example_waveforms(
    df: pd.DataFrame,
    processed_dir: Path,
    out_dir: Path,
    n_examples: int = 2,
) -> None:
    """
    График 3: Примеры waveform + спектрограмма для обоих классов
    """
    n_rows = n_examples * 2
    fig = plt.figure(figsize=(14, 3 * n_rows))
    gs = gridspec.GridSpec(n_rows, 2, hspace=0.55, wspace=0.3)
    
    row = 0
    for label_name in ["healthy", "parkinson"]:
        subset = df[df["label_name"] == label_name]
        if len(subset) == 0:
            logger.warning(f"No examples found for {label_name}")
            continue
            
        for _, record in subset.head(n_examples).iterrows():
            fname = record["filename"]
            fpath = processed_dir / fname
            if not fpath.exists():
                fpath = Path(record["file_path"])
            if not fpath.exists():
                logger.warning(f"File not found: {fname}")
                continue
            
            try:
                y, sr = librosa.load(str(fpath), sr=None, mono=True)
            except Exception as e:
                logger.warning(f"Cannot load {fname}: {e}")
                continue
            
            # Waveform
            ax_wave = fig.add_subplot(gs[row, 0])
            t = np.linspace(0, len(y) / sr, len(y))
            ax_wave.plot(t, y, color=COLORS[label_name], linewidth=0.5)
            ax_wave.set_title(f"Waveform — {label_name} ({fname})",
                              fontsize=10, fontweight="bold")
            ax_wave.set_xlabel("Время (сек)")
            ax_wave.set_ylabel("Амплитуда")
            ax_wave.spines[["top", "right"]].set_visible(False)
            
            # Mel-спектрограмма
            ax_spec = fig.add_subplot(gs[row, 1])
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
            S_db = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_db, sr=sr, x_axis="time",
                                           y_axis="mel", ax=ax_spec)
            ax_spec.set_title(f"Mel-спектрограмма — {label_name}",
                              fontsize=10, fontweight="bold")
            plt.colorbar(img, ax=ax_spec, format="%+2.0f dB")
            
            row += 1
    
    out_path = out_dir / "03_example_waveforms.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def compute_durations(df: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    """Вычисляет длительность аудиофайлов и добавляет колонку duration_sec"""
    import soundfile as sf
    
    durations = []
    for _, row in df.iterrows():
        fpath = processed_dir / row["filename"]
        if not fpath.exists():
            fpath = Path(row["file_path"])
        
        try:
            info = sf.info(str(fpath))
            duration = info.frames / info.samplerate
            durations.append(duration)
        except Exception as e:
            logger.warning(f"Cannot read duration for {row['filename']}: {e}")
            durations.append(np.nan)
    
    df["duration_sec"] = durations
    return df


def run_eda(metadata_csv: str | Path, out_dir: str | Path) -> None:
    """Запускает полное EDA и сохраняет все графики."""
    metadata_csv = Path(metadata_csv)
    out_dir = Path(out_dir)
    plots_dir = out_dir / "eda_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = out_dir / "processed"
    
    df = pd.read_csv(metadata_csv)
    logger.info(f"Loaded {len(df)} records from {metadata_csv}")
    logger.info(f"  Healthy: {(df['label']==0).sum()}")
    logger.info(f"  Parkinson: {(df['label']==1).sum()}")
    
    if "duration_sec" not in df.columns:
        logger.info("Computing durations...")
        df = compute_durations(df, processed_dir)
        df.to_csv(metadata_csv, index=False)
        logger.info(f"Updated metadata saved to {metadata_csv}")
    
    # Строим графики
    plot_class_balance(df, plots_dir)
    plot_duration_distribution(df, plots_dir)
    
    if processed_dir.exists():
        plot_example_waveforms(df, processed_dir, plots_dir)
    else:
        logger.warning(f"Processed directory not found: {processed_dir}")
        logger.warning("Skipping waveform plots")
    
    logger.info(f"✅ EDA complete! Plots saved to: {plots_dir}")
    
    # Выводим краткую статистику
    print("\n" + "="*55)
    print("  КРАТКАЯ СТАТИСТИКА EDA")
    print("="*55)
    print(f"📁 Всего файлов: {len(df)}")
    print(f"👤 Уникальных пациентов: {df['patient_id'].nunique()}")
    print(f"\n📊 Баланс классов:")
    print(f"   Здоровые (label=0): {(df['label']==0).sum()} файлов")
    print(f"   Паркинсон (label=1): {(df['label']==1).sum()} файлов")
    
    if "duration_sec" in df.columns:
        valid_dur = df["duration_sec"].dropna()
        if len(valid_dur) > 0:
            print(f"\n⏱ Длительности записей (сек):")
            print(f"   min={valid_dur.min():.2f}")
            print(f"   max={valid_dur.max():.2f}")
            print(f"   mean={valid_dur.mean():.2f}")
            print(f"   median={valid_dur.median():.2f}")
            print(f"   std={valid_dur.std():.2f}")
    
    if "split" in df.columns:
        print(f"\n✂️ По сплитам:")
        for split in ["train", "val", "test"]:
            sub = df[df["split"] == split]
            if len(sub) > 0:
                n_pd = (sub["label"] == 1).sum()
                n_hc = (sub["label"] == 0).sum()
                print(f"   {split}: {len(sub)} файлов | PD={n_pd}, HC={n_hc}")
    
    print("="*55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA для датасета Паркинсона")
    parser.add_argument("--metadata", type=str, default="../data/metadata.csv",
                        help="Путь к metadata.csv")
    parser.add_argument("--out_dir", type=str, default="../data",
                        help="Папка для сохранения результатов")
    args = parser.parse_args()
    run_eda(args.metadata, args.out_dir)