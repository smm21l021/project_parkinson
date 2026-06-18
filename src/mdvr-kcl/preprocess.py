"""
preprocess.py
=============
Функции предобработки аудио для задачи распознавания болезни Паркинсона.

Что делает модуль:
  - Загрузка аудио с конвертацией в mono + ресемплинг
  - Удаление тишины (librosa.effects.trim)
  - Нормализация громкости (peak normalization → [-1, 1])
  - Проверка файлов на повреждённость
  - Пакетная обработка всего датасета
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Константы ────────────────────────────────────────────────────────────────
TARGET_SR = 16_000       # Гц — стандарт для речевых задач
SILENCE_TOP_DB = 30      # порог для обрезки тишины (дБ ниже пика)
MIN_DURATION_SEC = 0.5   # минимальная длина записи после trim (сек)
# ──────────────────────────────────────────────────────────────────────────────


def check_audio_file(path: str | Path) -> Tuple[bool, str]:
    """
    Проверяет, можно ли открыть аудиофайл.

    Returns
    -------
    (is_valid, reason) — True/False и причина при ошибке.
    """
    path = Path(path)
    if not path.exists():
        return False, "file not found"
    try:
        info = sf.info(str(path))
        if info.frames == 0:
            return False, "zero frames"
        return True, "ok"
    except Exception as e:
        return False, str(e)


def load_audio(
    path: str | Path,
    sr: int = TARGET_SR,
    mono: bool = True,
) -> Tuple[Optional[np.ndarray], int]:
    """
    Загружает аудио: конвертирует в mono, ресемплирует до `sr`.

    Returns
    -------
    (y, sr) или (None, sr) при ошибке.
    """
    try:
        y, orig_sr = librosa.load(str(path), sr=sr, mono=mono)
        return y, sr
    except Exception as e:
        logger.warning(f"Cannot load {path}: {e}")
        return None, sr


def trim_silence(
    y: np.ndarray,
    top_db: int = SILENCE_TOP_DB,
) -> np.ndarray:
    """
    Убирает незначимую тишину в начале и конце записи.
    Использует librosa.effects.trim (energy-based).
    """
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed


def normalize_volume(y: np.ndarray) -> np.ndarray:
    """
    Peak normalization: масштабирует сигнал так, чтобы |max| = 1.0.
    Если сигнал пустой / нулевой — возвращает как есть.
    """
    peak = np.max(np.abs(y))
    if peak < 1e-8:
        logger.warning("Signal is near-silent, skipping normalization.")
        return y
    return y / peak


def preprocess_audio(
    input_path: str | Path,
    output_path: str | Path,
    sr: int = TARGET_SR,
    trim: bool = True,
    normalize: bool = True,
) -> bool:
    """
    Полный цикл предобработки одного файла:
      1. Проверка
      2. Загрузка (mono, ресемплинг)
      3. Trim тишины
      4. Нормализация громкости
      5. Сохранение в WAV (16-bit PCM)

    Returns
    -------
    True если успешно, False при ошибке.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # 1. Проверка файла
    valid, reason = check_audio_file(input_path)
    if not valid:
        logger.error(f"[SKIP] {input_path.name}: {reason}")
        return False

    # 2. Загрузка
    y, sr_out = load_audio(input_path, sr=sr)
    if y is None:
        return False

    # 3. Trim тишины
    if trim:
        y = trim_silence(y)

    # 4. Проверка минимальной длины после trim
    if len(y) / sr_out < MIN_DURATION_SEC:
        logger.warning(
            f"[SKIP] {input_path.name}: too short after trim "
            f"({len(y)/sr_out:.2f}s < {MIN_DURATION_SEC}s)"
        )
        return False

    # 5. Нормализация
    if normalize:
        y = normalize_volume(y)

    # 6. Сохранение
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), y, sr_out, subtype="PCM_16")
    return True


def batch_preprocess(
    file_list: list[Tuple[Path, Path]],
    sr: int = TARGET_SR,
    trim: bool = True,
    normalize: bool = True,
) -> dict:
    """
    Пакетная обработка списка файлов.

    Parameters
    ----------
    file_list : list of (input_path, output_path)

    Returns
    -------
    {
        "ok": [...],      # успешно обработанные пути
        "failed": [...],  # пути с ошибками
    }
    """
    results = {"ok": [], "failed": []}

    for in_path, out_path in tqdm(file_list, desc="Preprocessing"):
        success = preprocess_audio(
            in_path, out_path, sr=sr, trim=trim, normalize=normalize
        )
        if success:
            results["ok"].append(str(in_path))
        else:
            results["failed"].append(str(in_path))

    logger.info(
        f"Done: {len(results['ok'])} OK, {len(results['failed'])} FAILED"
    )
    return results


def get_duration(path: str | Path, sr: int = TARGET_SR) -> float:
    """Возвращает длину аудиофайла в секундах. -1 при ошибке."""
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        return -1.0
