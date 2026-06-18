"""
ml_features.py
==============
Extracts acoustic features for the classical ML block.

Typical direct usage:

    python ml/src/ml_features.py
    python ml/src/ml_features.py --metadata data/ipvs/metadata.csv --processed_dir data/ipvs/processed
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ML_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ML_ROOT.parent
TARGET_SR = 16_000
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
PITCH_MIN_HZ = 75.0
PITCH_MAX_HZ = 500.0


def _flatten_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Returns standard summary statistics for a 1D numeric array."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_median": np.nan,
        }

    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_median": float(np.median(arr)),
    }


def _matrix_stats(matrix: np.ndarray, prefix: str) -> dict[str, float]:
    """Returns stats for each feature row in a 2D matrix."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]

    features: dict[str, float] = {}
    for idx, row in enumerate(arr, start=1):
        features.update(_flatten_stats(row, f"{prefix}_{idx:02d}"))
    return features


def _safe_spectral_contrast(stft_mag: np.ndarray, sr: int) -> np.ndarray:
    """Spectral contrast may fail on some edge cases for very short files."""
    try:
        return librosa.feature.spectral_contrast(S=stft_mag, sr=sr)
    except Exception:
        return np.full((7, 1), np.nan, dtype=float)


def _estimate_pitch_track(
    stft_mag: np.ndarray,
    sr: int,
    fmin: float = PITCH_MIN_HZ,
    fmax: float = PITCH_MAX_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds a simple dominant pitch track with piptrack."""
    pitches, magnitudes = librosa.piptrack(
        S=stft_mag,
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        hop_length=HOP_LENGTH,
    )
    best_idx = np.argmax(magnitudes, axis=0)
    frame_ids = np.arange(magnitudes.shape[1])
    pitch_track = pitches[best_idx, frame_ids]
    magnitude_track = magnitudes[best_idx, frame_ids]
    return pitch_track, magnitude_track


def _resolve_shared_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _resolve_ml_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return ML_ROOT / path


def _resolve_audio_path(row: pd.Series, processed_dir: Path | None = None) -> Path:
    """
    Resolves audio in a robust order:
    1. explicit ``audio_path`` from a normalized manifest
    2. ``processed_dir / filename``
    3. original ``processed_path``
    """
    explicit = Path(str(row.get("audio_path", "")))
    if str(row.get("audio_path", "")) and explicit.exists():
        return explicit

    if processed_dir is not None and "filename" in row.index:
        candidate = processed_dir / str(row["filename"])
        if candidate.exists():
            return candidate

    fallback = Path(str(row.get("processed_path", "")))
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Audio file not found for {row.get('filename', 'unknown')}")


def extract_acoustic_features(audio_path: str | Path, sr: int = TARGET_SR) -> dict[str, float]:
    """
    Extracts a compact acoustic feature set suitable for baseline ML models.

    ``jitter`` and ``shimmer`` below are lightweight approximations derived
    from pitch and frame-energy contours, not clinical Praat measurements.
    """
    audio_path = Path(audio_path)
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)

    if y.size == 0:
        raise ValueError(f"Empty audio signal: {audio_path}")

    stft_mag = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(S=stft_mag)
    centroid = librosa.feature.spectral_centroid(S=stft_mag, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=stft_mag, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(S=stft_mag, sr=sr)
    flatness = librosa.feature.spectral_flatness(S=stft_mag)
    contrast = _safe_spectral_contrast(stft_mag, sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(S=stft_mag, sr=sr, hop_length=HOP_LENGTH)

    pitch_track, pitch_strength = _estimate_pitch_track(stft_mag=stft_mag, sr=sr)
    voiced_mask = pitch_track > 0
    voiced_pitch = pitch_track[voiced_mask]
    voiced_rms = rms.reshape(-1)[voiced_mask[: rms.size]]

    harmonic = librosa.effects.harmonic(y)
    residual = y - harmonic
    harmonic_energy = float(np.sum(np.square(harmonic)))
    residual_energy = float(np.sum(np.square(residual)))
    hnr_db = 10.0 * np.log10((harmonic_energy + 1e-10) / (residual_energy + 1e-10))

    features: dict[str, float] = {
        "duration_sec": float(len(y) / sr),
        "signal_mean": float(np.mean(y)),
        "signal_std": float(np.std(y)),
        "signal_abs_mean": float(np.mean(np.abs(y))),
        "voiced_fraction": float(np.mean(voiced_mask)) if voiced_mask.size else np.nan,
        "pitch_strength_mean": float(np.mean(pitch_strength)) if pitch_strength.size else np.nan,
        "hnr_db": float(hnr_db),
    }

    features.update(_flatten_stats(zcr, "zcr"))
    features.update(_flatten_stats(rms, "rms"))
    features.update(_flatten_stats(centroid, "spectral_centroid"))
    features.update(_flatten_stats(bandwidth, "spectral_bandwidth"))
    features.update(_flatten_stats(rolloff, "spectral_rolloff"))
    features.update(_flatten_stats(flatness, "spectral_flatness"))
    features.update(_matrix_stats(contrast, "spectral_contrast"))
    features.update(_matrix_stats(mfcc, "mfcc"))
    features.update(_matrix_stats(mfcc_delta, "mfcc_delta"))
    features.update(_matrix_stats(mfcc_delta2, "mfcc_delta2"))
    features.update(_matrix_stats(chroma, "chroma"))

    if voiced_pitch.size > 0:
        features.update(_flatten_stats(voiced_pitch, "pitch_hz"))
        features["pitch_range_hz"] = float(np.max(voiced_pitch) - np.min(voiced_pitch))
        periods = 1.0 / np.clip(voiced_pitch, 1e-6, None)
        if periods.size > 1:
            jitter_abs = float(np.mean(np.abs(np.diff(periods))))
            features["jitter_abs"] = jitter_abs
            features["jitter_rel"] = float(jitter_abs / (np.mean(periods) + 1e-10))
        else:
            features["jitter_abs"] = np.nan
            features["jitter_rel"] = np.nan
    else:
        features.update(_flatten_stats(np.array([]), "pitch_hz"))
        features["pitch_range_hz"] = np.nan
        features["jitter_abs"] = np.nan
        features["jitter_rel"] = np.nan

    if voiced_rms.size > 1:
        shimmer_abs = float(np.mean(np.abs(np.diff(voiced_rms))))
        features["shimmer_abs"] = shimmer_abs
        features["shimmer_rel"] = float(shimmer_abs / (np.mean(voiced_rms) + 1e-10))
    else:
        features["shimmer_abs"] = np.nan
        features["shimmer_rel"] = np.nan

    return features


def build_feature_table_from_frame(
    metadata_df: pd.DataFrame,
    output_csv: str | Path,
    summary_json: str | Path | None = None,
    *,
    processed_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Extracts features for every row in a prepared metadata frame."""
    output_csv = Path(output_csv)
    processed_dir_path = Path(processed_dir) if processed_dir is not None else None

    missing = {"filename", "label", "patient_id", "split"} - set(metadata_df.columns)
    if missing:
        raise ValueError(f"Metadata frame missing required columns: {sorted(missing)}")

    records: list[dict[str, object]] = []
    failed_files: list[str] = []

    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Extracting ML features"):
        try:
            audio_path = _resolve_audio_path(row, processed_dir=processed_dir_path)
            row_features = extract_acoustic_features(audio_path)
            base_record = row.to_dict()
            base_record["audio_path"] = str(audio_path)
            base_record.update(row_features)
            records.append(base_record)
        except Exception as exc:
            failed_files.append(str(row.get("filename", "unknown")))
            logger.warning("Skipping %s: %s", row.get("filename", "unknown"), exc)

    feature_df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_csv, index=False)
    logger.info("Saved feature table: %s", output_csv)

    if summary_json is not None:
        summary_json = Path(summary_json)
        numeric_columns = [
            column
            for column in feature_df.columns
            if pd.api.types.is_numeric_dtype(feature_df[column]) and column != "label"
        ]
        summary = {
            "n_records": int(len(feature_df)),
            "n_failed": int(len(failed_files)),
            "failed_files": failed_files,
            "n_numeric_features": int(len(numeric_columns)),
            "feature_columns": numeric_columns,
            "split_distribution": feature_df["split"].value_counts().sort_index().to_dict(),
            "dataset_distribution": feature_df["dataset"].value_counts().sort_index().to_dict()
            if "dataset" in feature_df.columns
            else {},
        }
        with open(summary_json, "w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, ensure_ascii=False, indent=2)
        logger.info("Saved feature summary: %s", summary_json)

    return feature_df


def build_feature_table(
    metadata_path: str | Path,
    processed_dir: str | Path,
    output_csv: str | Path,
    summary_json: str | Path | None = None,
) -> pd.DataFrame:
    """Compatibility wrapper for extracting features from one dataset metadata CSV."""
    metadata_df = pd.read_csv(metadata_path)
    metadata_df = metadata_df.copy()
    metadata_df["audio_path"] = metadata_df["filename"].map(
        lambda name: str((Path(processed_dir) / str(name)).resolve())
    )
    return build_feature_table_from_frame(
        metadata_df=metadata_df,
        output_csv=output_csv,
        summary_json=summary_json,
        processed_dir=processed_dir,
    )


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract acoustic features for one dataset")
    parser.add_argument("--metadata", type=str, default="data/uams/metadata.csv")
    parser.add_argument("--processed_dir", type=str, default="data/uams/processed")
    parser.add_argument(
        "--out_csv",
        type=str,
        default="data/cache/dataset_features/uams/ml_feature_table.csv",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default="data/cache/dataset_features/uams/ml_feature_summary.json",
    )
    return parser.parse_args(args=args)


def main(args: Iterable[str] | None = None) -> None:
    parsed = parse_args(args=args)
    build_feature_table(
        metadata_path=_resolve_shared_path(parsed.metadata),
        processed_dir=_resolve_shared_path(parsed.processed_dir),
        output_csv=_resolve_ml_path(parsed.out_csv),
        summary_json=_resolve_ml_path(parsed.summary_json),
    )


if __name__ == "__main__":
    main()
