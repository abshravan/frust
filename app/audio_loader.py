"""Audio loading, normalization, and preprocessing pipeline."""

import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000
_SILENCE_THRESHOLD_DB = -60.0


def load_audio(file_path: str | Path) -> tuple[np.ndarray, int]:
    """
    Load audio from WAV or MP3, convert to mono at 16kHz, and normalize.

    Returns:
        (audio_array, sample_rate) where audio_array is float32 in [-1, 1].

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is silent or too short (< 1 s).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    logger.info("Loading audio from %s", file_path)

    audio, sr = librosa.load(str(file_path), sr=TARGET_SAMPLE_RATE, mono=True)

    _validate_audio(audio, sr, file_path)

    audio = _normalize(audio)

    duration = len(audio) / sr
    logger.info(
        "Loaded %.2f s of audio | sr=%d | samples=%d", duration, sr, len(audio)
    )

    return audio, sr


def _validate_audio(audio: np.ndarray, sr: int, path: Path) -> None:
    min_samples = sr  # 1 second minimum
    if len(audio) < min_samples:
        raise ValueError(
            f"Audio too short ({len(audio) / sr:.2f} s). Minimum is 1 s. File: {path}"
        )

    rms_db = _rms_db(audio)
    if rms_db < _SILENCE_THRESHOLD_DB:
        raise ValueError(
            f"Audio appears to be silent (RMS {rms_db:.1f} dB). File: {path}"
        )


def _normalize(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize to [-1, 1]; guard against all-zero arrays."""
    peak = np.max(np.abs(audio))
    if peak < 1e-9:
        logger.warning("Near-silent audio detected; skipping normalization.")
        return audio
    return (audio / peak).astype(np.float32)


def _rms_db(audio: np.ndarray) -> float:
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-12:
        return -120.0
    return float(20 * np.log10(rms))


def audio_info(audio: np.ndarray, sr: int) -> dict:
    """Return a metadata dict useful for logging and debugging."""
    duration = len(audio) / sr
    return {
        "duration_s": round(duration, 3),
        "sample_rate": sr,
        "samples": len(audio),
        "rms_db": round(_rms_db(audio), 2),
        "peak": round(float(np.max(np.abs(audio))), 4),
    }
