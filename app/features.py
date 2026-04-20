"""Acoustic feature extraction for observability (prosodic + spectral)."""

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)


def extract_acoustic_features(y: np.ndarray, sr: int) -> dict:
    """
    Prosodic + spectral features per segment. Useful as observability
    metadata — not fed into the classifier, which operates on raw audio.
    """
    # ── Prosodic ─────────────────────────────────────────────────────────────
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    valid_pitches = pitches[pitches > 0]
    pitch_mean = float(np.nanmean(valid_pitches)) if valid_pitches.size else 0.0
    pitch_var = float(np.nanvar(valid_pitches)) if valid_pitches.size else 0.0
    rms_energy = float(librosa.feature.rms(y=y)[0].mean())

    # ── Spectral ─────────────────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).mean(axis=1)
    spec_centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    spec_bandwidth = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
    spec_contrast = float(librosa.feature.spectral_contrast(y=y, sr=sr).mean())

    # Zero-crossing rate as a cheap speaking-rate proxy.
    zcr = float(librosa.feature.zero_crossing_rate(y=y).mean())

    return {
        "pitch_mean": round(pitch_mean, 3),
        "pitch_variance": round(pitch_var, 3),
        "rms_energy": round(rms_energy, 4),
        "mfcc_mean": [round(v, 3) for v in mfcc.tolist()],
        "spectral_centroid": round(spec_centroid, 3),
        "spectral_bandwidth": round(spec_bandwidth, 3),
        "spectral_contrast": round(spec_contrast, 3),
        "zero_crossing_rate": round(zcr, 4),
    }
