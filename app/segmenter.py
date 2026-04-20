"""Sliding-window audio segmentation."""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

WINDOW_SIZE_S = 3.0
OVERLAP_S = 1.5  # 50% overlap — matches the validated reference pipeline


@dataclass(frozen=True)
class AudioSegment:
    index: int
    start_s: float
    end_s: float
    audio: np.ndarray
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def segment_audio(
    audio: np.ndarray,
    sr: int,
    window_s: float = WINDOW_SIZE_S,
    overlap_s: float = OVERLAP_S,
) -> list[AudioSegment]:
    """
    Slice audio into overlapping fixed-length windows.

    If the final chunk is shorter than window_s it is zero-padded so every
    segment fed to the model has the same length.
    """
    if overlap_s >= window_s:
        raise ValueError("overlap_s must be less than window_s")

    window_samples = int(window_s * sr)
    step_samples = int((window_s - overlap_s) * sr)
    total_samples = len(audio)

    segments: list[AudioSegment] = []
    idx = 0
    offset = 0

    while offset < total_samples:
        chunk = audio[offset : offset + window_samples]

        # Pad the last chunk if it is shorter than the window
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)), mode="constant")

        start_s = offset / sr
        end_s = min((offset + window_samples) / sr, total_samples / sr)

        segments.append(
            AudioSegment(
                index=idx,
                start_s=round(start_s, 3),
                end_s=round(end_s, 3),
                audio=chunk,
                sample_rate=sr,
            )
        )

        idx += 1
        offset += step_samples

    logger.info(
        "Segmented %d windows | window=%.1fs overlap=%.1fs step=%.1fs",
        len(segments),
        window_s,
        overlap_s,
        window_s - overlap_s,
    )
    return segments
