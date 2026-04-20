"""Emotion classification from audio segments using HuggingFace audio-classification pipeline."""

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
from transformers import pipeline

from segmenter import AudioSegment

logger = logging.getLogger(__name__)

# This is the model that works well in practice for hospital-call frustration
# detection. It returns 6 raw LABEL_X ids which we map to canonical emotions.
DEFAULT_MODEL_ID = "Khoa/w2v-speech-emotion-recognition"

# Raw label → canonical emotion (taken from the validated reference pipeline).
LABEL_MAP: dict[str, str] = {
    "LABEL_0": "sad",
    "LABEL_1": "angry",
    "LABEL_2": "disgust",
    "LABEL_3": "fear",
    "LABEL_4": "happy",
    "LABEL_5": "neutral",
}

# Emotions the downstream report consumers expect to see.
CANONICAL_EMOTIONS = ("angry", "frustrated", "neutral", "sad", "disgust", "happy", "fear")


@dataclass
class SegmentPrediction:
    segment_index: int
    start_s: float
    end_s: float
    dominant_emotion: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)


class EmotionClassifier:
    """
    Thin wrapper around HuggingFace's `pipeline("audio-classification")`.

    Created once and reused across requests (singleton via `get_classifier()`).
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: Optional[int] = None):
        self.model_id = model_id
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.device = device

        logger.info(
            "Loading emotion pipeline '%s' on %s",
            model_id,
            "cuda:0" if device == 0 else "cpu",
        )

        self.pipe = pipeline(
            "audio-classification",
            model=model_id,
            device=device,
            top_k=None,  # return scores for all labels
        )
        logger.info("Emotion pipeline ready.")

    def predict_segment(self, segment: AudioSegment) -> SegmentPrediction:
        """Run inference on a single AudioSegment and return canonical scores."""
        audio_input = {
            "array": segment.audio.astype(np.float32),
            "sampling_rate": segment.sample_rate,
        }
        preds = self.pipe(audio_input)

        # preds: list of {"label": "LABEL_X", "score": float}
        canonical_scores: dict[str, float] = {}
        for p in preds:
            emotion = LABEL_MAP.get(p["label"], p["label"].lower())
            canonical_scores[emotion] = canonical_scores.get(emotion, 0.0) + float(p["score"])

        # Ensure every canonical key is present (default 0.0)
        for emotion in CANONICAL_EMOTIONS:
            canonical_scores.setdefault(emotion, 0.0)

        dominant = max(canonical_scores, key=canonical_scores.get)
        confidence = canonical_scores[dominant]

        return SegmentPrediction(
            segment_index=segment.index,
            start_s=segment.start_s,
            end_s=segment.end_s,
            dominant_emotion=dominant,
            confidence=round(confidence, 4),
            scores={k: round(v, 4) for k, v in canonical_scores.items()},
        )

    def predict_all(self, segments: list[AudioSegment]) -> list[SegmentPrediction]:
        predictions: list[SegmentPrediction] = []
        for seg in segments:
            pred = self.predict_segment(seg)
            logger.debug(
                "Segment %d [%.2f-%.2f s]: %s (%.3f)",
                seg.index,
                seg.start_s,
                seg.end_s,
                pred.dominant_emotion,
                pred.confidence,
            )
            predictions.append(pred)
        return predictions


@lru_cache(maxsize=1)
def get_classifier(model_id: str = DEFAULT_MODEL_ID) -> EmotionClassifier:
    """Return a cached singleton classifier instance."""
    return EmotionClassifier(model_id=model_id)
