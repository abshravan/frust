"""Emotion classification from audio segments using HuggingFace audio-classification pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from transformers import pipeline

from segmenter import AudioSegment

logger = logging.getLogger(__name__)


# ── Model registry ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelConfig:
    """Describes one audio-emotion model and its raw-label → canonical mapping."""
    model_id: str
    short_name: str
    # Maps whatever the model returns (e.g. "LABEL_1", "ang") → canonical emotion.
    # Unrecognised labels are lower-cased and passed through unchanged.
    label_map: dict[str, str]


# Common aliases applied after per-model mapping as a fallback.
_COMMON_ALIASES: dict[str, str] = {
    "ang": "angry", "anger": "angry",
    "hap": "happy", "happiness": "happy", "excited": "happy",
    "neu": "neutral", "calm": "neutral", "boredom": "neutral",
    "sad": "sad", "sadness": "sad",
    "dis": "disgust",
    "fea": "fear", "fearful": "fear",
    "sur": "surprise", "surprised": "surprise",
    "fru": "frustrated", "frustrated": "frustrated",
}

# The three independent models registered for batch analysis.
MODELS_REGISTRY: list[ModelConfig] = [
    # ── Model 1 (primary, validated reference) ────────────────────────────────
    ModelConfig(
        model_id="Khoa/w2v-speech-emotion-recognition",
        short_name="khoa",
        label_map={
            "LABEL_0": "sad",
            "LABEL_1": "angry",
            "LABEL_2": "disgust",
            "LABEL_3": "fear",
            "LABEL_4": "happy",
            "LABEL_5": "neutral",
        },
    ),
    # ── Model 2: wav2vec2-LARGE fine-tuned on IEMOCAP via SUPERB benchmark ──────
    # Same 4-class taxonomy as the base SUPERB model but backed by the large
    # wav2vec2 encoder (317M params vs 95M).
    # Model card: https://huggingface.co/superb/wav2vec2-large-superb-er
    ModelConfig(
        model_id="superb/wav2vec2-large-superb-er",
        short_name="superb_large",
        label_map={
            "LABEL_0": "neutral",
            "LABEL_1": "happy",
            "LABEL_2": "angry",
            "LABEL_3": "sad",
        },
    ),
    # ── Model 3: XLSR-53-large fine-tuned on RAVDESS + SAVEE + TESS ──────────
    # 7-class: angry, disgust, fear, happy, neutral, sad, surprise
    # Based on facebook/wav2vec2-large-xlsr-53 (300M params cross-lingual).
    # Model card: https://huggingface.co/harshit345/xlsr-wav2vec-speech-emotion-recognition
    ModelConfig(
        model_id="harshit345/xlsr-wav2vec-speech-emotion-recognition",
        short_name="xlsr_large",
        label_map={
            # Model returns LABEL_X in RAVDESS alphabetical order.
            "LABEL_0": "angry",
            "LABEL_1": "disgust",
            "LABEL_2": "fear",
            "LABEL_3": "happy",
            "LABEL_4": "neutral",
            "LABEL_5": "sad",
            "LABEL_6": "surprise",
            # Passthrough for models that return string labels directly.
            "angry": "angry", "disgust": "disgust", "fear": "fear",
            "happy": "happy", "neutral": "neutral", "sad": "sad",
            "surprise": "surprise", "surprised": "surprise",
        },
    ),
]

DEFAULT_MODEL_ID = MODELS_REGISTRY[0].model_id

CANONICAL_EMOTIONS = ("angry", "frustrated", "neutral", "sad", "disgust", "happy", "fear")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SegmentPrediction:
    segment_index: int
    start_s: float
    end_s: float
    dominant_emotion: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)


# ── Classifier ────────────────────────────────────────────────────────────────

class EmotionClassifier:
    """
    Thin wrapper around HuggingFace `pipeline("audio-classification")`.
    One instance per model; use `get_classifier()` for the singleton cache.
    """

    def __init__(self, config: ModelConfig, device: Optional[int] = None):
        self.config = config
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.device = device

        logger.info(
            "Loading emotion pipeline [%s] '%s' on %s",
            config.short_name,
            config.model_id,
            "cuda:0" if device == 0 else "cpu",
        )
        self._pipe = pipeline(
            "audio-classification",
            model=config.model_id,
            device=device,
            top_k=None,
        )
        logger.info("[%s] ready.", config.short_name)

    def predict_segment(self, segment: AudioSegment) -> SegmentPrediction:
        audio_input = {
            "array": segment.audio.astype(np.float32),
            "sampling_rate": segment.sample_rate,
        }
        preds = self._pipe(audio_input)

        canonical_scores: dict[str, float] = {}
        for p in preds:
            raw = p["label"]
            # 1. per-model label map
            canon = self.config.label_map.get(raw)
            # 2. common aliases
            if canon is None:
                canon = _COMMON_ALIASES.get(raw.lower(), raw.lower())
            canonical_scores[canon] = canonical_scores.get(canon, 0.0) + float(p["score"])

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
                "[%s] seg %d [%.2f-%.2f s]: %s (%.3f)",
                self.config.short_name,
                seg.index,
                seg.start_s,
                seg.end_s,
                pred.dominant_emotion,
                pred.confidence,
            )
            predictions.append(pred)
        return predictions


# ── Singleton cache (one instance per model_id) ───────────────────────────────

_classifier_cache: dict[str, EmotionClassifier] = {}


def get_classifier(model_id: str = DEFAULT_MODEL_ID) -> EmotionClassifier:
    """Return a cached EmotionClassifier for the given model_id."""
    if model_id not in _classifier_cache:
        config = next(
            (m for m in MODELS_REGISTRY if m.model_id == model_id),
            ModelConfig(
                model_id=model_id,
                short_name=model_id.split("/")[-1],
                label_map={},
            ),
        )
        _classifier_cache[model_id] = EmotionClassifier(config)
    return _classifier_cache[model_id]


def preload_all_models() -> None:
    """Pre-load every registered model into the cache (call at startup)."""
    for cfg in MODELS_REGISTRY:
        get_classifier(cfg.model_id)
