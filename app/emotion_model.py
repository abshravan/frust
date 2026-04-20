"""Emotion classification from audio segments using HuggingFace wav2vec2."""

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from segmenter import AudioSegment

logger = logging.getLogger(__name__)

# Primary model: wav2vec2-based emotion classifier trained on speech emotion data.
# Fallback lets operators swap the model via env var or config.
DEFAULT_MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

# Canonical emotion labels the rest of the pipeline expects.
# Models may use different label names; LABEL_ALIASES maps them to canonical ones.
CANONICAL_EMOTIONS = {"angry", "frustrated", "neutral", "sad", "disgust", "happy", "fear", "surprise"}

LABEL_ALIASES: dict[str, str] = {
    # common wav2vec2 label variants → canonical
    "ang": "angry",
    "anger": "angry",
    "fru": "frustrated",
    "frustration": "frustrated",
    "neu": "neutral",
    "neutral_state": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "dis": "disgust",
    "disgust": "disgust",
    "hap": "happy",
    "happiness": "happy",
    "fea": "fear",
    "sur": "surprise",
    "calm": "neutral",
    "boredom": "neutral",
    "excited": "happy",
    "ps": "surprise",
}


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
    Wraps a HuggingFace audio classification model.

    The instance is designed to be created once and reused across requests
    (singleton pattern via get_classifier()).
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading emotion model '%s' on %s", model_id, self.device)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        self._id2label: dict[int, str] = {
            int(k): v for k, v in self.model.config.id2label.items()
        }
        logger.info("Model loaded. Labels: %s", list(self._id2label.values()))

    def predict_segment(self, segment: AudioSegment) -> SegmentPrediction:
        """Run inference on a single AudioSegment and return normalized scores."""
        inputs = self.feature_extractor(
            segment.audio,
            sampling_rate=segment.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        raw_scores: dict[str, float] = {
            self._id2label[i]: float(probs[i]) for i in range(len(probs))
        }

        canonical_scores = _canonicalize_scores(raw_scores)

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
        """Run inference over all segments sequentially."""
        predictions: list[SegmentPrediction] = []
        for seg in segments:
            pred = self.predict_segment(seg)
            logger.debug(
                "Segment %d [%.1f-%.1f s]: %s (%.2f)",
                seg.index,
                seg.start_s,
                seg.end_s,
                pred.dominant_emotion,
                pred.confidence,
            )
            predictions.append(pred)
        return predictions


def _canonicalize_scores(raw_scores: dict[str, float]) -> dict[str, float]:
    """
    Map model-specific label names to canonical emotion labels.
    Scores for the same canonical label are summed.
    """
    canonical: dict[str, float] = {}
    for label, score in raw_scores.items():
        normalized = label.lower().strip()
        canonical_label = LABEL_ALIASES.get(normalized, normalized)
        canonical[canonical_label] = canonical.get(canonical_label, 0.0) + score

    # Ensure all key canonical emotions are present (default 0.0)
    for emotion in ("angry", "neutral", "sad", "disgust", "frustrated"):
        canonical.setdefault(emotion, 0.0)

    return canonical


@lru_cache(maxsize=1)
def get_classifier(model_id: str = DEFAULT_MODEL_ID) -> EmotionClassifier:
    """Return a cached singleton classifier instance."""
    return EmotionClassifier(model_id=model_id)
