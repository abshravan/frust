"""Emotion classification from audio segments using HuggingFace audio-classification pipeline."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

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
    label_map: dict[str, str]


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

MODELS_REGISTRY: list[ModelConfig] = [
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
]

DEFAULT_MODEL_ID = MODELS_REGISTRY[0].model_id

CANONICAL_EMOTIONS = ("angry", "frustrated", "neutral", "sad", "disgust", "happy", "fear")


# ── Device / dtype helpers ────────────────────────────────────────────────────

def _resolve_device() -> str:
    """
    Return the best available device string.

    Priority: EMOTION_DEVICE env var → CUDA → MPS → CPU.
    """
    env = os.getenv("EMOTION_DEVICE", "").strip()
    if env:
        return env
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(device: str) -> torch.dtype:
    """fp16 on CUDA (halves VRAM and speeds inference); fp32 everywhere else."""
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


def _resolve_batch_size(device: str) -> int:
    env = os.getenv("EMOTION_BATCH_SIZE", "").strip()
    if env:
        return int(env)
    # GPU can process many segments in parallel; CPU is memory-bound.
    return 16 if device.startswith("cuda") or device == "mps" else 4


def _log_gpu_info(device: str) -> None:
    if not device.startswith("cuda"):
        return
    try:
        idx = int(device.split(":")[-1])
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / 1024 ** 3
        alloc_gb = torch.cuda.memory_allocated(idx) / 1024 ** 3
        logger.info(
            "GPU: %s | VRAM %.1f GB total / %.1f GB allocated",
            props.name, total_gb, alloc_gb,
        )
    except Exception:
        pass


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

    def __init__(
        self,
        config: ModelConfig,
        device: str | None = None,
        batch_size: int | None = None,
    ):
        self.config = config
        self.device = device or _resolve_device()
        self.batch_size = batch_size or _resolve_batch_size(self.device)
        dtype = _resolve_dtype(self.device)

        logger.info(
            "Loading [%s] '%s' | device=%s | dtype=%s | batch_size=%d",
            config.short_name, config.model_id,
            self.device, dtype, self.batch_size,
        )
        _log_gpu_info(self.device)

        self._pipe = pipeline(
            "audio-classification",
            model=config.model_id,
            device=self.device,
            torch_dtype=dtype,
            top_k=None,
            batch_size=self.batch_size,
        )
        logger.info("[%s] ready.", config.short_name)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_all(self, segments: list[AudioSegment]) -> list[SegmentPrediction]:
        """
        Classify all segments in one batched call to the HF pipeline.

        Passing a list lets the pipeline chunk inputs into batches of
        `self.batch_size` automatically, saturating the GPU without
        loading all segments into VRAM at once.
        """
        if not segments:
            return []

        inputs = [
            {"array": seg.audio.astype(np.float32), "sampling_rate": seg.sample_rate}
            for seg in segments
        ]

        # Returns list[list[{label, score}]] when top_k=None
        all_raw: list[list[dict]] = self._pipe(inputs)

        predictions: list[SegmentPrediction] = []
        for seg, raw_preds in zip(segments, all_raw):
            pred = self._decode(seg, raw_preds)
            logger.debug(
                "[%s] seg %d [%.2f-%.2f s]: %s (%.3f)",
                self.config.short_name, seg.index,
                seg.start_s, seg.end_s,
                pred.dominant_emotion, pred.confidence,
            )
            predictions.append(pred)

        return predictions

    def predict_segment(self, segment: AudioSegment) -> SegmentPrediction:
        """Single-segment convenience wrapper (batches of 1)."""
        return self.predict_all([segment])[0]

    def _decode(self, segment: AudioSegment, raw_preds: list[dict]) -> SegmentPrediction:
        canonical_scores: dict[str, float] = {}
        for p in raw_preds:
            raw = p["label"]
            canon = self.config.label_map.get(raw)
            if canon is None:
                canon = _COMMON_ALIASES.get(raw.lower(), raw.lower())
            canonical_scores[canon] = canonical_scores.get(canon, 0.0) + float(p["score"])

        for emotion in CANONICAL_EMOTIONS:
            canonical_scores.setdefault(emotion, 0.0)

        dominant = max(canonical_scores, key=canonical_scores.get)
        return SegmentPrediction(
            segment_index=segment.index,
            start_s=segment.start_s,
            end_s=segment.end_s,
            dominant_emotion=dominant,
            confidence=round(canonical_scores[dominant], 4),
            scores={k: round(v, 4) for k, v in canonical_scores.items()},
        )


# ── Singleton cache (one instance per model_id) ───────────────────────────────

_classifier_cache: dict[str, EmotionClassifier] = {}


def get_classifier(model_id: str = DEFAULT_MODEL_ID) -> EmotionClassifier:
    """Return a cached EmotionClassifier for the given model_id."""
    if model_id not in _classifier_cache:
        config = next(
            (m for m in MODELS_REGISTRY if m.model_id == model_id),
            ModelConfig(model_id=model_id, short_name=model_id.split("/")[-1], label_map={}),
        )
        _classifier_cache[model_id] = EmotionClassifier(config)
    return _classifier_cache[model_id]


def preload_all_models() -> None:
    """Pre-load every registered model into the cache (call at startup)."""
    for cfg in MODELS_REGISTRY:
        get_classifier(cfg.model_id)
