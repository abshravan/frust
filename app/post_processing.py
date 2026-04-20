"""Smoothing, flagging, escalation detection, and report generation."""

import logging
from dataclasses import dataclass, field

import numpy as np

from emotion_model import SegmentPrediction

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
ANGRY_THRESHOLD = 0.65
DISGUST_THRESHOLD = 0.60

# ── Flagging criteria ─────────────────────────────────────────────────────────
CONSECUTIVE_ANGRY_LIMIT = 2
ANGER_RATIO_LIMIT = 0.25
MAX_ANGER_SCORE_LIMIT = 0.80

# ── Smoothing ─────────────────────────────────────────────────────────────────
SMOOTHING_WINDOW = 3

# ── Escalation pattern ────────────────────────────────────────────────────────
ESCALATION_CHAIN = ("neutral", "frustrated", "angry")

# ── Severity bands ────────────────────────────────────────────────────────────
_SEVERITY_BANDS = [
    (0.0, 0.25, "mild"),
    (0.25, 0.50, "moderate"),
    (0.50, 1.01, "severe"),
]


@dataclass
class TimelineEntry:
    start: float
    end: float
    emotion: str
    confidence: float


@dataclass
class AnalysisReport:
    flagged: bool
    reason: str
    anger_ratio: float
    max_anger_score: float
    escalation_detected: bool
    severity: str
    timeline: list[TimelineEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "flagged": self.flagged,
            "reason": self.reason,
            "anger_ratio": round(self.anger_ratio, 4),
            "max_anger_score": round(self.max_anger_score, 4),
            "escalation_detected": self.escalation_detected,
            "severity": self.severity,
            "timeline": [
                {
                    "start": e.start,
                    "end": e.end,
                    "emotion": e.emotion,
                    "confidence": round(e.confidence, 4),
                }
                for e in self.timeline
            ],
        }


# ── Public API ────────────────────────────────────────────────────────────────

def build_report(predictions: list[SegmentPrediction]) -> AnalysisReport:
    """
    Full post-processing pipeline:
      1. Smooth probability scores
      2. Re-classify dominant emotion per window
      3. Compute metrics
      4. Apply flagging rules
      5. Detect escalation
      6. Build timeline
    """
    if not predictions:
        return AnalysisReport(
            flagged=False,
            reason="No audio segments to analyze.",
            anger_ratio=0.0,
            max_anger_score=0.0,
            escalation_detected=False,
            severity="mild",
            timeline=[],
        )

    smoothed = _smooth_predictions(predictions)
    classified = _classify(smoothed)

    anger_ratio = _anger_ratio(classified)
    max_anger = _max_anger_score(smoothed)
    consecutive_angry = _max_consecutive_angry(classified)
    escalation = _detect_escalation(classified)

    flagged, reason = _apply_flagging_rules(
        consecutive_angry, anger_ratio, max_anger
    )

    severity = _compute_severity(anger_ratio, max_anger, escalation)

    timeline = [
        TimelineEntry(
            start=p.start_s,
            end=p.end_s,
            emotion=label,
            confidence=score,
        )
        for p, (label, score) in zip(predictions, classified)
    ]

    report = AnalysisReport(
        flagged=flagged,
        reason=reason,
        anger_ratio=anger_ratio,
        max_anger_score=max_anger,
        escalation_detected=escalation,
        severity=severity,
        timeline=timeline,
    )

    logger.info(
        "Report: flagged=%s reason='%s' anger_ratio=%.2f max_anger=%.2f "
        "escalation=%s severity=%s windows=%d",
        flagged,
        reason,
        anger_ratio,
        max_anger,
        escalation,
        severity,
        len(predictions),
    )
    return report


# ── Internal helpers ──────────────────────────────────────────────────────────

def _smooth_predictions(
    predictions: list[SegmentPrediction],
) -> list[dict[str, float]]:
    """
    Moving-average smoothing over per-emotion probability scores.
    Returns a list of smoothed score dicts aligned with predictions.
    """
    n = len(predictions)
    half = SMOOTHING_WINDOW // 2

    # Collect all emotion keys across all predictions
    emotion_keys: set[str] = set()
    for p in predictions:
        emotion_keys.update(p.scores.keys())

    # Build (n x emotions) matrix
    score_matrix = {
        key: np.array([p.scores.get(key, 0.0) for p in predictions], dtype=float)
        for key in emotion_keys
    }

    smoothed_scores: list[dict[str, float]] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_scores = {key: float(np.mean(score_matrix[key][lo:hi])) for key in emotion_keys}
        smoothed_scores.append(window_scores)

    return smoothed_scores


def _classify(
    smoothed: list[dict[str, float]],
) -> list[tuple[str, float]]:
    """
    Determine dominant emotion and confidence from smoothed scores.
    Applies per-emotion thresholds to override if below threshold.
    """
    results: list[tuple[str, float]] = []
    for scores in smoothed:
        dominant = max(scores, key=scores.get)
        conf = scores[dominant]

        # Threshold gating: if the dominant emotion's score is below its
        # threshold, fall back to neutral.
        if dominant == "angry" and conf < ANGRY_THRESHOLD:
            dominant = _next_best(scores, exclude="angry")
        elif dominant == "disgust" and conf < DISGUST_THRESHOLD:
            dominant = _next_best(scores, exclude="disgust")

        conf = scores[dominant]
        results.append((dominant, round(conf, 4)))

    return results


def _next_best(scores: dict[str, float], exclude: str) -> str:
    filtered = {k: v for k, v in scores.items() if k != exclude}
    if not filtered:
        return "neutral"
    return max(filtered, key=filtered.get)


def _anger_ratio(classified: list[tuple[str, float]]) -> float:
    if not classified:
        return 0.0
    angry_count = sum(1 for label, _ in classified if label == "angry")
    return round(angry_count / len(classified), 4)


def _max_anger_score(smoothed: list[dict[str, float]]) -> float:
    scores = [s.get("angry", 0.0) for s in smoothed]
    return round(float(max(scores)), 4) if scores else 0.0


def _max_consecutive_angry(classified: list[tuple[str, float]]) -> int:
    max_run = current_run = 0
    for label, _ in classified:
        if label == "angry":
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def _detect_escalation(classified: list[tuple[str, float]]) -> bool:
    """
    True if ESCALATION_CHAIN (neutral → frustrated → angry) appears
    as a subsequence in the dominant-emotion sequence.
    """
    chain = list(ESCALATION_CHAIN)
    chain_idx = 0
    for label, _ in classified:
        if label == chain[chain_idx]:
            chain_idx += 1
            if chain_idx == len(chain):
                return True
    return False


def _apply_flagging_rules(
    consecutive_angry: int,
    anger_ratio: float,
    max_anger: float,
) -> tuple[bool, str]:
    reasons: list[str] = []

    if consecutive_angry >= CONSECUTIVE_ANGRY_LIMIT:
        reasons.append(
            f"{consecutive_angry} consecutive angry windows "
            f"(threshold: {CONSECUTIVE_ANGRY_LIMIT})"
        )
    if anger_ratio > ANGER_RATIO_LIMIT:
        reasons.append(
            f"anger_ratio={anger_ratio:.2%} exceeds {ANGER_RATIO_LIMIT:.0%}"
        )
    if max_anger > MAX_ANGER_SCORE_LIMIT:
        reasons.append(
            f"max_anger_score={max_anger:.2f} exceeds {MAX_ANGER_SCORE_LIMIT}"
        )

    if reasons:
        return True, "; ".join(reasons)
    return False, "No significant anger detected."


def _compute_severity(
    anger_ratio: float,
    max_anger: float,
    escalation: bool,
) -> str:
    composite = (anger_ratio + max_anger) / 2
    if escalation:
        composite = min(composite + 0.15, 1.0)

    for lo, hi, label in _SEVERITY_BANDS:
        if lo <= composite < hi:
            return label
    return "severe"
