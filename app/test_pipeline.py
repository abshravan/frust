"""
Example test script for the patient emotion analysis pipeline.

Usage:
    # Unit tests (no model download required)
    python test_pipeline.py --unit

    # Integration test against a local API server
    python test_pipeline.py --api --file path/to/audio.wav

    # Full offline pipeline test (downloads model on first run)
    python test_pipeline.py --offline --file path/to/audio.wav
"""

import argparse
import json
import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestAudioLoader(unittest.TestCase):
    def test_normalize_positive(self):
        from audio_loader import _normalize
        audio = np.array([0.0, 0.5, 1.0, -0.5], dtype=np.float32)
        result = _normalize(audio)
        self.assertAlmostEqual(float(np.max(np.abs(result))), 1.0)

    def test_normalize_silent(self):
        from audio_loader import _normalize
        audio = np.zeros(100, dtype=np.float32)
        result = _normalize(audio)
        self.assertTrue(np.all(result == 0))

    def test_rms_db_silent(self):
        from audio_loader import _rms_db
        audio = np.zeros(100, dtype=np.float32)
        self.assertLessEqual(_rms_db(audio), -100)

    def test_rms_db_full_scale(self):
        from audio_loader import _rms_db
        audio = np.ones(100, dtype=np.float32)
        self.assertAlmostEqual(_rms_db(audio), 0.0, places=1)


class TestSegmenter(unittest.TestCase):
    def _make_audio(self, duration_s: float, sr: int = 16_000) -> np.ndarray:
        return np.random.randn(int(duration_s * sr)).astype(np.float32)

    def test_segment_count(self):
        from segmenter import segment_audio
        audio = self._make_audio(10.0)
        segs = segment_audio(audio, 16_000, window_s=5.0, overlap_s=2.0)
        # step = 3s → windows at 0, 3, 6, 9 → 4 windows for 10s
        self.assertGreaterEqual(len(segs), 3)

    def test_segment_shape(self):
        from segmenter import segment_audio
        sr = 16_000
        audio = self._make_audio(7.0, sr)
        segs = segment_audio(audio, sr, window_s=5.0, overlap_s=2.0)
        for seg in segs:
            self.assertEqual(len(seg.audio), int(5.0 * sr))

    def test_overlap_invalid(self):
        from segmenter import segment_audio
        with self.assertRaises(ValueError):
            segment_audio(np.zeros(16_000), 16_000, window_s=3.0, overlap_s=5.0)

    def test_short_audio_padded(self):
        from segmenter import segment_audio
        sr = 16_000
        # 3s audio, window=5s → one padded segment
        audio = self._make_audio(3.0, sr)
        segs = segment_audio(audio, sr, window_s=5.0, overlap_s=2.0)
        self.assertEqual(len(segs), 1)
        self.assertEqual(len(segs[0].audio), 5 * sr)


class TestPostProcessing(unittest.TestCase):
    def _make_prediction(self, idx, start, end, emotion, conf=0.9):
        from emotion_model import SegmentPrediction
        scores = {emotion: conf, "neutral": 1 - conf}
        return SegmentPrediction(
            segment_index=idx,
            start_s=start,
            end_s=end,
            dominant_emotion=emotion,
            confidence=conf,
            scores=scores,
        )

    def test_empty_predictions(self):
        from post_processing import build_report
        report = build_report([])
        self.assertFalse(report.flagged)
        self.assertEqual(report.timeline, [])

    def test_anger_ratio(self):
        from post_processing import _anger_ratio
        classified = [("angry", 0.9), ("neutral", 0.8), ("angry", 0.7), ("angry", 0.85)]
        self.assertAlmostEqual(_anger_ratio(classified), 0.75)

    def test_max_consecutive_angry(self):
        from post_processing import _max_consecutive_angry
        classified = [
            ("neutral", 0.8),
            ("angry", 0.9),
            ("angry", 0.85),
            ("angry", 0.7),
            ("neutral", 0.8),
        ]
        self.assertEqual(_max_consecutive_angry(classified), 3)

    def test_escalation_detected(self):
        from post_processing import _detect_escalation
        seq = [
            ("neutral", 0.9),
            ("sad", 0.7),
            ("frustrated", 0.8),
            ("angry", 0.9),
        ]
        self.assertTrue(_detect_escalation(seq))

    def test_escalation_not_detected(self):
        from post_processing import _detect_escalation
        seq = [("neutral", 0.9), ("angry", 0.9)]
        self.assertFalse(_detect_escalation(seq))

    def test_flagging_consecutive(self):
        from post_processing import _apply_flagging_rules
        flagged, reason = _apply_flagging_rules(
            consecutive_angry=3, anger_ratio=0.1, max_anger=0.5
        )
        self.assertTrue(flagged)
        self.assertIn("consecutive", reason)

    def test_flagging_ratio(self):
        from post_processing import _apply_flagging_rules
        flagged, reason = _apply_flagging_rules(
            consecutive_angry=1, anger_ratio=0.5, max_anger=0.5
        )
        self.assertTrue(flagged)
        self.assertIn("anger_ratio", reason)

    def test_no_flag(self):
        from post_processing import _apply_flagging_rules
        flagged, _ = _apply_flagging_rules(
            consecutive_angry=1, anger_ratio=0.1, max_anger=0.5
        )
        self.assertFalse(flagged)

    def test_full_report_structure(self):
        from post_processing import build_report
        preds = [
            self._make_prediction(0, 0.0, 5.0, "neutral", 0.9),
            self._make_prediction(1, 3.0, 8.0, "frustrated", 0.8),
            self._make_prediction(2, 6.0, 11.0, "angry", 0.9),
        ]
        report = build_report(preds)
        d = report.to_dict()
        for key in ("flagged", "reason", "anger_ratio", "max_anger_score",
                     "escalation_detected", "severity", "timeline"):
            self.assertIn(key, d)
        self.assertEqual(len(d["timeline"]), 3)


# ── Offline pipeline test ─────────────────────────────────────────────────────

def run_offline_pipeline(audio_file: str) -> None:
    """Run the full pipeline without the HTTP server."""
    sys.path.insert(0, str(Path(__file__).parent))

    from audio_loader import load_audio, audio_info
    from segmenter import segment_audio
    from emotion_model import get_classifier
    from post_processing import build_report

    logger.info("Loading audio: %s", audio_file)
    audio, sr = load_audio(audio_file)
    logger.info("Audio info: %s", audio_info(audio, sr))

    segments = segment_audio(audio, sr)
    logger.info("Segments: %d", len(segments))

    classifier = get_classifier()
    predictions = classifier.predict_all(segments)

    report = build_report(predictions)
    print(json.dumps(report.to_dict(), indent=2))


# ── API integration test ──────────────────────────────────────────────────────

def run_api_test(audio_file: str, base_url: str = "http://localhost:8000") -> None:
    import requests  # type: ignore

    with open(audio_file, "rb") as f:
        files = {"file": (Path(audio_file).name, f, "audio/wav")}
        logger.info("POST %s/analyze", base_url)
        resp = requests.post(f"{base_url}/analyze", files=files, timeout=120)

    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2))


# ── Synthetic smoke test ──────────────────────────────────────────────────────

def generate_synthetic_audio(path: str, duration_s: float = 15.0, sr: int = 16_000) -> None:
    """Generate a synthetic WAV file for smoke-testing (no real speech)."""
    import soundfile as sf

    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(path, audio, sr)
    logger.info("Synthetic audio written to %s (%.1fs)", path, duration_s)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Patient emotion analysis tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--offline", action="store_true", help="Run offline pipeline test")
    parser.add_argument("--api", action="store_true", help="Run API integration test")
    parser.add_argument("--file", default=None, help="Audio file for offline/api tests")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate + test a synthetic WAV (smoke test, no real emotion)"
    )
    args = parser.parse_args()

    if args.unit:
        sys.argv = [sys.argv[0]]  # prevent unittest from re-parsing args
        unittest.main(verbosity=2, exit=True)

    if args.synthetic:
        tmp = "/tmp/synthetic_test.wav"
        generate_synthetic_audio(tmp)
        args.file = tmp
        args.offline = True

    if args.offline:
        if not args.file:
            parser.error("--offline requires --file")
        run_offline_pipeline(args.file)

    if args.api:
        if not args.file:
            parser.error("--api requires --file")
        run_api_test(args.file, base_url=args.url)

    if not any([args.unit, args.offline, args.api, args.synthetic]):
        parser.print_help()


if __name__ == "__main__":
    main()
