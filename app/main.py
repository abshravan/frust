"""
FastAPI application entry point.

Start with:
    uvicorn main:app --host 0.0.0.0 --port 8000

Environment variables:
    EMOTION_MODEL_ID  – HuggingFace model ID (default: ehcalabres/wav2vec2-...)
    LOG_LEVEL         – Python log level string (default: INFO)
    ENABLE_WHISPER    – "1" to enable Whisper transcription bonus feature
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from audio_loader import audio_info, load_audio
from emotion_model import DEFAULT_MODEL_ID, get_classifier
from post_processing import build_report
from segmenter import segment_audio

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = os.getenv("EMOTION_MODEL_ID", DEFAULT_MODEL_ID)
ENABLE_WHISPER = os.getenv("ENABLE_WHISPER", "0") == "1"
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Patient Emotion Analyzer",
    description=(
        "Analyzes mono patient-side call audio for emotional states "
        "to evaluate voice-bot call quality."
    ),
    version="1.0.0",
)


@app.on_event("startup")
async def _startup() -> None:
    """Pre-load the model so the first request isn't slow."""
    logger.info("Pre-loading emotion model: %s", MODEL_ID)
    get_classifier(MODEL_ID)
    logger.info("Model ready.")


@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {"status": "ok", "model": MODEL_ID}


@app.post("/analyze", tags=["analysis"])
async def analyze(
    file: Annotated[UploadFile, File(description="WAV or MP3 patient audio (mono preferred)")],
) -> JSONResponse:
    """
    Analyze a patient audio file for emotional content.

    Returns a structured JSON report with flagging, anger metrics,
    escalation detection, and a per-window timeline.
    """
    _validate_upload(file)

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024**2)} MB limit.",
        )

    suffix = Path(file.filename or "audio.wav").suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = Path(tmp.name)

    try:
        report_dict = _run_pipeline(tmp_path, file.filename or "unknown")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during analysis of %s", file.filename)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)

    return JSONResponse(content=report_dict)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _run_pipeline(audio_path: Path, filename: str) -> dict:
    logger.info("── BEGIN analysis: %s ──", filename)

    # 1. Load & normalize
    audio, sr = load_audio(audio_path)
    info = audio_info(audio, sr)
    logger.info("Audio info: %s", info)

    # 2. Segment
    segments = segment_audio(audio, sr)
    if not segments:
        raise ValueError("Audio produced zero segments after segmentation.")

    # 3. Classify emotions
    classifier = get_classifier(MODEL_ID)
    predictions = classifier.predict_all(segments)

    # 4. Bonus: Whisper transcription + text-emotion fusion
    if ENABLE_WHISPER:
        try:
            _apply_whisper_fusion(audio_path, predictions)
        except Exception as exc:
            logger.warning("Whisper transcription failed (non-fatal): %s", exc)

    # 5. Build report
    report = build_report(predictions)

    logger.info("── END analysis: flagged=%s severity=%s ──", report.flagged, report.severity)
    return report.to_dict()


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_upload(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )


# ── Bonus: Whisper + text emotion fusion ─────────────────────────────────────

def _apply_whisper_fusion(audio_path: Path, predictions: list) -> None:
    """
    Transcribe audio with Whisper, run a simple text-based sentiment heuristic,
    and blend scores into each segment's prediction.

    This is a lightweight text-fusion approach; replace with a proper
    text-emotion classifier for production use.
    """
    import whisper  # type: ignore

    logger.info("Running Whisper transcription…")
    model = whisper.load_model("base")
    result = model.transcribe(str(audio_path))
    transcript: str = result.get("text", "").lower()
    logger.info("Transcript: %s", transcript[:200])

    anger_words = {
        "angry", "furious", "hate", "terrible", "awful", "unacceptable",
        "ridiculous", "useless", "stupid", "idiot", "terrible", "horrible",
    }
    frustrated_words = {
        "frustrated", "again", "waiting", "still", "never", "keep", "broken",
        "wrong", "fix", "problem", "issue", "not working",
    }

    words = set(transcript.split())
    anger_boost = 0.05 * len(words & anger_words)
    frustration_boost = 0.03 * len(words & frustrated_words)

    for pred in predictions:
        pred.scores["angry"] = min(1.0, pred.scores.get("angry", 0.0) + anger_boost)
        pred.scores["frustrated"] = min(
            1.0, pred.scores.get("frustrated", 0.0) + frustration_boost
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
