"""Save angry/frustrated audio windows as individual WAV files."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

FLAGGED_EMOTIONS = {"angry", "frustrated"}


def save_flagged_segments(
    audio: np.ndarray,
    sr: int,
    timeline: list[dict],
    call_id: str,
    output_dir: Path,
) -> list[dict]:
    """
    For every angry/frustrated window in *timeline*, write a WAV clip to
    ``output_dir/segments/<call_id>/window_<start_s>.wav``.

    Returns the timeline with an added ``"segment_path"`` key on flagged
    entries (relative to *output_dir*), so the dashboard can reference them.
    """
    seg_dir = output_dir / "segments" / call_id
    seg_dir.mkdir(parents=True, exist_ok=True)

    enriched: list[dict] = []
    for entry in timeline:
        entry = dict(entry)
        if entry.get("emotion") in FLAGGED_EMOTIONS:
            start_s = entry["start"]
            end_s = entry["end"]
            start_sample = int(start_s * sr)
            end_sample = min(int(end_s * sr), len(audio))
            clip = audio[start_sample:end_sample]

            fname = f"window_{start_s:.2f}.wav"
            clip_path = seg_dir / fname
            try:
                sf.write(str(clip_path), clip, sr)
                # Store path relative to output_dir so the HTML can reference it.
                entry["segment_path"] = str(
                    clip_path.relative_to(output_dir)
                )
            except Exception as exc:
                logger.warning("Could not save segment %s: %s", fname, exc)

        enriched.append(entry)

    return enriched
