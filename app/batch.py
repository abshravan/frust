"""
Batch-process a folder of call recordings and write results to Excel + JSON.

Usage:
    python batch.py --input-dir /path/to/audio
    python batch.py --input-dir /path/to/audio --output-dir results \
                    --excel results.xlsx --recursive

The emotion model is loaded once and reused across all files. Files are
processed sequentially by default (safest on GPU); pass --workers N > 1 to
fan out to a thread pool on CPU-only machines.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from emotion_model import get_classifier
from main import analyze_file

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

EXCEL_COLUMNS = [
    "filename",
    "call_id",
    "is_frustrating",
    "severity",
    "frustration_score",
    "max_anger_score",
    "num_frustrating_windows",
    "num_frustration_segments",
    "pct_frustrated_windows",
    "longest_angry_streak",
    "escalation_detected",
    "total_windows",
    "duration_s",
    "processing_time_s",
    "reason",
    "status",
]


# ── Per-file processing ──────────────────────────────────────────────────────

def _process_one(audio_path: Path) -> tuple[dict, dict | None]:
    """Run the pipeline on a single file. Returns (row, full_result_or_None)."""
    filename = audio_path.name
    call_id = audio_path.stem
    started = time.time()

    try:
        result = analyze_file(audio_path, filename=filename)
        elapsed = time.time() - started

        timeline = result.get("timeline", [])
        num_windows = len(timeline)
        num_angry = sum(1 for e in timeline if e["emotion"] == "angry")
        num_frustrated = sum(1 for e in timeline if e["emotion"] == "frustrated")
        num_frustrating_windows = num_angry + num_frustrated
        duration_s = timeline[-1]["end"] if timeline else 0.0
        num_segments = _count_angry_segments(timeline)
        longest_streak = _longest_angry_streak(timeline)

        row = {
            "filename": filename,
            "call_id": call_id,
            "is_frustrating": bool(result["flagged"]),
            "severity": result.get("severity", ""),
            "frustration_score": result.get("anger_ratio", 0.0),
            "max_anger_score": result.get("max_anger_score", 0.0),
            "num_frustrating_windows": num_frustrating_windows,
            "num_frustration_segments": num_segments,
            "pct_frustrated_windows": round(num_frustrating_windows / num_windows, 4)
            if num_windows else 0.0,
            "longest_angry_streak": longest_streak,
            "escalation_detected": bool(result.get("escalation_detected", False)),
            "total_windows": num_windows,
            "duration_s": round(duration_s, 2),
            "processing_time_s": round(elapsed, 2),
            "reason": result.get("reason", ""),
            "status": "Success",
        }

        result_with_file = {"filename": filename, **result}
        return row, result_with_file

    except Exception as exc:
        logger.error("Failed %s: %s", filename, exc)
        row = {col: "" for col in EXCEL_COLUMNS}
        row.update({
            "filename": filename,
            "call_id": call_id,
            "is_frustrating": False,
            "status": f"Failed: {str(exc)[:140]}",
        })
        return row, None


def _count_angry_segments(timeline: list[dict]) -> int:
    """Count runs of consecutive 'angry' windows (frustration segments)."""
    segments = 0
    in_segment = False
    for entry in timeline:
        if entry["emotion"] == "angry":
            if not in_segment:
                segments += 1
                in_segment = True
        else:
            in_segment = False
    return segments


def _longest_angry_streak(timeline: list[dict]) -> int:
    longest = current = 0
    for entry in timeline:
        if entry["emotion"] == "angry":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


# ── Folder walk ──────────────────────────────────────────────────────────────

def _find_audio_files(root: Path, recursive: bool) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    pattern_fn = root.rglob if recursive else root.glob
    found: set[Path] = set()
    for ext in AUDIO_EXTENSIONS:
        found.update(pattern_fn(f"*{ext}"))
        found.update(pattern_fn(f"*{ext.upper()}"))
    return sorted(found)


# ── Batch driver ─────────────────────────────────────────────────────────────

def batch_process(
    input_dir: str | Path,
    output_dir: str | Path = "results",
    excel_filename: str = "frustration_detection_results.xlsx",
    workers: int = 1,
    recursive: bool = False,
) -> pd.DataFrame:
    """Walk `input_dir`, analyze each audio file, and write Excel + JSON output."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _find_audio_files(input_dir, recursive=recursive)
    if not files:
        raise ValueError(f"No audio files found in {input_dir}")

    logger.info("Found %d audio files in %s", len(files), input_dir)

    # Pre-load the classifier once so it's shared across all invocations.
    get_classifier()

    rows: list[dict] = []
    full_results: list[dict] = []

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, f): f for f in files}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Analyzing"):
                row, full = fut.result()
                rows.append(row)
                if full:
                    full_results.append(full)
    else:
        for f in tqdm(files, desc="Analyzing"):
            row, full = _process_one(f)
            rows.append(row)
            if full:
                full_results.append(full)

    df = pd.DataFrame(rows, columns=EXCEL_COLUMNS)
    df = df.sort_values("filename").reset_index(drop=True)

    excel_path = output_dir / excel_filename
    _write_excel(df, excel_path)

    json_path = output_dir / "full_results.json"
    with json_path.open("w") as f:
        json.dump(full_results, f, indent=2)

    _print_summary(df, excel_path, json_path)
    return df


# ── Excel writer with light formatting ───────────────────────────────────────

def _write_excel(df: pd.DataFrame, path: Path) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Summary", index=False)
        worksheet = writer.sheets["Summary"]

        # Auto-size columns to longest value (capped at 50 chars).
        for idx, col in enumerate(df.columns, start=1):
            width = max(len(str(col)), df[col].astype(str).str.len().max() or 0)
            worksheet.column_dimensions[_col_letter(idx)].width = min(width + 2, 50)


def _col_letter(idx: int) -> str:
    letters = ""
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _print_summary(df: pd.DataFrame, excel_path: Path, json_path: Path) -> None:
    total = len(df)
    frustrating = int(df["is_frustrating"].sum())
    success = int((df["status"] == "Success").sum())
    avg_time = df.loc[df["status"] == "Success", "processing_time_s"].mean() or 0.0

    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETED")
    print("=" * 60)
    print(f"Total files           : {total}")
    print(f"Successful            : {success}")
    print(f"Frustrating calls     : {frustrating} ({frustrating/total*100:.1f}%)")
    print(f"Avg processing time   : {avg_time:.2f}s / file")
    print(f"Excel report          : {excel_path}")
    print(f"Full JSON results     : {json_path}")
    print("=" * 60)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch-analyze a folder of call audio for patient frustration."
    )
    p.add_argument("--input-dir", required=True, help="Folder containing audio files")
    p.add_argument("--output-dir", default="results", help="Where to write outputs")
    p.add_argument(
        "--excel", default="frustration_detection_results.xlsx",
        help="Excel filename (written inside --output-dir)",
    )
    p.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers (keep at 1 on GPU; raise on CPU).",
    )
    p.add_argument(
        "--recursive", action="store_true",
        help="Recurse into subdirectories",
    )
    return p


def main() -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    args = _build_parser().parse_args()
    batch_process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        excel_filename=args.excel,
        workers=args.workers,
        recursive=args.recursive,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
