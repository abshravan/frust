"""
Batch-process a folder of call recordings across three independent models
and write a consolidated Excel report + JSON dump.

Usage:
    python batch.py --input-dir /path/to/audio
    python batch.py --input-dir ./calls --output-dir results \
                    --excel results.xlsx --recursive --workers 4

Each audio file is run through all three registered models independently.
The Excel sheet contains per-model columns plus a majority-vote ensemble.
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

from audio_loader import load_audio
from emotion_model import MODELS_REGISTRY, ModelConfig, get_classifier, preload_all_models
from post_processing import build_report
from segmenter import segment_audio

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# ── Column layout ─────────────────────────────────────────────────────────────
# Filled once per file (independent of models).
_BASE_COLUMNS = [
    "filename",
    "call_id",
    "duration_s",
    "total_windows",
    "processing_time_s",
    "status",
]

# Filled per model (prefixed with model short_name).
_PER_MODEL_SUFFIX = [
    "is_frustrating",
    "severity",
    "frustration_score",       # anger_ratio
    "max_anger_score",
    "num_frustrating_windows", # angry + frustrated windows
    "num_frustration_segments",# consecutive angry runs
    "pct_frustrated_windows",
    "longest_angry_streak",
    "escalation_detected",
    "reason",
]

# Ensemble columns (added at the end).
_ENSEMBLE_COLUMNS = [
    "ensemble_is_frustrating",  # majority vote across models
    "ensemble_num_models_flagged",
]


def _all_columns() -> list[str]:
    cols = list(_BASE_COLUMNS)
    for cfg in MODELS_REGISTRY:
        for suffix in _PER_MODEL_SUFFIX:
            cols.append(f"{cfg.short_name}_{suffix}")
    cols.extend(_ENSEMBLE_COLUMNS)
    return cols


# ── Per-model metrics from a report dict ─────────────────────────────────────

def _extract_metrics(report: dict, short_name: str) -> dict:
    """Pull the scalar metrics we want into a flat dict with prefixed keys."""
    timeline = report.get("timeline", [])
    num_windows = len(timeline)
    num_angry = sum(1 for e in timeline if e["emotion"] == "angry")
    num_frustrated = sum(1 for e in timeline if e["emotion"] == "frustrated")
    num_frustrating = num_angry + num_frustrated
    duration_s = timeline[-1]["end"] if timeline else 0.0
    longest = _longest_angry_streak(timeline)
    num_segs = _count_angry_segments(timeline)

    p = short_name
    return {
        f"{p}_is_frustrating":        bool(report.get("flagged", False)),
        f"{p}_severity":               report.get("severity", ""),
        f"{p}_frustration_score":      report.get("anger_ratio", 0.0),
        f"{p}_max_anger_score":        report.get("max_anger_score", 0.0),
        f"{p}_num_frustrating_windows": num_frustrating,
        f"{p}_num_frustration_segments": num_segs,
        f"{p}_pct_frustrated_windows":  round(num_frustrating / num_windows, 4)
                                        if num_windows else 0.0,
        f"{p}_longest_angry_streak":   longest,
        f"{p}_escalation_detected":    bool(report.get("escalation_detected", False)),
        f"{p}_reason":                 report.get("reason", ""),
        "_duration_s":                 round(duration_s, 2),
        "_total_windows":              num_windows,
    }


def _longest_angry_streak(timeline: list[dict]) -> int:
    longest = current = 0
    for e in timeline:
        if e["emotion"] == "angry":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _count_angry_segments(timeline: list[dict]) -> int:
    count, in_seg = 0, False
    for e in timeline:
        if e["emotion"] == "angry":
            if not in_seg:
                count += 1
                in_seg = True
        else:
            in_seg = False
    return count


# ── Core: run all models on one file ─────────────────────────────────────────

def _run_all_models(audio_path: Path) -> tuple[dict, list[dict]]:
    """
    Load audio once, segment once, then classify with every registered model.

    Returns:
        row       – flat dict of all Excel columns for this file
        full_data – list of per-model report dicts (for JSON dump)
    """
    filename = audio_path.name
    call_id = audio_path.stem
    t0 = time.time()

    # Shared audio loading + segmentation.
    audio, sr = load_audio(audio_path)
    segments = segment_audio(audio, sr)

    row: dict = {"filename": filename, "call_id": call_id}
    full_data: list[dict] = []
    duration_s: float = 0.0
    total_windows: int = len(segments)

    for cfg in MODELS_REGISTRY:
        classifier = get_classifier(cfg.model_id)
        predictions = classifier.predict_all(segments)
        report = build_report(predictions)
        report_dict = report.to_dict()

        metrics = _extract_metrics(report_dict, cfg.short_name)
        duration_s = max(duration_s, metrics.pop("_duration_s", 0.0))
        total_windows = metrics.pop("_total_windows", total_windows)
        row.update(metrics)

        full_data.append({
            "filename": filename,
            "model": cfg.short_name,
            "model_id": cfg.model_id,
            **report_dict,
        })

    # Ensemble majority vote.
    flags = [bool(row.get(f"{cfg.short_name}_is_frustrating", False))
             for cfg in MODELS_REGISTRY]
    num_flagged = sum(flags)
    row["ensemble_is_frustrating"] = num_flagged >= (len(MODELS_REGISTRY) / 2 + 0.5)
    row["ensemble_num_models_flagged"] = num_flagged

    row["duration_s"] = duration_s
    row["total_windows"] = total_windows
    row["processing_time_s"] = round(time.time() - t0, 2)
    row["status"] = "Success"

    return row, full_data


def _process_one(audio_path: Path) -> tuple[dict, list[dict]]:
    """Wrapper that catches exceptions and returns a failure row."""
    try:
        return _run_all_models(audio_path)
    except Exception as exc:
        logger.error("Failed %s: %s", audio_path.name, exc)
        row: dict = {col: "" for col in _all_columns()}
        row.update({
            "filename": audio_path.name,
            "call_id":  audio_path.stem,
            "status":   f"Failed: {str(exc)[:140]}",
        })
        return row, []


# ── Folder walker ─────────────────────────────────────────────────────────────

def _find_audio_files(root: Path, recursive: bool) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    glob = root.rglob if recursive else root.glob
    found: set[Path] = set()
    for ext in AUDIO_EXTENSIONS:
        found.update(glob(f"*{ext}"))
        found.update(glob(f"*{ext.upper()}"))
    return sorted(found)


# ── Main batch driver ─────────────────────────────────────────────────────────

def batch_process(
    input_dir: str | Path,
    output_dir: str | Path = "results",
    excel_filename: str = "frustration_detection_results.xlsx",
    workers: int = 1,
    recursive: bool = False,
) -> pd.DataFrame:
    """Walk `input_dir`, run all three models on each file, write Excel + JSON."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _find_audio_files(input_dir, recursive=recursive)
    if not files:
        raise ValueError(f"No audio files found in {input_dir}")

    logger.info("Found %d audio files in %s", len(files), input_dir)

    # Pre-load all models before spawning threads.
    logger.info("Pre-loading %d models…", len(MODELS_REGISTRY))
    preload_all_models()

    rows: list[dict] = []
    all_full: list[dict] = []

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, f): f for f in files}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Analyzing"):
                row, full = fut.result()
                rows.append(row)
                all_full.extend(full)
    else:
        for f in tqdm(files, desc="Analyzing"):
            row, full = _process_one(f)
            rows.append(row)
            all_full.extend(full)

    columns = _all_columns()
    df = pd.DataFrame(rows, columns=columns)
    df = df.sort_values("filename").reset_index(drop=True)

    excel_path = output_dir / excel_filename
    _write_excel(df, excel_path)

    json_path = output_dir / "full_results.json"
    with json_path.open("w") as f:
        json.dump(all_full, f, indent=2)

    _print_summary(df, excel_path, json_path)
    return df


# ── Excel writer ──────────────────────────────────────────────────────────────

def _write_excel(df: pd.DataFrame, path: Path) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Summary", index=False)
        ws = writer.sheets["Summary"]

        # Auto-size + basic header formatting.
        from openpyxl.styles import Font, PatternFill
        header_fill = PatternFill("solid", fgColor="1F4E79")
        header_font = Font(color="FFFFFF", bold=True)

        for idx, col in enumerate(df.columns, start=1):
            cell = ws.cell(row=1, column=idx)
            cell.fill = header_fill
            cell.font = header_font

            max_len = max(
                len(str(col)),
                df[col].astype(str).str.len().max() or 0,
            )
            ws.column_dimensions[_col_letter(idx)].width = min(max_len + 2, 52)

        # Shade every other data row for readability.
        alt_fill = PatternFill("solid", fgColor="D9E1F2")
        for row_idx in range(2, len(df) + 2):
            if row_idx % 2 == 0:
                for col_idx in range(1, len(df.columns) + 1):
                    ws.cell(row=row_idx, column=col_idx).fill = alt_fill

        # Freeze the header row.
        ws.freeze_panes = "A2"


def _col_letter(idx: int) -> str:
    letters = ""
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame, excel_path: Path, json_path: Path) -> None:
    total = len(df)
    ok = df["status"] == "Success"
    print("\n" + "=" * 64)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 64)
    print(f"  Files processed  : {total}")
    print(f"  Successful       : {int(ok.sum())}")

    for cfg in MODELS_REGISTRY:
        col = f"{cfg.short_name}_is_frustrating"
        if col in df.columns:
            n = int(df.loc[ok, col].sum())
            print(f"  [{cfg.short_name:>10}] frustrating : {n}/{int(ok.sum())} "
                  f"({n/max(int(ok.sum()),1)*100:.1f}%)")

    ens = int(df.loc[ok, "ensemble_is_frustrating"].sum())
    print(f"  [  ensemble] frustrating : {ens}/{int(ok.sum())} "
          f"({ens/max(int(ok.sum()),1)*100:.1f}%)")
    avg_t = df.loc[ok, "processing_time_s"].mean()
    print(f"  Avg time / file  : {avg_t:.2f}s")
    print(f"  Excel saved      : {excel_path}")
    print(f"  JSON saved       : {json_path}")
    print("=" * 64)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch-analyze call audio with 3 independent emotion models."
    )
    p.add_argument("--input-dir",  required=True, help="Folder containing audio files")
    p.add_argument("--output-dir", default="results", help="Output folder (created if absent)")
    p.add_argument("--excel",      default="frustration_detection_results.xlsx")
    p.add_argument("--workers",    type=int, default=1,
                   help="Parallel threads. Keep 1 on GPU.")
    p.add_argument("--recursive",  action="store_true",
                   help="Recurse into sub-directories")
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
