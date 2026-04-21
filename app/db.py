"""Optional InfluxDB v2 writer for per-window emotion time series."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import — only needed when InfluxDB is enabled.
try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
    _INFLUX_AVAILABLE = True
except ImportError:
    _INFLUX_AVAILABLE = False


def _is_enabled() -> bool:
    return os.getenv("ENABLE_INFLUX", "0").strip().lower() in ("1", "true", "yes")


def _get_client():
    url    = os.environ["INFLUX_URL"]
    token  = os.environ["INFLUX_TOKEN"]
    org    = os.environ["INFLUX_ORG"]
    return InfluxDBClient(url=url, token=token, org=org)


def write_call_to_influx(call_id: str, report: dict) -> None:
    """
    Write per-window emotion data + call summary to InfluxDB v2.

    Environment variables (all required when ENABLE_INFLUX=1):
        INFLUX_URL    – e.g. http://localhost:8086
        INFLUX_TOKEN  – operator/all-access token
        INFLUX_ORG    – organisation name
        INFLUX_BUCKET – destination bucket
    """
    if not _is_enabled():
        return
    if not _INFLUX_AVAILABLE:
        logger.warning("influxdb-client not installed; skipping InfluxDB write.")
        return

    bucket = os.environ.get("INFLUX_BUCKET", "emotions")
    timeline: list[dict] = report.get("timeline", [])
    if not timeline:
        return

    try:
        client = _get_client()
        write_api = client.write_api(write_options=SYNCHRONOUS)
        points: list[Any] = []

        # ── Per-window points ─────────────────────────────────────────────────
        for entry in timeline:
            emotion = entry.get("emotion", "unknown")
            start_s = entry.get("start", 0.0)
            confidence = entry.get("confidence", 0.0)

            p = (
                Point("emotion_window")
                .tag("call_id",  call_id)
                .tag("emotion",  emotion)
                .tag("severity", report.get("severity", ""))
                .tag("flagged",  str(report.get("flagged", False)).lower())
                .field("confidence",      confidence)
                .field("window_start_s",  float(start_s))
                .field("is_angry",        int(emotion == "angry"))
                .field("is_frustrated",   int(emotion == "frustrated"))
                # Use start_s as an offset from epoch in seconds → nanoseconds.
                # Real deployments should pass an absolute timestamp per call.
                .time(int(start_s * 1_000_000_000), WritePrecision.NANOSECONDS)
            )
            points.append(p)

        # ── Call-level summary point ──────────────────────────────────────────
        summary = (
            Point("call_summary")
            .tag("call_id",  call_id)
            .tag("severity", report.get("severity", ""))
            .tag("flagged",  str(report.get("flagged", False)).lower())
            .field("anger_ratio",       float(report.get("anger_ratio", 0.0)))
            .field("max_anger_score",   float(report.get("max_anger_score", 0.0)))
            .field("escalation",        int(report.get("escalation_detected", False)))
            .field("total_windows",     len(timeline))
            .time(0, WritePrecision.NANOSECONDS)
        )
        points.append(summary)

        write_api.write(bucket=bucket, record=points)
        logger.info("InfluxDB: wrote %d points for call %s", len(points), call_id)
        client.close()

    except Exception as exc:
        logger.error("InfluxDB write failed for %s: %s", call_id, exc)
