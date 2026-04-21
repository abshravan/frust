"""Per-call timeline chart generator."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Emotion colour palette ────────────────────────────────────────────────────
_EMOTION_COLORS: dict[str, str] = {
    "angry":      "#E53935",   # red
    "frustrated": "#FB8C00",   # orange
    "disgust":    "#8E24AA",   # purple
    "fear":       "#6D4C41",   # brown
    "sad":        "#1E88E5",   # blue
    "neutral":    "#757575",   # grey
    "happy":      "#43A047",   # green
    "surprise":   "#00ACC1",   # teal
}
_DEFAULT_COLOR = "#9E9E9E"

_THRESHOLD_COLOR = "#E53935"
_ANGRY_SHADE     = "#FFCDD2"   # light red fill under anger spikes
_FRUSTRATED_SHADE = "#FFE0B2"  # light orange fill


def plot_call(
    report: dict,
    call_id: str,
    output_path: Path,
    threshold: float = 0.60,
) -> None:
    """
    Generate a two-panel PNG chart for a single call:

    Top panel  – anger score line + shaded problem zones + threshold line.
    Bottom panel – per-window dominant-emotion colour strip.

    Args:
        report:      The dict returned by AnalysisReport.to_dict().
        call_id:     Used as the figure title and filename stem.
        output_path: Directory where the PNG is written.
        threshold:   Anger threshold line drawn on the score panel.
    """
    timeline = report.get("timeline", [])
    if not timeline:
        return

    starts      = np.array([e["start"]      for e in timeline])
    ends        = np.array([e["end"]        for e in timeline])
    emotions    = [e["emotion"]              for e in timeline]
    confidences = np.array([e["confidence"] for e in timeline])

    # Mid-point of each window for the x-axis.
    mids = (starts + ends) / 2

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, (ax_score, ax_strip) = plt.subplots(
        2, 1,
        figsize=(max(10, len(timeline) * 0.45), 6),
        gridspec_kw={"height_ratios": [5, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.08)

    # ── Top panel: anger score + problem zones ────────────────────────────────
    angry_scores = np.array([
        c if e == "angry" else 0.0
        for e, c in zip(emotions, confidences)
    ])
    frustrated_scores = np.array([
        c if e == "frustrated" else 0.0
        for e, c in zip(emotions, confidences)
    ])
    # Use confidence score as the y-value for the dominant emotion's relevance.
    anger_line = np.array([
        c if e in ("angry", "frustrated") else c * 0.4
        for e, c in zip(emotions, confidences)
    ])

    # Shade angry and frustrated regions.
    ax_score.fill_between(mids, angry_scores, alpha=0.25, color=_ANGRY_SHADE,
                          label="_nolegend_")
    ax_score.fill_between(mids, frustrated_scores, alpha=0.25,
                          color=_FRUSTRATED_SHADE, label="_nolegend_")

    # Main score line.
    ax_score.plot(mids, anger_line, color="#455A64", linewidth=1.6,
                  marker="o", markersize=4, zorder=3, label="Emotion score")

    # Highlight flagged windows (angry / frustrated) with larger markers.
    for i, (mid, emo, conf) in enumerate(zip(mids, emotions, confidences)):
        if emo in ("angry", "frustrated"):
            color = _EMOTION_COLORS.get(emo, _DEFAULT_COLOR)
            ax_score.scatter(mid, conf, color=color, s=70, zorder=5)

    # Threshold line.
    ax_score.axhline(threshold, color=_THRESHOLD_COLOR, linewidth=1.2,
                     linestyle="--", label=f"Threshold ({threshold})")

    # Escalation annotation.
    if report.get("escalation_detected"):
        ax_score.annotate(
            "⚠ Escalation detected",
            xy=(mids[-1], threshold),
            xytext=(-10, 10), textcoords="offset points",
            fontsize=8, color=_THRESHOLD_COLOR,
            arrowprops=dict(arrowstyle="->", color=_THRESHOLD_COLOR),
        )

    ax_score.set_ylim(0, 1.05)
    ax_score.set_ylabel("Score", fontsize=9)
    ax_score.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax_score.legend(fontsize=8, loc="upper left")
    ax_score.grid(axis="y", linestyle=":", alpha=0.5)
    ax_score.spines[["top", "right"]].set_visible(False)

    # Title block.
    flagged   = report.get("flagged", False)
    severity  = report.get("severity", "")
    flag_str  = f"FLAGGED  |  severity: {severity}" if flagged else "OK"
    flag_color = _THRESHOLD_COLOR if flagged else "#43A047"
    ax_score.set_title(
        f"{call_id}   [{flag_str}]   "
        f"anger_ratio={report.get('anger_ratio', 0):.0%}  "
        f"max_anger={report.get('max_anger_score', 0):.2f}",
        fontsize=10, color=flag_color, pad=8,
    )

    # ── Bottom panel: emotion colour strip ────────────────────────────────────
    for i, (start, end, emo) in enumerate(zip(starts, ends, emotions)):
        color = _EMOTION_COLORS.get(emo, _DEFAULT_COLOR)
        ax_strip.barh(
            0, end - start, left=start, height=1,
            color=color, edgecolor="white", linewidth=0.4,
        )
        # Label short windows without text, wider ones with abbreviation.
        if (end - start) >= 2.5:
            ax_strip.text(
                (start + end) / 2, 0, emo[:3].upper(),
                ha="center", va="center", fontsize=6,
                color="white", fontweight="bold",
            )

    ax_strip.set_yticks([])
    ax_strip.set_xlabel("Time (seconds)", fontsize=9)
    ax_strip.set_xlim(mids[0] - (mids[1] - mids[0]) / 2 if len(mids) > 1 else 0,
                      ends[-1])
    ax_strip.spines[["top", "right", "left"]].set_visible(False)

    # ── Legend for emotion colours ────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=col, label=emo.capitalize())
        for emo, col in _EMOTION_COLORS.items()
        if emo in set(emotions)
    ]
    if legend_patches:
        ax_strip.legend(
            handles=legend_patches, fontsize=7,
            loc="lower right", ncol=len(legend_patches),
            bbox_to_anchor=(1, -0.6), frameon=False,
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_file = Path(output_path) / f"{call_id}_timeline.png"
    fig.savefig(out_file, dpi=130, bbox_inches="tight")
    plt.close(fig)
