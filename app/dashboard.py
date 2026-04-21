"""Generate a self-contained HTML dashboard for batch results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Emotion colours (match visualizer.py) ────────────────────────────────────
_EMOTION_COLORS: dict[str, str] = {
    "angry":      "#E53935",
    "frustrated": "#FB8C00",
    "disgust":    "#8E24AA",
    "fear":       "#6D4C41",
    "sad":        "#1E88E5",
    "neutral":    "#757575",
    "happy":      "#43A047",
    "surprise":   "#00ACC1",
}
_DEFAULT_COLOR = "#9E9E9E"

_SEVERITY_BADGE: dict[str, str] = {
    "severe":   "danger",
    "moderate": "warning",
    "mild":     "success",
}


def generate_dashboard(
    calls: list[dict],
    output_dir: Path,
    filename: str = "dashboard.html",
) -> Path:
    """
    Build a self-contained HTML dashboard and write it to *output_dir*.

    ``calls`` is a list of dicts, each with keys:
        call_id, filename, report (the report.to_dict() dict),
        timeline (same as report["timeline"] but possibly enriched with
        ``segment_path`` keys by segment_extractor).

    Returns the path to the written HTML file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html = _render_html(calls)
    out = output_dir / filename
    out.write_text(html, encoding="utf-8")
    logger.info("Dashboard written → %s", out)
    return out


# ── HTML rendering ────────────────────────────────────────────────────────────

def _render_html(calls: list[dict]) -> str:
    total        = len(calls)
    n_flagged    = sum(1 for c in calls if c.get("report", {}).get("flagged"))
    n_escalated  = sum(1 for c in calls if c.get("report", {}).get("escalation_detected"))
    pct_flagged  = round(n_flagged / max(total, 1) * 100, 1)

    cards_html          = _summary_cards(total, n_flagged, n_escalated, pct_flagged)
    charts_html         = _call_panels(calls)
    calls_json          = json.dumps(calls, ensure_ascii=False)
    emotion_colors_json = json.dumps(_EMOTION_COLORS)
    default_color       = _DEFAULT_COLOR

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Patient Emotion Analysis Dashboard</title>
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
        crossorigin="anonymous"/>
  <script src="https://cdn.jsdelivr.net/npm/plotly.js-dist@2.30.0/plotly.min.js"></script>
  <style>
    body {{ background:#f4f6f9; font-family:'Segoe UI',sans-serif; padding-bottom:110px; }}
    .card-metric {{ border-left:4px solid; }}
    .card-metric.danger  {{ border-color:#E53935; }}
    .card-metric.warning {{ border-color:#FB8C00; }}
    .card-metric.success {{ border-color:#43A047; }}
    .card-metric.info    {{ border-color:#1E88E5; }}
    .call-card {{ background:#fff; border-radius:10px; box-shadow:0 1px 4px rgba(0,0,0,.1);
                  margin-bottom:18px; overflow:hidden; }}
    .call-header {{ padding:12px 18px; display:flex; align-items:center; gap:10px;
                    cursor:pointer; border-bottom:1px solid #e9ecef; }}
    .call-header:hover {{ background:#f8f9fa; }}
    .call-body {{ padding:14px 18px; display:none; }}
    .call-body.open {{ display:block; }}
    .severity-badge {{ font-size:.75rem; padding:3px 10px; border-radius:20px; font-weight:600; }}
    .plot-container {{ width:100%; height:320px; }}
    /* ── Sticky audio player ───────────────────────────────────────────────── */
    #audio-player-bar {{
      position:fixed; bottom:0; left:0; right:0; z-index:1050;
      background:#1a1a2e; color:#fff; padding:10px 20px;
      display:flex; align-items:center; gap:14px;
      box-shadow:0 -2px 12px rgba(0,0,0,.35);
      transition:transform .25s ease;
    }}
    #audio-player-bar.hidden {{ transform:translateY(100%); }}
    #audio-player-bar audio {{ flex:1; min-width:0; }}
    #player-label {{ font-size:.82rem; white-space:nowrap; overflow:hidden;
                     text-overflow:ellipsis; max-width:280px; }}
    #player-close {{ cursor:pointer; font-size:1.1rem; opacity:.7; }}
    #player-close:hover {{ opacity:1; }}
    .tab-btn {{ border-radius:20px; font-size:.82rem; }}
  </style>
</head>
<body>
<div class="container-fluid py-4">

  <!-- Header -->
  <div class="d-flex align-items-center mb-4 gap-3">
    <h4 class="mb-0 fw-bold">🏥 Patient Emotion Analysis</h4>
    <span class="badge bg-secondary">{total} calls</span>
  </div>

  <!-- Summary cards -->
  {cards_html}

  <!-- Filter tabs -->
  <div class="d-flex gap-2 mb-4">
    <button class="btn btn-sm btn-outline-secondary tab-btn active" data-filter="all">All</button>
    <button class="btn btn-sm btn-outline-danger  tab-btn" data-filter="flagged">Flagged</button>
    <button class="btn btn-sm btn-outline-success  tab-btn" data-filter="clean">Clean</button>
  </div>

  <!-- Call panels -->
  <div id="calls-container">
    {charts_html}
  </div>
</div>

<!-- Sticky audio player -->
<div id="audio-player-bar" class="hidden">
  <div>
    <div style="font-size:.65rem;opacity:.6;text-transform:uppercase;letter-spacing:.05em">Now playing</div>
    <div id="player-label">—</div>
  </div>
  <audio id="player-audio" controls preload="none" style="flex:1"></audio>
  <span id="player-close" title="Close">✕</span>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
        crossorigin="anonymous"></script>
<script>
// ── Embedded call data ────────────────────────────────────────────────────────
const CALLS = {calls_json};

// ── Accordion toggle ──────────────────────────────────────────────────────────
document.querySelectorAll('.call-header').forEach(hdr => {{
  hdr.addEventListener('click', () => {{
    const body = hdr.nextElementSibling;
    body.classList.toggle('open');
    const id = hdr.dataset.callId;
    if (body.classList.contains('open') && !hdr.dataset.plotted) {{
      hdr.dataset.plotted = '1';
      renderPlot(id);
    }}
  }});
}});

// ── Filter tabs ───────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const filter = btn.dataset.filter;
    document.querySelectorAll('.call-card').forEach(card => {{
      const show =
        filter === 'all' ||
        (filter === 'flagged' && card.dataset.flagged === 'true') ||
        (filter === 'clean'   && card.dataset.flagged === 'false');
      card.style.display = show ? '' : 'none';
    }});
  }});
}});

// ── Plotly chart per call ─────────────────────────────────────────────────────
const EMOTION_COLORS = {emotion_colors_json};
const DEFAULT_COLOR  = "{default_color}";
const THRESHOLD      = 0.45;

function renderPlot(callId) {{
  const callData = CALLS.find(c => c.call_id === callId);
  if (!callData) return;

  const timeline = callData.timeline || [];
  const mids       = timeline.map(e => (e.start + e.end) / 2);
  const emotions   = timeline.map(e => e.emotion);
  const confs      = timeline.map(e => e.confidence);
  const segPaths   = timeline.map(e => e.segment_path || null);

  // Score line
  const scoreLine = timeline.map((e, i) =>
    ['angry','frustrated'].includes(e.emotion) ? e.confidence : e.confidence * 0.4
  );

  const colors = emotions.map(em =>
    EMOTION_COLORS[em] || DEFAULT_COLOR
  );

  const markerSizes = emotions.map(em =>
    ['angry','frustrated'].includes(em) ? 11 : 6
  );

  const hoverText = timeline.map((e, i) => {{
    const seg = segPaths[i] ? '🔊 Click to play' : '';
    return `${{e.emotion}} (${{(e.confidence*100).toFixed(1)}}%)<br>`+
           `${{e.start.toFixed(1)}}s – ${{e.end.toFixed(1)}}s ${{seg}}`;
  }});

  const trace = {{
    x: mids,
    y: scoreLine,
    mode: 'lines+markers',
    type: 'scatter',
    marker: {{ color: colors, size: markerSizes, line: {{width:1, color:'#fff'}} }},
    line:   {{ color:'#455A64', width:2 }},
    text:   hoverText,
    hoverinfo: 'text',
    name: 'Emotion score',
    customdata: segPaths,
  }};

  const threshLine = {{
    x: [mids[0], mids[mids.length-1]],
    y: [THRESHOLD, THRESHOLD],
    mode: 'lines',
    type: 'scatter',
    line: {{ color:'#E53935', dash:'dash', width:1.5 }},
    name: `Threshold (${{THRESHOLD}})`,
    hoverinfo: 'skip',
  }};

  const report  = callData.report || {{}};
  const flagged = report.flagged;
  const title   = `${{callId}}  [${{flagged ? 'FLAGGED | ' + (report.severity||'') : 'OK'}}]  `+
                  `anger_ratio=${{(report.anger_ratio||0).toFixed(0)}}%  `+
                  `max_anger=${{(report.max_anger_score||0).toFixed(2)}}`;

  const layout = {{
    title: {{ text: title, font: {{ size:12, color: flagged ? '#E53935' : '#43A047' }} }},
    xaxis: {{ title:'Time (s)', gridcolor:'#eee' }},
    yaxis: {{ title:'Score', range:[0,1.05], tickformat:'.0%', gridcolor:'#eee' }},
    margin: {{ t:50, l:55, r:20, b:45 }},
    plot_bgcolor:'#fff',
    paper_bgcolor:'#fff',
    showlegend: true,
    legend: {{ font:{{size:10}}, x:0, y:1.15, orientation:'h' }},
  }};

  const config = {{ responsive:true, displayModeBar:false }};
  const plotDiv = document.getElementById('plot-' + callId);
  Plotly.newPlot(plotDiv, [trace, threshLine], layout, config);

  // Click on flagged point → play audio
  plotDiv.on('plotly_click', data => {{
    const pt = data.points[0];
    if (pt.curveNumber !== 0) return;           // only the score trace
    const segPath = pt.customdata;
    if (!segPath) return;
    const em      = emotions[pt.pointIndex];
    if (!['angry','frustrated'].includes(em)) return;
    playSegment(segPath, callId, timeline[pt.pointIndex]);
  }});
}}

// ── Audio player ──────────────────────────────────────────────────────────────
const playerBar   = document.getElementById('audio-player-bar');
const playerAudio = document.getElementById('player-audio');
const playerLabel = document.getElementById('player-label');

function playSegment(relPath, callId, entry) {{
  playerAudio.src = relPath;
  playerLabel.textContent =
    `${{callId}} · ${{entry.emotion}} · ${{entry.start.toFixed(1)}}s–${{entry.end.toFixed(1)}}s`;
  playerBar.classList.remove('hidden');
  playerAudio.load();
  playerAudio.play().catch(() => {{}});   // user gesture may be needed in some browsers
}}

document.getElementById('player-close').addEventListener('click', () => {{
  playerAudio.pause();
  playerBar.classList.add('hidden');
}});
</script>
</body>
</html>"""


# ── Helper: summary cards ─────────────────────────────────────────────────────

def _summary_cards(total: int, n_flagged: int, n_escalated: int, pct: float) -> str:
    return f"""
<div class="row g-3 mb-4">
  <div class="col-sm-6 col-lg-3">
    <div class="card card-metric info p-3">
      <div class="text-muted small">Total calls</div>
      <div class="fs-3 fw-bold">{total}</div>
    </div>
  </div>
  <div class="col-sm-6 col-lg-3">
    <div class="card card-metric danger p-3">
      <div class="text-muted small">Flagged calls</div>
      <div class="fs-3 fw-bold text-danger">{n_flagged}</div>
    </div>
  </div>
  <div class="col-sm-6 col-lg-3">
    <div class="card card-metric warning p-3">
      <div class="text-muted small">% Flagged</div>
      <div class="fs-3 fw-bold text-warning">{pct}%</div>
    </div>
  </div>
  <div class="col-sm-6 col-lg-3">
    <div class="card card-metric {'danger' if n_escalated else 'success'} p-3">
      <div class="text-muted small">Escalations</div>
      <div class="fs-3 fw-bold {'text-danger' if n_escalated else 'text-success'}">{n_escalated}</div>
    </div>
  </div>
</div>"""


# ── Helper: one panel per call ────────────────────────────────────────────────

def _call_panels(calls: list[dict]) -> str:
    panels: list[str] = []
    for c in calls:
        call_id = c.get("call_id", "unknown")
        report  = c.get("report", {})
        flagged = report.get("flagged", False)
        severity = report.get("severity", "")
        badge_cls = _SEVERITY_BADGE.get(severity, "secondary")
        flag_label = "FLAGGED" if flagged else "OK"
        flag_color = "text-danger" if flagged else "text-success"
        escalation_html = (
            '<span class="badge bg-warning text-dark ms-2">⚠ Escalation</span>'
            if report.get("escalation_detected") else ""
        )

        panels.append(f"""
<div class="call-card" data-flagged="{str(flagged).lower()}">
  <div class="call-header" data-call-id="{call_id}">
    <span class="fw-semibold">{call_id}</span>
    <span class="badge bg-{badge_cls} severity-badge">{severity}</span>
    <span class="ms-1 small {flag_color}">{flag_label}</span>
    {escalation_html}
    <span class="ms-auto text-muted small">
      anger_ratio={report.get('anger_ratio', 0):.0%} &nbsp;|&nbsp;
      max_anger={report.get('max_anger_score', 0):.2f}
    </span>
  </div>
  <div class="call-body">
    <p class="text-muted small mb-2">{report.get('reason','')}</p>
    <div id="plot-{call_id}" class="plot-container"></div>
  </div>
</div>""")

    return "\n".join(panels)
