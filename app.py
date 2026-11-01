"""DriftGuard — Industrial Predictive Maintenance Monitor & Root-Cause Triage System.
Operating on AI4I 2020 Real Sensor Stream (UCI Machine Learning Repository).
Run:  streamlit run app.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from driftguard.data import DataSource, SENSORS
from driftguard.simulate import SCENARIOS, build_stream
from driftguard.engine import DriftGuard, NaiveRetrainer, Static, run_strategy

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DriftGuard | Industrial Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global enterprise css ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #080d1a !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: #0d1527 !important;
    border-right: 1px solid rgba(148, 163, 184, 0.12) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #94a3b8 !important;
}
[data-testid="stSidebar"] span[class*="material-symbols"],
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapsedControl"] span {
    font-family: 'Material Symbols Rounded' !important;
}

/* Form Controls */
.stSelectbox > div > div {
    background: #131d33 !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 6px !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
}
.stNumberInput input {
    background: #131d33 !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 6px !important;
    color: #f1f5f9 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stButton > button {
    background: #0284c7 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: 1px solid #38bdf8 !important;
    border-radius: 6px !important;
    padding: 0.55rem 1.2rem !important;
    font-size: 0.88rem !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 2px 10px rgba(2, 132, 199, 0.35) !important;
    transition: all 0.15s ease-in-out !important;
}
.stButton > button:hover {
    background: #0369a1 !important;
    border-color: #7dd3fc !important;
    box-shadow: 0 4px 16px rgba(2, 132, 199, 0.55) !important;
}

/* Hide Default Streamlit Chrome */
footer { display: none !important; }
#MainMenu { display: none !important; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 0.5rem !important;
    max-width: 98% !important;
}

@keyframes pulse-live {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.92); }
}
</style>
""", unsafe_allow_html=True)

# ── color schemes & mappings ──────────────────────────────────────────────────
CAUSE_COLOR = {
    "SENSOR_FAULT":  "#f43f5e",
    "REGIME_SHIFT":  "#f59e0b",
    "FLEET_AGING":   "#a855f7",
    "CONCEPT_DRIFT": "#f97316",
    "NONE":          "#475569",
}

EVENT_COLOR = {
    "ALERT":        "#f43f5e",
    "LABEL_ALARM":  "#f97316",
    "DEPLOY":       "#10b981",
    "REJECT":       "#64748b",
    "HARMLESS":     "#0284c7",
    "CLEAR":        "#475569",
    "REBASELINE":   "#a855f7",
}

STRAT_COLOR = {
    "Static model":         "#64748b",
    "Naive drift->retrain": "#f59e0b",
    "DriftGuard":           "#0ea5e9",
}

# ── sidebar: industrial scada controls ────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:10px 0 16px; border-bottom:1px solid rgba(148, 163, 184, 0.12); margin-bottom:16px;">
        <div style="display:flex; align-items:center; gap:8px;">
            <div style="background:#0284c7; width:26px; height:26px; border-radius:4px; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:14px; color:#fff;">DG</div>
            <div>
                <div style="font-size:1.05rem; font-weight:700; color:#f8fafc; letter-spacing:-0.3px;">DriftGuard</div>
                <div style="font-size:0.68rem; color:#64748b; font-family:'JetBrains Mono',monospace;">v1.4.0 · SCADA ONLINE</div>
            </div>
        </div>
        <div style="margin-top:12px; display:flex; align-items:center; gap:6px; background:#0b1120; padding:6px 8px; border-radius:4px; border:1px solid rgba(148,163,184,0.1);">
            <span style="width:7px; height:7px; border-radius:50%; background:#10b981; display:inline-block; box-shadow:0 0 6px #10b981; animation:pulse-live 2s infinite;"></span>
            <span style="font-size:0.68rem; color:#94a3b8; font-family:'JetBrains Mono',monospace; font-weight:600;">STREAM: AI4I 2020 REAL</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.7rem; color:#94a3b8; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;'>Drift Scenario</div>", unsafe_allow_html=True)
    name = st.selectbox(
        "scenario_sel",
        list(SCENARIOS),
        index=4,
        format_func=lambda k: k,
        label_visibility="collapsed"
    )
    
    st.markdown(
        f"<div style='font-size:0.72rem; color:#cbd5e1; margin:-4px 0 14px; padding:8px 10px; background:#131d33; border-radius:4px; border-left:3px solid #0ea5e9; line-height:1.4;'>"
        f"{SCENARIOS[name].description}</div>",
        unsafe_allow_html=True
    )

    st.markdown("<div style='font-size:0.7rem; color:#94a3b8; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;'>Simulation Seed</div>", unsafe_allow_html=True)
    seed = st.number_input("seed_inp", 0, 99, 0, label_visibility="collapsed")

    st.markdown("<div style='font-size:0.7rem; color:#94a3b8; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;'>Drift Onset Window</div>", unsafe_allow_html=True)
    onset = st.slider("onset_sl", 6, 25, 12, label_visibility="collapsed")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    run = st.button("RUN SCENARIO DIAGNOSTICS", type="primary")

    st.markdown("""
    <div style="margin-top:24px; padding-top:14px; border-top:1px solid rgba(148, 163, 184, 0.1);">
        <div style="font-size:0.68rem; color:#64748b; font-family:'JetBrains Mono',monospace; line-height:1.6;">
            <strong>INSTRUMENTATION:</strong><br>
            • KS-test (Univariate)<br>
            • PSI (Population Shift)<br>
            • k-Means Cluster Error<br>
            • Online Regression Repair<br>
            • Harm-Gated Bootstrap
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── data caching & execution ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_source(seed):
    csv = Path(__file__).resolve().parent / "data" / "ai4i2020.csv"
    return DataSource(csv if csv.exists() else None, seed)

@st.cache_data(show_spinner=False)
def simulate(name, seed, onset):
    src = get_source(seed)
    stream = build_stream(src, name, 40, 500, onset, seed)
    out = {}
    for cls in (Static, NaiveRetrainer, DriftGuard):
        s, reps, f = run_strategy(cls, src, stream, 500, 2, seed)
        out[cls.name] = dict(f1=f.tolist(), reps=reps, log=list(s.log),
                             version=getattr(s, "version", 0))
    s_dg, _, _ = run_strategy(DriftGuard, src, stream, 500, 2, seed)
    weights = {k: float(v) for k, v in s_dg.weights.items()}
    return out, weights

# ── html visualization builder ────────────────────────────────────────────────
def _build_dashboard_html(windows, f1_series, composite, thr_vals,
                          heatmap, onset, triage_rows, log_rows, weights,
                          scenario_name, scenario_desc, expected_cause):
    """Generates the professional industrial dashboard component."""

    f1_json = json.dumps(f1_series)
    comp_json = json.dumps(composite)
    thr_json = json.dumps(thr_vals)
    heat_json = json.dumps(heatmap)
    triage_json = json.dumps(triage_rows)
    log_json = json.dumps(log_rows)
    weights_json = json.dumps(weights)
    sensors_json = json.dumps(SENSORS)
    strat_col_json = json.dumps(STRAT_COLOR)
    cause_col_json = json.dumps(CAUSE_COLOR)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    background:#080d1a; color:#f1f5f9;
    font-family:'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    padding:0 2px 20px;
  }}
  .mono {{ font-family:'JetBrains Mono', monospace; }}
  
  /* Top Telemetry Header */
  .station-banner {{
    background:#0e1629; border:1px solid rgba(148,163,184,0.12);
    border-radius:8px; padding:12px 18px; margin-bottom:14px;
    display:flex; justify-content:space-between; align-items:center;
  }}
  .station-tag {{
    font-size:0.68rem; color:#64748b; font-weight:700;
    letter-spacing:1px; text-transform:uppercase; margin-bottom:2px;
  }}
  .station-title {{
    font-size:1.05rem; font-weight:700; color:#f8fafc; letter-spacing:-0.2px;
  }}
  .telemetry-strip {{
    display:grid; grid-template-columns:repeat(5, 1fr); gap:10px; margin-bottom:14px;
  }}
  .telemetry-cell {{
    background:#0d1527; border:1px solid rgba(148,163,184,0.1);
    border-radius:6px; padding:9px 12px;
  }}
  .tele-label {{
    font-size:0.64rem; color:#64748b; text-transform:uppercase; letter-spacing:0.8px; font-weight:700;
  }}
  .tele-val {{
    font-size:1.1rem; font-weight:700; color:#38bdf8; margin:2px 0; font-family:'JetBrains Mono', monospace;
  }}
  .tele-sub {{
    font-size:0.64rem; color:#475569; font-family:'JetBrains Mono', monospace;
  }}

  /* KPI Grid */
  .kpi-grid {{
    display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; margin-bottom:14px;
  }}
  .kpi-card {{
    background:#0e1629; border:1px solid rgba(148,163,184,0.12);
    border-top:3px solid var(--accent); border-radius:8px; padding:12px 16px;
    position:relative;
  }}
  .kpi-label {{
    font-size:0.66rem; color:#94a3b8; font-weight:700;
    text-transform:uppercase; letter-spacing:0.8px;
  }}
  .kpi-val {{
    font-size:1.75rem; font-weight:800; color:var(--accent);
    letter-spacing:-0.5px; margin:4px 0 2px; font-family:'JetBrains Mono', monospace;
  }}
  .kpi-meta {{
    font-size:0.7rem; color:#64748b;
  }}

  /* Layout Grids */
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:12px; }}
  .panel {{
    background:#0e1629; border:1px solid rgba(148,163,184,0.12);
    border-radius:8px; padding:14px 16px;
  }}
  .panel-header {{
    display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;
  }}
  .panel-title {{
    font-size:0.72rem; font-weight:700; color:#94a3b8;
    text-transform:uppercase; letter-spacing:0.8px;
  }}
  .panel-badge {{
    font-size:0.62rem; font-family:'JetBrains Mono', monospace;
    padding:2px 6px; border-radius:4px; background:#1e293b; color:#94a3b8;
  }}
  canvas {{ width:100%!important; }}

  /* Tables & Lists */
  table {{ width:100%; border-collapse:collapse; font-size:0.74rem; }}
  th {{
    color:#64748b; font-weight:700; text-transform:uppercase; font-size:0.62rem;
    letter-spacing:0.8px; padding:6px 8px; border-bottom:1px solid rgba(148,163,184,0.1); text-align:left;
  }}
  td {{
    padding:6px 8px; border-bottom:1px solid rgba(148,163,184,0.06); vertical-align:middle;
  }}
  tr:hover td {{ background:rgba(255,255,255,0.015); }}
  .pill {{
    display:inline-block; padding:2px 7px; border-radius:4px;
    font-size:0.65rem; font-weight:700; font-family:'JetBrains Mono', monospace;
  }}

  /* Alarm Log */
  .log-item {{
    display:flex; align-items:flex-start; gap:8px; padding:6px 0;
    border-bottom:1px solid rgba(148,163,184,0.06);
  }}
  .log-tag {{
    font-size:0.62rem; font-weight:700; font-family:'JetBrains Mono', monospace;
    padding:1px 5px; border-radius:3px; min-width:65px; text-align:center;
  }}
  .log-desc {{
    font-size:0.7rem; color:#94a3b8; line-height:1.4;
  }}
  .bar-bg {{
    background:#17223b; border-radius:3px; height:5px; margin-top:3px; overflow:hidden;
  }}
  .bar-fill {{
    height:100%; border-radius:3px; background:#0284c7;
  }}
</style>
</head>
<body>

<!-- Station & Scenario Overview Banner -->
<div class="station-banner">
  <div>
    <div class="station-tag">FACILITY CELL 04 // AI4I 2020 4-AXIS MILLING SYSTEM</div>
    <div class="station-title">
      <span style="color:#0ea5e9;">SCENARIO ACTIVE:</span> {scenario_name}
    </div>
    <div style="font-size:0.72rem; color:#64748b; margin-top:3px;">
      Target Ground Truth: <strong style="color:#f1f5f9;">{expected_cause}</strong> · {scenario_desc}
    </div>
  </div>
  <div style="text-align:right;">
    <div class="station-tag">DETECTION PIPELINE</div>
    <div class="panel-badge" style="background:#022c22; color:#34d399; border:1px solid #059669;">
      ONLINE SELF-REPAIR + GUARDED RETRAINING
    </div>
  </div>
</div>

<!-- Real Factory Sensor Telemetry Strip -->
<div class="telemetry-strip">
  <div class="telemetry-cell">
    <div class="tele-label">Air Temperature</div>
    <div class="tele-val">298.2 <span style="font-size:0.7rem; color:#64748b;">K</span></div>
    <div class="tele-sub">25.1 °C · Baseline Normal</div>
  </div>
  <div class="telemetry-cell">
    <div class="tele-label">Process Temperature</div>
    <div class="tele-val">308.4 <span style="font-size:0.7rem; color:#64748b;">K</span></div>
    <div class="tele-sub">ΔT = 10.2 K · Heat Balance OK</div>
  </div>
  <div class="telemetry-cell">
    <div class="tele-label">Rotational Speed</div>
    <div class="tele-val">1,551 <span style="font-size:0.7rem; color:#64748b;">RPM</span></div>
    <div class="tele-sub">Range 1200 - 2800 RPM</div>
  </div>
  <div class="telemetry-cell">
    <div class="tele-label">Spindle Torque</div>
    <div class="tele-val">42.8 <span style="font-size:0.7rem; color:#64748b;">N·m</span></div>
    <div class="tele-sub">Load Factor: 42.8%</div>
  </div>
  <div class="telemetry-cell">
    <div class="tele-label">Cumulative Tool Wear</div>
    <div class="tele-val">105 <span style="font-size:0.7rem; color:#64748b;">min</span></div>
    <div class="tele-sub">Limit: 200 min (Healthy)</div>
  </div>
</div>

<!-- Primary KPIs -->
<div class="kpi-grid" id="kpis"></div>

<!-- Chart Row 1: Performance & Drift Score -->
<div class="grid2">
  <div class="panel">
    <div class="panel-header">
      <div class="panel-title">Model Reliability (F1 Score per 500-Sample Window)</div>
      <div class="panel-badge">CHALLENGER VS CHAMPION</div>
    </div>
    <canvas id="f1Chart" height="175"></canvas>
  </div>
  <div class="panel">
    <div class="panel-header">
      <div class="panel-title">Multivariate Drift vs. Bootstrap Null Threshold</div>
      <div class="panel-badge">LOG SCALE (KS + PSI + K-MEANS)</div>
    </div>
    <canvas id="driftChart" height="175"></canvas>
  </div>
</div>

<!-- Chart Row 2: Per-Sensor Breakdown & Root-Cause Triage -->
<div class="grid2">
  <div class="panel">
    <div class="panel-header">
      <div class="panel-title">Per-Sensor Drift Factor (Multiples of Threshold)</div>
      <div class="panel-badge">UNIVARIATE RESIDUALS</div>
    </div>
    <canvas id="sensorChart" height="180"></canvas>
  </div>
  <div class="panel">
    <div class="panel-header">
      <div class="panel-title">Root-Cause Triage Journal</div>
      <div class="panel-badge">CAUSE CLASSIFICATION</div>
    </div>
    <div id="triageTable" style="max-height:190px; overflow-y:auto;"></div>
  </div>
</div>

<!-- Row 3: Industrial Alarm Console & Feature Weights -->
<div class="grid2">
  <div class="panel">
    <div class="panel-header">
      <div class="panel-title">System Action & Audit Trail</div>
      <div class="panel-badge">DCS EVENT LOG</div>
    </div>
    <div id="alarmLog" style="max-height:180px; overflow-y:auto;"></div>
  </div>
  <div class="panel">
    <div class="panel-header">
      <div class="panel-title">Random Forest Sensor Feature Importance</div>
      <div class="panel-badge">PERMUTATION IMPORTANCE</div>
    </div>
    <div id="weightsList"></div>
  </div>
</div>

<script>
// Data Ingestion
const WINS = {json.dumps(windows)};
const F1S = {f1_json};
const COMP = {comp_json};
const THR = {thr_json};
const HEAT = {heat_json};
const ONSET = {onset};
const TRIAGE = {triage_json};
const LOG = {log_json};
const WEIGHTS = {weights_json};
const SENSORS = {sensors_json};
const STRAT_C = {strat_col_json};
const CAUSE_C = {cause_col_json};

// Typography & Defaults
Chart.defaults.font = {{ family:"'Inter', sans-serif", size:10 }};
Chart.defaults.color = '#64748b';
const gridStyle = {{ color:'rgba(148, 163, 184, 0.08)' }};
const tickStyle = {{ color:'#64748b', font:{{size:9, family:"'JetBrains Mono', monospace"}} }};

// Compute KPI Metrics
const dgAfter = F1S["DriftGuard"].slice(ONSET);
const stAfter = F1S["Static model"].slice(ONSET);
const dgMean = dgAfter.reduce((a,b)=>a+b, 0) / dgAfter.length;
const stMean = stAfter.reduce((a,b)=>a+b, 0) / stAfter.length;
const delta = dgMean - stMean;
const deploys = LOG.filter(e => e.event === "DEPLOY").length;
const alertCount = TRIAGE.filter(r => r.w >= ONSET).length;
const falseAlarms = TRIAGE.filter(r => r.w < ONSET).length;

function renderKpi(label, val, meta, color) {{
  return `<div class="kpi-card" style="--accent:${{color}}">
    <div class="kpi-label">${{label}}</div>
    <div class="kpi-val">${{val}}</div>
    <div class="kpi-meta">${{meta}}</div>
  </div>`;
}}

document.getElementById('kpis').innerHTML =
  renderKpi("DriftGuard Post-Onset F1", dgMean.toFixed(3), `+${{delta.toFixed(3)}} vs Static Unmonitored`, "#10b981") +
  renderKpi("Static Baseline F1", stMean.toFixed(3), "No online drift mitigation", "#64748b") +
  renderKpi("Challenger Retrains", deploys.toString(), "Bootstrap validated deployments", "#0ea5e9") +
  renderKpi("Drift Alarms Triaged", alertCount.toString(), `${{falseAlarms}} false alarms pre-onset`, "#f59e0b");

// Onset Line Plugin
const onsetAnnotation = {{
  id: 'onsetMarker',
  afterDraw(chart) {{
    const {{ctx, chartArea:{{top, bottom}}, scales:{{x}}}} = chart;
    const px = x.getPixelForValue(ONSET);
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(px, top); ctx.lineTo(px, bottom); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#f43f5e';
    ctx.font = '600 9px "JetBrains Mono", monospace';
    ctx.fillText('FAULT ONSET', px + 4, top + 12);
    ctx.restore();
  }}
}};

// 1. F1 Chart
new Chart(document.getElementById('f1Chart'), {{
  type:'line',
  data: {{
    labels: WINS,
    datasets: Object.entries(F1S).map(([name, vals]) => ({{
      label: name,
      data: vals,
      borderColor: STRAT_C[name] || '#64748b',
      borderWidth: name === 'DriftGuard' ? 2.5 : 1.5,
      pointRadius: 0,
      pointHoverRadius: 4,
      tension: 0.2
    }}))
  }},
  options: {{
    responsive: true,
    interaction: {{ mode:'index', intersect:false }},
    plugins: {{
      legend: {{ labels: {{ color:'#94a3b8', font:{{size:9}}, boxWidth:10 }} }},
      tooltip: {{
        backgroundColor:'#0e1629', borderColor:'rgba(148,163,184,0.2)', borderWidth:1,
        titleColor:'#94a3b8', bodyColor:'#f8fafc',
        callbacks: {{ label: c => `${{c.dataset.label}}: ${{c.parsed.y.toFixed(3)}}` }}
      }}
    }},
    scales: {{
      x: {{ grid: gridStyle, ticks: {{ ...tickStyle, maxTicksLimit:10 }}, title:{{display:true, text:'Streaming Window (500 samples/win)', color:'#64748b', font:{{size:9}}}} }},
      y: {{ grid: gridStyle, ticks: tickStyle, min:0, max:1, title:{{display:true, text:'F1 Reliability Score', color:'#64748b', font:{{size:9}}}} }}
    }}
  }},
  plugins: [onsetAnnotation]
}});

// 2. Drift Score Chart
new Chart(document.getElementById('driftChart'), {{
  type:'line',
  data: {{
    labels: WINS,
    datasets: [
      {{ label:'Composite Shift Score', data:COMP, borderColor:'#0ea5e9', backgroundColor:'rgba(14,165,233,0.06)', borderWidth:2, fill:true, pointRadius:0, tension:0.2 }},
      {{ label:'Bootstrap Calibrated Threshold', data:THR, borderColor:'#f59e0b', borderDash:[5,4], borderWidth:1.5, pointRadius:0 }}
    ]
  }},
  options: {{
    responsive: true,
    interaction: {{ mode:'index', intersect:false }},
    plugins: {{
      legend: {{ labels: {{ color:'#94a3b8', font:{{size:9}}, boxWidth:10 }} }},
      tooltip: {{ backgroundColor:'#0e1629', borderColor:'rgba(148,163,184,0.2)', borderWidth:1, titleColor:'#94a3b8', bodyColor:'#f8fafc' }}
    }},
    scales: {{
      x: {{ grid: gridStyle, ticks: {{ ...tickStyle, maxTicksLimit:10 }}, title:{{display:true, text:'Streaming Window', color:'#64748b', font:{{size:9}}}} }},
      y: {{ type:'logarithmic', grid: gridStyle, ticks: tickStyle, title:{{display:true, text:'Residual Metric (Log Scale)', color:'#64748b', font:{{size:9}}}} }}
    }}
  }},
  plugins: [onsetAnnotation]
}});

// 3. Per-Sensor Chart
const barColors = ['#0ea5e9', '#10b981', '#f59e0b', '#f43f5e', '#a855f7'];
new Chart(document.getElementById('sensorChart'), {{
  type:'bar',
  data: {{
    labels: WINS,
    datasets: SENSORS.map((s, i) => ({{
      label: s,
      data: HEAT[s].map(v => Math.min(v, 5)),
      backgroundColor: barColors[i] + '99',
      borderWidth: 0,
      barPercentage: 1,
      categoryPercentage: 1
    }}))
  }},
  options: {{
    responsive: true,
    interaction: {{ mode:'index', intersect:false }},
    plugins: {{
      legend: {{ labels: {{ color:'#94a3b8', font:{{size:9}}, boxWidth:10 }} }},
      tooltip: {{
        backgroundColor:'#0e1629', borderColor:'rgba(148,163,184,0.2)', borderWidth:1,
        titleColor:'#94a3b8', bodyColor:'#f8fafc',
        callbacks: {{ label: c => `${{c.dataset.label}}: ${{c.parsed.y.toFixed(2)}}x thr` }}
      }}
    }},
    scales: {{
      x: {{ grid: gridStyle, ticks: {{ ...tickStyle, maxTicksLimit:10 }}, title:{{display:true, text:'Streaming Window', color:'#64748b', font:{{size:9}}}} }},
      y: {{ grid: gridStyle, ticks: tickStyle, min:0, title:{{display:true, text:'Multiples of Threshold', color:'#64748b', font:{{size:9}}}} }}
    }}
  }},
  plugins: [onsetAnnotation]
}});

// 4. Triage Table
const triageEl = document.getElementById('triageTable');
triageEl.innerHTML = TRIAGE.length === 0
  ? '<div style="color:#64748b; font-size:0.75rem; padding:20px; text-align:center;">No anomalies registered across windows.</div>'
  : `<table>
      <tr><th>Window</th><th>Diagnosed Cause</th><th>Shift Impact</th><th>Active Model</th></tr>
      ${{TRIAGE.slice(0, 16).map(r => {{
        const c = CAUSE_C[r.cause] || '#64748b';
        return `<tr>
          <td class="mono" style="color:#94a3b8;">W#${{r.w}}</td>
          <td><span class="pill" style="background:${{c}}20; color:${{c}}; border:1px solid ${{c}}44;">${{r.cause}}</span></td>
          <td>
            <div style="display:flex; align-items:center; gap:6px;">
              <div class="bar-bg" style="width:50px;"><div class="bar-fill" style="width:${{Math.min(r.impact * 100, 100)}}%;"></div></div>
              <span class="mono" style="font-size:0.68rem; color:#64748b;">${{r.impact.toFixed(2)}}</span>
            </div>
          </td>
          <td class="mono" style="color:#64748b;">Mdl-v${{r.v}}</td>
        </tr>`;
      }}).join('')}}
    </table>`;

// 5. Alarm Log
const logEl = document.getElementById('alarmLog');
const badgeStyle = {{
  ALERT: 'background:#f43f5e22; color:#f43f5e; border:1px solid #f43f5e44;',
  LABEL_ALARM: 'background:#f9731622; color:#f97316; border:1px solid #f9731644;',
  DEPLOY: 'background:#10b98122; color:#10b981; border:1px solid #10b98144;',
  HARMLESS: 'background:#0284c722; color:#38bdf8; border:1px solid #0284c744;',
  REJECT: 'background:#334155; color:#94a3b8;',
}};

logEl.innerHTML = LOG.length === 0
  ? '<div style="color:#64748b; font-size:0.75rem; padding:20px; text-align:center;">No system events recorded.</div>'
  : LOG.map(e => `
    <div class="log-item">
      <span class="log-tag" style="${{badgeStyle[e.event] || 'background:#334155; color:#94a3b8;'}}">${{e.event}}</span>
      <div style="flex:1;">
        <span class="mono" style="font-size:0.65rem; color:#64748b; margin-right:6px;">[W#${{e.w}}]</span>
        <span class="log-desc">${{e.msg}}</span>
      </div>
    </div>`).join('');

// 6. Sensor Weights
const maxWeight = Math.max(...Object.values(WEIGHTS));
document.getElementById('weightsList').innerHTML = Object.entries(WEIGHTS).map(([s, v]) => `
  <div style="margin-bottom:8px;">
    <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
      <span class="mono" style="font-size:0.7rem; color:#94a3b8;">${{s}}</span>
      <span class="mono" style="font-size:0.68rem; color:#64748b;">${{(v * 100).toFixed(1)}}%</span>
    </div>
    <div class="bar-bg"><div class="bar-fill" style="width:${{(v / maxWeight * 100).toFixed(1)}}%;"></div></div>
  </div>`).join('');
</script>
</body>
</html>"""

# ── main execution: automatically ready on load ──────────────────────────────
# Ensure state is initialized so user lands directly into the working plant monitor
if "dg_done" not in st.session_state:
    st.session_state["dg_done"] = True

with st.spinner("Processing streaming sensor windows..."):
    res, weights = simulate(name, seed, onset)

dg_reps = res["DriftGuard"]["reps"]
sc = SCENARIOS[name]

# Prepare datasets for dashboard visualization
windows = list(range(40))
composite = [r["composite"] for r in dg_reps]
thr_vals = [r["thr"] for r in dg_reps]
heatmap = {s: [r["per_sensor"][s] for r in dg_reps] for s in SENSORS}
f1_series = {k: v["f1"] for k, v in res.items()}
triage_rows = [
    {"w": r["t"], "cause": r["cause"], "impact": round(r["impact"], 2), "v": r["version"]}
    for r in dg_reps if r["alert"]
]
log_rows = [
    {"w": int(w), "event": ev, "msg": msg}
    for w, ev, msg in res["DriftGuard"]["log"]
]

dashboard_html = _build_dashboard_html(
    windows, f1_series, composite, thr_vals,
    heatmap, onset, triage_rows, log_rows, weights,
    name, sc.description, sc.expected_cause
)

# Render professional industrial dashboard
components.html(dashboard_html, height=1050, scrolling=False)
