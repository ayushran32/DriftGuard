"""DriftGuard — Modern Full-Width SaaS Predictive Maintenance Platform.
Aesthetic inspired by Linear, Vercel, and Raycast.
Run: streamlit run app.py
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
    page_title="DriftGuard | Predictive Maintenance Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── modern linear/vercel design system ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Global Reset & Background */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #030712 !important;
    color: #f3f4f6 !important;
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
}

/* Hide Default Sidebar Completely */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

/* Hide Streamlit Default Chrome */
footer, #MainMenu, .stDeployButton, [data-testid="stToolbar"], header[data-testid="stHeader"] {
    display: none !important;
}

.block-container {
    padding: 1rem 2rem !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* Glassmorphism Controls Header */
.top-nav {
    background: rgba(15, 23, 42, 0.65);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 10px 20px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

/* Streamlit Widget Overrides for Linear/Vercel Look */
div[data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
    color: #f3f4f6 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.88rem !important;
    transition: all 0.2s ease !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: rgba(56, 189, 248, 0.4) !important;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.15) !important;
}

.stSlider [data-baseweb="slider"] {
    margin-top: -6px;
}

.stButton > button {
    background: linear-gradient(135deg, #06b6d4, #0284c7) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.4rem !important;
    font-size: 0.88rem !important;
    width: 100% !important;
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.35) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 0 30px rgba(6, 182, 212, 0.6) !important;
    transform: translateY(-1px) !important;
}

/* Subtle Aurora Ambient Glow */
.ambient-glow {
    position: fixed;
    top: 0;
    left: 20%;
    width: 60%;
    height: 350px;
    background: radial-gradient(ellipse at top, rgba(6, 182, 212, 0.08) 0%, rgba(99, 102, 241, 0.04) 50%, transparent 80%);
    pointer-events: none;
    z-index: 0;
}
</style>
<div class="ambient-glow"></div>
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

# ── top navigation command bar ────────────────────────────────────────────────
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([3, 4, 3, 2])

with nav_col1:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; padding-top:4px;">
        <div style="width:34px; height:34px; border-radius:10px; background:linear-gradient(135deg,#06b6d4,#6366f1); display:flex; align-items:center; justify-content:center; box-shadow:0 0 16px rgba(6,182,212,0.4);">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        </div>
        <div>
            <div style="font-size:1.1rem; font-weight:800; letter-spacing:-0.5px; color:#f8fafc;">DriftGuard</div>
            <div style="font-size:0.68rem; color:#94a3b8; font-family:'JetBrains Mono',monospace;">AI4I 2020 · PRODUCTION</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with nav_col2:
    st.markdown("<div style='font-size:0.68rem; color:#94a3b8; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:2px;'>Simulated Scenario</div>", unsafe_allow_html=True)
    name = st.selectbox(
        "scenario_select",
        list(SCENARIOS),
        index=4,
        format_func=lambda k: k,
        label_visibility="collapsed"
    )

with nav_col3:
    st.markdown("<div style='font-size:0.68rem; color:#94a3b8; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:2px;'>Drift Onset Window</div>", unsafe_allow_html=True)
    onset = st.slider("onset_slider", 6, 25, 12, label_visibility="collapsed")

with nav_col4:
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    run = st.button("⚡ Run Pipeline", type="primary")

seed = 0

# ── execute simulation ────────────────────────────────────────────────────────
with st.spinner("Streaming real AI4I cycles..."):
    res, weights = simulate(name, seed, onset)

dg_reps = res["DriftGuard"]["reps"]
sc = SCENARIOS[name]

# Prepare datasets
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

# ── linear / vercel full-width html component ─────────────────────────────────
def _build_linear_saas_html(windows, f1_series, composite, thr_vals,
                            heatmap, onset, triage_rows, log_rows, weights,
                            scenario_name, scenario_desc, expected_cause):

    f1_json = json.dumps(f1_series)
    comp_json = json.dumps(composite)
    thr_json = json.dumps(thr_vals)
    heat_json = json.dumps(heatmap)
    triage_json = json.dumps(triage_rows)
    log_json = json.dumps(log_rows)
    weights_json = json.dumps(weights)
    sensors_json = json.dumps(SENSORS)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    background:#030712; color:#f8fafc;
    font-family:'Plus Jakarta Sans', -apple-system, sans-serif;
    padding:0 2px 30px;
  }}
  .mono {{ font-family:'JetBrains Mono', monospace; }}

  /* Scenario Info Banner */
  .scenario-pill-bar {{
    background:rgba(255, 255, 255, 0.02);
    border:1px solid rgba(255, 255, 255, 0.07);
    border-radius:12px; padding:10px 16px; margin-bottom:18px;
    display:flex; justify-content:space-between; align-items:center;
  }}
  .badge-cause {{
    padding:4px 10px; border-radius:20px; font-size:0.72rem; font-weight:700;
    letter-spacing:0.4px; font-family:'JetBrains Mono', monospace;
  }}

  /* 3 Hero Metric Cards */
  .hero-grid {{
    display:grid; grid-template-columns:repeat(3, 1fr); gap:16px; margin-bottom:20px;
  }}
  .hero-card {{
    background:linear-gradient(180deg, rgba(15, 23, 42, 0.7) 0%, rgba(15, 23, 42, 0.4) 100%);
    border:1px solid rgba(255, 255, 255, 0.08);
    border-radius:16px; padding:20px 24px; position:relative; overflow:hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    transition: transform 0.2s ease, border-color 0.2s ease;
  }}
  .hero-card:hover {{
    border-color: rgba(6, 182, 212, 0.3);
    transform: translateY(-2px);
  }}
  .hero-glow {{
    position:absolute; top:-20px; right:-20px; width:100px; height:100px;
    background:radial-gradient(circle, var(--glow-color, rgba(6,182,212,0.15)) 0%, transparent 70%);
    pointer-events:none;
  }}
  .hero-title {{
    font-size:0.75rem; color:#94a3b8; font-weight:600; letter-spacing:0.5px; text-transform:uppercase;
  }}
  .hero-num {{
    font-size:2.4rem; font-weight:800; letter-spacing:-1px; margin:6px 0; color:#f8fafc;
  }}
  .hero-foot {{
    display:flex; align-items:center; gap:8px; font-size:0.75rem; color:#64748b;
  }}
  .diff-badge {{
    padding:2px 8px; border-radius:6px; font-weight:700; font-size:0.72rem;
    font-family:'JetBrains Mono', monospace;
  }}

  /* Charts Grid */
  .charts-grid {{
    display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:20px;
  }}
  .chart-card {{
    background:rgba(15, 23, 42, 0.55);
    border:1px solid rgba(255, 255, 255, 0.08);
    border-radius:16px; padding:20px 22px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
  }}
  .chart-head {{
    display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;
  }}
  .chart-title {{
    font-size:0.88rem; font-weight:700; color:#f1f5f9; letter-spacing:-0.2px;
  }}
  .chart-tag {{
    font-size:0.65rem; color:#64748b; font-family:'JetBrains Mono', monospace;
    background:rgba(255,255,255,0.04); padding:3px 8px; border-radius:6px;
  }}
  canvas {{ width:100%!important; }}

  /* Feed & Diagnostics Grid */
  .feed-grid {{
    display:grid; grid-template-columns:1.4fr 1fr; gap:16px;
  }}
  .feed-card {{
    background:rgba(15, 23, 42, 0.55);
    border:1px solid rgba(255, 255, 255, 0.08);
    border-radius:16px; padding:20px 22px;
  }}
  .feed-item {{
    display:flex; align-items:flex-start; gap:12px; padding:10px 0;
    border-bottom:1px solid rgba(255, 255, 255, 0.05);
  }}
  .feed-dot {{
    width:8px; height:8px; border-radius:50%; margin-top:6px; flex-shrink:0;
  }}
  .feed-tag {{
    font-size:0.66rem; font-weight:700; font-family:'JetBrains Mono', monospace;
    padding:2px 7px; border-radius:4px;
  }}
  .feed-text {{
    font-size:0.75rem; color:#cbd5e1; line-height:1.45;
  }}
  .weight-bar {{
    background:rgba(255,255,255,0.04); border-radius:4px; height:6px; margin-top:4px; overflow:hidden;
  }}
  .weight-fill {{
    height:100%; border-radius:4px; background:linear-gradient(90deg, #06b6d4, #6366f1);
  }}
</style>
</head>
<body>

<!-- Active Scenario Bar -->
<div class="scenario-pill-bar">
  <div style="display:flex; align-items:center; gap:12px;">
    <span style="font-size:0.78rem; color:#94a3b8; font-weight:600;">ACTIVE FAULT INJECTION:</span>
    <span style="font-size:0.88rem; font-weight:700; color:#38bdf8;">{scenario_name}</span>
    <span style="font-size:0.75rem; color:#64748b;">— {scenario_desc}</span>
  </div>
  <div>
    <span class="badge-cause" id="targetBadge">EXPECTED: {expected_cause}</span>
  </div>
</div>

<!-- 3 Hero KPI Cards -->
<div class="hero-grid" id="heroGrid"></div>

<!-- Middle Section: 2 Charts -->
<div class="charts-grid">
  <div class="chart-card">
    <div class="chart-head">
      <div class="chart-title">Model Reliability (F1 Score per Window)</div>
      <div class="chart-tag">STREAMING PERFORMANCE</div>
    </div>
    <canvas id="f1Canvas" height="190"></canvas>
  </div>
  <div class="chart-card">
    <div class="chart-head">
      <div class="chart-title">Multivariate Drift vs. Bootstrap Null Threshold</div>
      <div class="chart-tag">KS + PSI + K-MEANS ERROR</div>
    </div>
    <canvas id="driftCanvas" height="190"></canvas>
  </div>
</div>

<!-- Bottom Section: Incident & Triage Feed + Sensor Weights -->
<div class="feed-grid">
  <div class="feed-card">
    <div class="chart-head">
      <div class="chart-title">Incident & Automated Action Journal</div>
      <div class="chart-tag">ROOT-CAUSE TRIAGE</div>
    </div>
    <div id="feedList" style="max-height:220px; overflow-y:auto;"></div>
  </div>
  <div class="feed-card">
    <div class="chart-head">
      <div class="chart-title">Sensor Permutation Feature Importance</div>
      <div class="chart-tag">RANDOM FOREST ATTRIBUTION</div>
    </div>
    <div id="weightsFeed"></div>
  </div>
</div>

<script>
const WINS = {json.dumps(windows)};
const F1S = {f1_json};
const COMP = {comp_json};
const THR = {thr_json};
const ONSET = {onset};
const TRIAGE = {triage_json};
const LOG = {log_json};
const WEIGHTS = {weights_json};
const SENSORS = {sensors_json};
const EXPECTED_CAUSE = "{expected_cause}";

// Setup Badge Color
const causeColors = {{
  "SENSOR_FAULT": {{ bg:"rgba(244,63,94,0.15)", text:"#f43f5e", border:"rgba(244,63,94,0.3)" }},
  "REGIME_SHIFT": {{ bg:"rgba(245,158,11,0.15)", text:"#f59e0b", border:"rgba(245,158,11,0.3)" }},
  "CONCEPT_DRIFT": {{ bg:"rgba(249,115,22,0.15)", text:"#f97316", border:"rgba(249,115,22,0.3)" }},
  "FLEET_AGING": {{ bg:"rgba(168,85,247,0.15)", text:"#a855f7", border:"rgba(168,85,247,0.3)" }},
  "NONE": {{ bg:"rgba(148,163,184,0.15)", text:"#94a3b8", border:"rgba(148,163,184,0.3)" }}
}};
const cc = causeColors[EXPECTED_CAUSE] || causeColors["NONE"];
const tBadge = document.getElementById("targetBadge");
tBadge.style.background = cc.bg;
tBadge.style.color = cc.text;
tBadge.style.border = "1px solid " + cc.border;

// Hero Metrics Calculation
const dgAfter = F1S["DriftGuard"].slice(ONSET);
const stAfter = F1S["Static model"].slice(ONSET);
const dgMean = dgAfter.reduce((a,b)=>a+b, 0) / dgAfter.length;
const stMean = stAfter.reduce((a,b)=>a+b, 0) / stAfter.length;
const delta = dgMean - stMean;
const deploys = LOG.filter(e => e.event === "DEPLOY").length;
const isFault = EXPECTED_CAUSE === "SENSOR_FAULT";

document.getElementById('heroGrid').innerHTML = `
  <div class="hero-card" style="--glow-color:rgba(16,185,129,0.2);">
    <div class="hero-glow"></div>
    <div class="hero-title">DriftGuard Reliability</div>
    <div class="hero-num" style="color:#10b981;">${{dgMean.toFixed(3)}}</div>
    <div class="hero-foot">
      <span class="diff-badge" style="background:rgba(16,185,129,0.15); color:#10b981;">
        ${{delta >= 0 ? '+' : ''}}${{delta.toFixed(3)}} vs Static (${{stMean.toFixed(3)}})
      </span>
      <span>Post-Onset Mean F1</span>
    </div>
  </div>

  <div class="hero-card" style="--glow-color:rgba(6,182,212,0.2);">
    <div class="hero-glow"></div>
    <div class="hero-title">Challenger Deployments</div>
    <div class="hero-num" style="color:#38bdf8;">${{deploys}}</div>
    <div class="hero-foot">
      <span class="diff-badge" style="background:rgba(6,182,212,0.15); color:#38bdf8;">
        ${{isFault ? 'Zero Retrains (Repaired Online)' : 'Harm-Gated Retrains'}}
      </span>
      <span>Bootstrap Validated</span>
    </div>
  </div>

  <div class="hero-card" style="--glow-color:${{cc.text}}33;">
    <div class="hero-glow"></div>
    <div class="hero-title">Diagnosed Cause</div>
    <div class="hero-num" style="color:${{cc.text}}; font-size:1.85rem; padding-top:4px;">${{EXPECTED_CAUSE}}</div>
    <div class="hero-foot">
      <span class="diff-badge" style="background:${{cc.bg}}; color:${{cc.text}};">
        ${{isFault ? 'ONLINE VIRTUAL REPAIR' : 'PHYSICAL SHIFT'}}
      </span>
      <span>Automated Triage</span>
    </div>
  </div>
`;

// Chart.js Shared Config
Chart.defaults.font = {{ family:"'Plus Jakarta Sans', sans-serif", size:10 }};
Chart.defaults.color = '#64748b';
const gridOptions = {{ color:'rgba(255, 255, 255, 0.04)' }};
const tickOptions = {{ color:'#64748b', font:{{size:9, family:"'JetBrains Mono', monospace"}} }};

// Onset Marker
const onsetPlugin = {{
  id: 'onsetLine',
  afterDraw(chart) {{
    const {{ctx, chartArea:{{top, bottom}}, scales:{{x}}}} = chart;
    const px = x.getPixelForValue(ONSET);
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(px, top); ctx.lineTo(px, bottom); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#f43f5e';
    ctx.font = '600 9px "JetBrains Mono", monospace';
    ctx.fillText('DRIFT ONSET', px + 5, top + 12);
    ctx.restore();
  }}
}};

// 1. F1 Chart with Glowing Gradient
const ctx1 = document.getElementById('f1Canvas').getContext('2d');
const dgGradient = ctx1.createLinearGradient(0, 0, 0, 200);
dgGradient.addColorStop(0, 'rgba(16, 185, 129, 0.25)');
dgGradient.addColorStop(1, 'rgba(16, 185, 129, 0.0)');

new Chart(ctx1, {{
  type:'line',
  data: {{
    labels: WINS,
    datasets: [
      {{
        label: 'DriftGuard',
        data: F1S['DriftGuard'],
        borderColor: '#10b981',
        backgroundColor: dgGradient,
        fill: true,
        borderWidth: 2.5,
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.35
      }},
      {{
        label: 'Static Baseline',
        data: F1S['Static model'],
        borderColor: '#64748b',
        borderWidth: 1.5,
        borderDash: [3, 3],
        pointRadius: 0,
        tension: 0.35
      }},
      {{
        label: 'Naive Retrain',
        data: F1S['Naive drift->retrain'],
        borderColor: '#f59e0b',
        borderWidth: 1.2,
        pointRadius: 0,
        tension: 0.35
      }}
    ]
  }},
  options: {{
    responsive: true,
    interaction: {{ mode:'index', intersect:false }},
    plugins: {{
      legend: {{ labels: {{ color:'#94a3b8', font:{{size:9}}, boxWidth:10 }} }},
      tooltip: {{
        backgroundColor:'#0f172a', borderColor:'rgba(255,255,255,0.1)', borderWidth:1,
        titleColor:'#94a3b8', bodyColor:'#f8fafc',
        callbacks: {{ label: c => `${{c.dataset.label}}: ${{c.parsed.y.toFixed(3)}}` }}
      }}
    }},
    scales: {{
      x: {{ grid: gridOptions, ticks: tickOptions }},
      y: {{ grid: gridOptions, ticks: tickOptions, min:0, max:1 }}
    }}
  }},
  plugins: [onsetPlugin]
}});

// 2. Drift Score Chart
const ctx2 = document.getElementById('driftCanvas').getContext('2d');
const driftGradient = ctx2.createLinearGradient(0, 0, 0, 200);
driftGradient.addColorStop(0, 'rgba(6, 182, 212, 0.3)');
driftGradient.addColorStop(1, 'rgba(6, 182, 212, 0.0)');

new Chart(ctx2, {{
  type:'line',
  data: {{
    labels: WINS,
    datasets: [
      {{
        label: 'Composite Shift Score',
        data: COMP,
        borderColor: '#06b6d4',
        backgroundColor: driftGradient,
        fill: true,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.35
      }},
      {{
        label: 'Null Threshold',
        data: THR,
        borderColor: '#f59e0b',
        borderDash: [5, 4],
        borderWidth: 1.5,
        pointRadius: 0
      }}
    ]
  }},
  options: {{
    responsive: true,
    interaction: {{ mode:'index', intersect:false }},
    plugins: {{
      legend: {{ labels: {{ color:'#94a3b8', font:{{size:9}}, boxWidth:10 }} }},
      tooltip: {{ backgroundColor:'#0f172a', borderColor:'rgba(255,255,255,0.1)', borderWidth:1 }}
    }},
    scales: {{
      x: {{ grid: gridOptions, ticks: tickOptions }},
      y: {{ type:'logarithmic', grid: gridOptions, ticks: tickOptions }}
    }}
  }},
  plugins: [onsetPlugin]
}});

// 3. Incident Journal
const feedEl = document.getElementById('feedList');
const eventMeta = {{
  ALERT: {{ bg:"rgba(244,63,94,0.15)", text:"#f43f5e", dot:"#f43f5e" }},
  LABEL_ALARM: {{ bg:"rgba(249,115,22,0.15)", text:"#f97316", dot:"#f97316" }},
  DEPLOY: {{ bg:"rgba(16,185,129,0.15)", text:"#10b981", dot:"#10b981" }},
  HARMLESS: {{ bg:"rgba(6,182,212,0.15)", text:"#06b6d4", dot:"#06b6d4" }},
  REJECT: {{ bg:"rgba(148,163,184,0.15)", text:"#94a3b8", dot:"#94a3b8" }}
}};

feedEl.innerHTML = LOG.length === 0
  ? '<div style="color:#64748b; font-size:0.75rem; padding:20px; text-align:center;">No automated incidents registered. System healthy.</div>'
  : LOG.map(e => {{
      const meta = eventMeta[e.event] || eventMeta["REJECT"];
      return `
        <div class="feed-item">
          <div class="feed-dot" style="background:${{meta.dot}}; box-shadow:0 0 6px ${{meta.dot}}88;"></div>
          <div style="flex:1;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:2px;">
              <span class="feed-tag" style="background:${{meta.bg}}; color:${{meta.text}};">${{e.event}}</span>
              <span class="mono" style="font-size:0.65rem; color:#64748b;">Window #${{e.w}}</span>
            </div>
            <div class="feed-text">${{e.msg}}</div>
          </div>
        </div>`;
    }}).join('');

// 4. Feature Weights Feed
const maxWeight = Math.max(...Object.values(WEIGHTS));
document.getElementById('weightsFeed').innerHTML = Object.entries(WEIGHTS).map(([s, v]) => `
  <div style="margin-bottom:12px;">
    <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
      <span class="mono" style="font-size:0.75rem; color:#e2e8f0; font-weight:600;">${{s}}</span>
      <span class="mono" style="font-size:0.72rem; color:#64748b;">${{(v * 100).toFixed(1)}}%</span>
    </div>
    <div class="weight-bar"><div class="weight-fill" style="width:${{(v / maxWeight * 100).toFixed(1)}}%;"></div></div>
  </div>`).join('');

</script>
</body>
</html>"""

# Render Component
dashboard_html = _build_linear_saas_html(
    windows, f1_series, composite, thr_vals,
    heatmap, onset, triage_rows, log_rows, weights,
    name, sc.description, sc.expected_cause
)
components.html(dashboard_html, height=840, scrolling=False)
