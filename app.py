"""DriftGuard — premium dark-industrial dashboard.
Run with:  streamlit run app.py
"""
import json
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from driftguard.data import DataSource, SENSORS
from driftguard.simulate import SCENARIOS, build_stream
from driftguard.engine import DriftGuard, NaiveRetrainer, Static, run_strategy

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DriftGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #060d1f !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: #0a1628 !important;
    border-right: 1px solid rgba(0,212,255,0.12) !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; font-family: 'Inter',sans-serif !important; }
.stSelectbox > div > div {
    background: #0f1e35 !important;
    border: 1px solid rgba(0,212,255,0.22) !important;
    border-radius: 8px !important;
}
.stNumberInput input {
    background: #0f1e35 !important;
    border: 1px solid rgba(0,212,255,0.22) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
.stButton > button {
    background: linear-gradient(135deg,#00d4ff,#0099cc) !important;
    color: #060d1f !important; font-weight: 700 !important;
    border: none !important; border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important; font-size: 0.95rem !important;
    width: 100% !important; letter-spacing: 0.5px !important;
    box-shadow: 0 0 22px rgba(0,212,255,0.35) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { box-shadow: 0 0 38px rgba(0,212,255,0.55) !important; }
footer { display:none !important; }
#MainMenu { display:none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 1rem !important; padding-bottom: 0 !important; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.35} }
</style>
""", unsafe_allow_html=True)

# ── constants ─────────────────────────────────────────────────────────────────
CAUSE_COLOR = {
    "SENSOR_FAULT":  "#ef4444",
    "REGIME_SHIFT":  "#ffb300",
    "FLEET_AGING":   "#a78bfa",
    "CONCEPT_DRIFT": "#f97316",
    "NONE":          "#334155",
}
CAUSE_ICON = {
    "SENSOR_FAULT": "⚠️", "REGIME_SHIFT": "📡",
    "FLEET_AGING": "🔧", "CONCEPT_DRIFT": "🧠", "NONE": "✓",
}
EVENT_COLOR = {
    "ALERT": "#ef4444", "LABEL_ALARM": "#f97316", "DEPLOY": "#22c55e",
    "REJECT": "#64748b", "HARMLESS": "#22d3ee", "CLEAR": "#334155",
    "REBASELINE": "#a78bfa",
}
STRAT_COLOR = {
    "Static model":         "#64748b",
    "Naive drift->retrain": "#f97316",
    "DriftGuard":           "#00d4ff",
}

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 20px;border-bottom:1px solid rgba(0,212,255,0.12);margin-bottom:20px;">
      <div style="display:flex;align-items:center;gap:10px;">
        <span style="font-size:1.7rem;">🛡️</span>
        <div>
          <div style="font-size:1.2rem;font-weight:800;color:#00d4ff;letter-spacing:-0.5px;">DriftGuard</div>
          <div style="font-size:0.68rem;color:#475569;margin-top:1px;">Predictive Maintenance AI</div>
        </div>
      </div>
      <div style="margin-top:12px;display:flex;align-items:center;gap:6px;">
        <span style="width:7px;height:7px;border-radius:50%;background:#22d3ee;
              display:inline-block;box-shadow:0 0 8px #22d3ee;animation:pulse 2s infinite;"></span>
        <span style="font-size:0.7rem;color:#22d3ee;font-weight:600;letter-spacing:0.5px;">AI4I 2020 DATA STREAM</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.72rem;color:#475569;font-weight:700;text-transform:uppercase;"
                "letter-spacing:1px;margin-bottom:5px;'>Scenario</div>", unsafe_allow_html=True)
    name = st.selectbox("scenario_sel", list(SCENARIOS), index=4,
                        format_func=lambda k: k, label_visibility="collapsed")
    st.markdown(
        f"<div style='font-size:0.7rem;color:#64748b;margin:-6px 0 14px;padding:6px 10px;"
        f"background:#0f1e35;border-radius:6px;border-left:3px solid #ffb300;'>"
        f"{SCENARIOS[name].description}</div>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.72rem;color:#475569;font-weight:700;text-transform:uppercase;"
                "letter-spacing:1px;margin-bottom:5px;'>Random Seed</div>", unsafe_allow_html=True)
    seed = st.number_input("seed_inp", 0, 99, 0, label_visibility="collapsed")

    st.markdown("<div style='font-size:0.72rem;color:#475569;font-weight:700;text-transform:uppercase;"
                "letter-spacing:1px;margin-bottom:5px;'>Drift Onset (window)</div>", unsafe_allow_html=True)
    onset = st.slider("onset_sl", 6, 25, 12, label_visibility="collapsed")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    run = st.button("⚡  Run Simulation", type="primary")

    st.markdown("""
    <div style="margin-top:20px;padding-top:14px;border-top:1px solid rgba(255,255,255,0.05);">
      <div style="font-size:0.65rem;color:#1e293b;text-align:center;line-height:1.7;">
        KS-test · PSI · k-means · Bootstrap calibration<br/>
        Triage · Online repair · Harm-gated retraining
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── data helpers ──────────────────────────────────────────────────────────────
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



# ═══════════════════════ builder functions ════════════════════════════════════

def _landing():
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
         min-height:72vh;text-align:center;padding:40px;">
      <div style="font-size:4rem;margin-bottom:16px;filter:drop-shadow(0 0 24px #00d4ff99);">🛡️</div>
      <h1 style="font-size:2.1rem;font-weight:800;color:#e2e8f0;letter-spacing:-1px;margin-bottom:8px;">DriftGuard</h1>
      <p style="font-size:.95rem;color:#475569;max-width:520px;line-height:1.7;margin-bottom:6px;">
        AI-powered predictive maintenance using <strong style="color:#00d4ff">AI4I 2020</strong> sensor data.
        Detects drift, triages its root cause, repairs faulty inputs online, and retrains only when
        delayed labels confirm real performance harm.
      </p>
      <div style="display:flex;gap:14px;margin:26px 0;flex-wrap:wrap;justify-content:center;">
        <div style="background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);border-radius:10px;padding:12px 18px;min-width:130px;">
          <div style="font-size:1.2rem;">⚠️</div>
          <div style="font-size:.73rem;color:#ef4444;font-weight:700;margin-top:4px;">SENSOR_FAULT</div>
          <div style="font-size:.63rem;color:#475569;margin-top:2px;">Repair online; no retrain</div>
        </div>
        <div style="background:rgba(255,179,0,.1);border:1px solid rgba(255,179,0,.3);border-radius:10px;padding:12px 18px;min-width:130px;">
          <div style="font-size:1.2rem;">📡</div>
          <div style="font-size:.73rem;color:#ffb300;font-weight:700;margin-top:4px;">REGIME_SHIFT</div>
          <div style="font-size:.63rem;color:#475569;margin-top:2px;">Validate then retrain</div>
        </div>
        <div style="background:rgba(249,115,22,.1);border:1px solid rgba(249,115,22,.3);border-radius:10px;padding:12px 18px;min-width:130px;">
          <div style="font-size:1.2rem;">🧠</div>
          <div style="font-size:.73rem;color:#f97316;font-weight:700;margin-top:4px;">CONCEPT_DRIFT</div>
          <div style="font-size:.63rem;color:#475569;margin-top:2px;">Label-gated retrain</div>
        </div>
        <div style="background:rgba(167,139,250,.1);border:1px solid rgba(167,139,250,.3);border-radius:10px;padding:12px 18px;min-width:130px;">
          <div style="font-size:1.2rem;">🔧</div>
          <div style="font-size:.73rem;color:#a78bfa;font-weight:700;margin-top:4px;">FLEET_AGING</div>
          <div style="font-size:.63rem;color:#475569;margin-top:2px;">Informational only</div>
        </div>
      </div>
      <p style="font-size:.78rem;color:#1e293b;">← Pick a scenario in the sidebar and press
        <strong style="color:#00d4ff;">Run Simulation</strong></p>
    </div>
    """, unsafe_allow_html=True)


def _build_html(windows, f1_series, composite, thr_vals,
                heatmap, onset, triage_rows, log_rows, weights):
    """Return a fully self-contained HTML page rendered inside components.html."""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{background:#060d1f;color:#e2e8f0;
        font-family:'Inter',system-ui,sans-serif;padding:0 4px 8px;}}
  .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;}}
  .grid4{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:12px;}}
  .card{{background:rgba(255,255,255,0.028);border:1px solid rgba(255,255,255,0.07);
         border-radius:12px;padding:14px 16px;}}
  .card-t{{font-size:.72rem;font-weight:700;color:#475569;text-transform:uppercase;
           letter-spacing:1px;margin-bottom:10px;}}
  canvas{{display:block;width:100%!important;}}
  .kpi-top{{border-top:3px solid var(--c);}}
  .kpi-icon{{position:absolute;top:10px;right:12px;font-size:1rem;opacity:.4;}}
  .kpi-lbl{{font-size:.62rem;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:1px;}}
  .kpi-val{{font-size:1.6rem;font-weight:800;letter-spacing:-1px;margin:3px 0;color:var(--c);}}
  .kpi-sub{{font-size:.65rem;color:#475569;}}
  .kpi-hint{{font-size:.6rem;color:#1e293b;margin-top:2px;}}
  .pill{{display:inline-block;padding:2px 8px;border-radius:20px;font-size:.65rem;font-weight:700;letter-spacing:.4px;}}
  table{{width:100%;border-collapse:collapse;font-size:.76rem;}}
  th{{color:#334155;font-weight:700;text-transform:uppercase;font-size:.62rem;letter-spacing:.8px;
      padding:5px 7px;border-bottom:1px solid rgba(255,255,255,0.06);text-align:left;}}
  td{{padding:6px 7px;border-bottom:1px solid rgba(255,255,255,0.04);vertical-align:middle;}}
  tr:hover td{{background:rgba(255,255,255,0.018);}}
  .log-row{{display:flex;align-items:flex-start;gap:9px;padding:7px 0;
            border-bottom:1px solid rgba(255,255,255,0.04);}}
  .log-dot{{width:7px;height:7px;border-radius:50%;margin-top:4px;flex-shrink:0;}}
  .log-w{{font-size:.65rem;color:#334155;min-width:62px;margin-top:2px;font-variant-numeric:tabular-nums;}}
  .log-ev{{font-size:.7rem;font-weight:700;}}
  .log-msg{{font-size:.67rem;color:#475569;margin-top:1px;line-height:1.4;}}
  .wb{{background:#0f1e35;border-radius:4px;height:5px;margin-top:3px;overflow:hidden;}}
  .wf{{height:100%;border-radius:4px;background:linear-gradient(90deg,#00d4ff,#0099cc);}}
</style>
</head>
<body>

<!-- KPI row -->
<div class="grid4" id="kpis"></div>

<!-- F1 + Drift Score -->
<div class="grid2">
  <div class="card">
    <div class="card-t">📈 Model Performance — F1 per Window</div>
    <canvas id="f1c" height="170"></canvas>
  </div>
  <div class="card">
    <div class="card-t">📡 Drift Score vs Calibrated Threshold</div>
    <canvas id="dc" height="170"></canvas>
  </div>
</div>

<!-- Heatmap + Triage -->
<div class="grid2">
  <div class="card">
    <div class="card-t">🌡️ Per-Sensor Drift Intensity (× threshold)</div>
    <canvas id="hc" height="175"></canvas>
  </div>
  <div class="card">
    <div class="card-t">🔎 Triage Timeline</div>
    <div id="tri" style="max-height:210px;overflow-y:auto;"></div>
  </div>
</div>

<!-- Log + Weights -->
<div class="grid2">
  <div class="card">
    <div class="card-t">📋 Incident Log</div>
    <div id="log" style="max-height:200px;overflow-y:auto;"></div>
  </div>
  <div class="card">
    <div class="card-t">⚖️ Sensor Importance (permutation)</div>
    <div id="wts"></div>
  </div>
</div>

<script>
// ── data ──────────────────────────────────────────────────────────────────────
const WINS   = {json.dumps(windows)};
const F1S    = {json.dumps(f1_series)};
const COMP   = {json.dumps(composite)};
const THR    = {json.dumps(thr_vals)};
const HEAT   = {json.dumps(heatmap)};
const ONSET  = {onset};
const TRIAGE = {json.dumps(triage_rows)};
const LOG    = {json.dumps(log_rows)};
const CC     = {json.dumps(CAUSE_COLOR)};
const EC     = {json.dumps(EVENT_COLOR)};
const SNSR   = {json.dumps(SENSORS)};
const WTS    = {json.dumps(weights)};
const SC     = {json.dumps(STRAT_COLOR)};

// ── chart defaults ────────────────────────────────────────────────────────────
const gl  = {{color:'rgba(255,255,255,0.04)'}};
const tk  = {{color:'#334155',font:{{size:9,family:'Inter,system-ui,sans-serif'}}}};
Chart.defaults.font = {{family:'Inter,system-ui,sans-serif',size:10}};
Chart.defaults.color = '#475569';

// onset annotation plugin
const onsetPlugin = {{
  id:'onset',
  afterDraw(chart){{
    const {{ctx,chartArea:{{top,bottom}},scales:{{x}}}} = chart;
    const px = x.getPixelForValue(ONSET);
    ctx.save();
    ctx.setLineDash([4,4]); ctx.strokeStyle='#ef4444'; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.moveTo(px,top); ctx.lineTo(px,bottom); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle='#ef4444'; ctx.font='500 8px Inter';
    ctx.fillText('ONSET',px+3,top+10);
    ctx.restore();
  }}
}};

// ── KPIs ──────────────────────────────────────────────────────────────────────
const dga = F1S["DriftGuard"].slice(ONSET);
const sta = F1S["Static model"].slice(ONSET);
const dgM = dga.reduce((a,b)=>a+b,0)/dga.length;
const stM = sta.reduce((a,b)=>a+b,0)/sta.length;
const dlt = (dgM-stM).toFixed(3);
const dep = LOG.filter(e=>e.event==="DEPLOY").length;
const alr = TRIAGE.filter(r=>r.w>=ONSET).length;
const fa  = TRIAGE.filter(r=>r.w<ONSET).length;

function kpi(title,val,sub,hint,c,icon){{
  return `<div class="card kpi-top" style="--c:${{c}};position:relative">
    <span class="kpi-icon">${{icon}}</span>
    <div class="kpi-lbl">${{title}}</div>
    <div class="kpi-val">${{val}}</div>
    <div class="kpi-sub">${{sub}}</div>
    <div class="kpi-hint">${{hint}}</div>
  </div>`;
}}
document.getElementById('kpis').innerHTML =
  kpi('DriftGuard F1', dgM.toFixed(3), `+${{dlt}} vs Static`, 'After drift onset','#22c55e','📈')+
  kpi('Static Baseline', stM.toFixed(3), 'No drift handling',  'After drift onset','#64748b','📊')+
  kpi('Models Deployed', dep, 'Validated challengers only',    'Harm-gated','#ffb300','🚀')+
  kpi('Alert Windows',   alr, `${{fa}} false alarm(s) pre-onset`,'After onset','#22d3ee','🔔');

// ── F1 lines ──────────────────────────────────────────────────────────────────
new Chart(document.getElementById('f1c'),{{
  type:'line',
  data:{{
    labels:WINS,
    datasets:Object.entries(F1S).map(([n,d])=>{{
      const c=SC[n]||'#94a3b8';
      return {{label:n,data:d,borderColor:c,backgroundColor:'transparent',
              borderWidth:n==='DriftGuard'?2.5:1.5,pointRadius:0,pointHoverRadius:3,tension:.3}};
    }})
  }},
  options:{{
    responsive:true,animation:{{duration:500}},
    interaction:{{mode:'index',intersect:false}},
    plugins:{{
      legend:{{labels:{{color:'#475569',font:{{size:9}},boxWidth:10}}}},
      tooltip:{{backgroundColor:'#0f1e35',borderColor:'rgba(255,255,255,.08)',borderWidth:1,
        titleColor:'#64748b',bodyColor:'#e2e8f0',
        callbacks:{{label:c=>`${{c.dataset.label}}: ${{c.parsed.y.toFixed(3)}}`}}}}
    }},
    scales:{{
      x:{{grid:gl,ticks:{{...tk,maxTicksLimit:8}},title:{{display:true,text:'window',color:'#334155',font:{{size:9}}}}}},
      y:{{grid:gl,ticks:tk,min:0,max:1,title:{{display:true,text:'F1',color:'#334155',font:{{size:9}}}}}}
    }}
  }},
  plugins:[onsetPlugin]
}});

// ── Drift Score ───────────────────────────────────────────────────────────────
new Chart(document.getElementById('dc'),{{
  type:'line',
  data:{{labels:WINS,datasets:[
    {{label:'Composite drift score',data:COMP,borderColor:'#00d4ff',
      backgroundColor:'rgba(0,212,255,0.05)',fill:true,borderWidth:2,pointRadius:0,tension:.3}},
    {{label:'Calibrated threshold',data:THR,borderColor:'#ffb300',borderDash:[5,4],
      borderWidth:1.5,pointRadius:0}}
  ]}},
  options:{{
    responsive:true,animation:{{duration:500}},
    interaction:{{mode:'index',intersect:false}},
    plugins:{{
      legend:{{labels:{{color:'#475569',font:{{size:9}},boxWidth:10}}}},
      tooltip:{{backgroundColor:'#0f1e35',borderColor:'rgba(255,255,255,.08)',borderWidth:1,
        titleColor:'#64748b',bodyColor:'#e2e8f0'}}
    }},
    scales:{{
      x:{{grid:gl,ticks:{{...tk,maxTicksLimit:8}},title:{{display:true,text:'window',color:'#334155',font:{{size:9}}}}}},
      y:{{type:'logarithmic',grid:gl,ticks:tk,title:{{display:true,text:'score (log)',color:'#334155',font:{{size:9}}}}}}
    }}
  }},
  plugins:[onsetPlugin]
}});

// ── Per-sensor drift bars ─────────────────────────────────────────────────────
const hCols=['#00d4ff','#22c55e','#ffb300','#f97316','#a78bfa'];
new Chart(document.getElementById('hc'),{{
  type:'bar',
  data:{{labels:WINS,datasets:SNSR.map((s,i)=>{{
    return {{label:s,data:HEAT[s].map(v=>Math.min(v,5)),
             backgroundColor:hCols[i]+'77',borderWidth:0,
             barPercentage:1,categoryPercentage:1}};
  }})}},
  options:{{
    responsive:true,animation:{{duration:500}},
    interaction:{{mode:'index',intersect:false}},
    plugins:{{
      legend:{{labels:{{color:'#475569',font:{{size:9}},boxWidth:10}}}},
      tooltip:{{backgroundColor:'#0f1e35',borderColor:'rgba(255,255,255,.08)',borderWidth:1,
        titleColor:'#64748b',bodyColor:'#e2e8f0',
        callbacks:{{label:c=>`${{c.dataset.label}}: ${{c.parsed.y.toFixed(2)}}×`}}}}
    }},
    scales:{{
      x:{{stacked:false,grid:gl,ticks:{{...tk,maxTicksLimit:8}},
          title:{{display:true,text:'window',color:'#334155',font:{{size:9}}}}}},
      y:{{grid:gl,ticks:tk,min:0,
          title:{{display:true,text:'drift × threshold',color:'#334155',font:{{size:9}}}}}}
    }}
  }},
  plugins:[onsetPlugin]
}});

// ── Triage table ──────────────────────────────────────────────────────────────
function pill(cause){{
  const c=CC[cause]||'#334155';
  return `<span class="pill" style="background:${{c}}20;color:${{c}};border:1px solid ${{c}}40">${{cause}}</span>`;
}}
const tri=document.getElementById('tri');
tri.innerHTML = TRIAGE.length===0
  ? '<div style="color:#1e293b;font-size:.75rem;padding:18px 0;text-align:center">No alert windows</div>'
  : `<table><tr><th>Win</th><th>Cause</th><th>Impact</th><th>Mdl</th></tr>
      ${{TRIAGE.slice(0,14).map(r=>`<tr>
        <td style="color:#64748b;font-variant-numeric:tabular-nums">${{r.w}}</td>
        <td>${{pill(r.cause)}}</td>
        <td><div style="display:flex;align-items:center;gap:5px">
          <div class="wb" style="width:52px"><div class="wf" style="width:${{Math.min(r.impact*100,100).toFixed(0)}}%"></div></div>
          <span style="color:#334155;font-size:.63rem">${{r.impact.toFixed(2)}}</span></div></td>
        <td style="color:#334155">v${{r.v}}</td>
      </tr>`).join('')}}</table>`;

// ── Incident log ──────────────────────────────────────────────────────────────
const logEl=document.getElementById('log');
logEl.innerHTML = LOG.length===0
  ? '<div style="color:#1e293b;font-size:.75rem;padding:18px 0;text-align:center">No incidents</div>'
  : LOG.map(e=>`
    <div class="log-row">
      <div class="log-dot" style="background:${{e.color}};box-shadow:0 0 5px ${{e.color}}88"></div>
      <div style="flex:1;min-width:0">
        <div style="display:flex;align-items:center;gap:7px;flex-wrap:wrap">
          <span class="log-ev" style="color:${{e.color}}">${{e.event}}</span>
          <span class="log-w">window ${{e.w}}</span>
        </div>
        <div class="log-msg">${{e.msg}}</div>
      </div>
    </div>`).join('');

// ── Sensor weights ────────────────────────────────────────────────────────────
const maxW=Math.max(...Object.values(WTS));
document.getElementById('wts').innerHTML=Object.entries(WTS).map(([s,v])=>`
  <div style="margin-bottom:9px">
    <div style="display:flex;justify-content:space-between;margin-bottom:2px">
      <span style="font-size:.72rem;color:#94a3b8;font-weight:600">${{s}}</span>
      <span style="font-size:.68rem;color:#334155">${{(v*100).toFixed(1)}}%</span>
    </div>
    <div class="wb"><div class="wf" style="width:${{(v/maxW*100).toFixed(1)}}%"></div></div>
  </div>`).join('');
</script>
</body>
</html>"""


# ── main area ─────────────────────────────────────────────────────────────────
if run or "dg_done" in st.session_state:
    st.session_state["dg_done"] = True
    with st.spinner("Simulating 40 sensor windows × 3 strategies…"):
        res, weights = simulate(name, seed, onset)

    dg_reps = res["DriftGuard"]["reps"]
    sc = SCENARIOS[name]
    truth_col = CAUSE_COLOR.get(sc.expected_cause, "#64748b")
    truth_icon = CAUSE_ICON.get(sc.expected_cause, "⚡")

    # scenario truth banner
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:12px;padding:11px 18px;"
        f"background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);"
        f"border-left:4px solid {truth_col};border-radius:10px;margin-bottom:16px;'>"
        f"<span style='font-size:1.3rem;'>{truth_icon}</span>"
        f"<div><div style='font-size:0.68rem;color:#64748b;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:1px;'>Ground-truth drift cause</div>"
        f"<div style='font-size:0.92rem;color:#e2e8f0;font-weight:600;margin-top:2px;'>"
        f"<span style='color:{truth_col};font-weight:800;'>{sc.expected_cause}</span>"
        f"&nbsp;·&nbsp;{sc.description}</div></div></div>",
        unsafe_allow_html=True,
    )

    # prepare data for JS
    windows    = list(range(40))
    composite  = [r["composite"] for r in dg_reps]
    thr_vals   = [r["thr"] for r in dg_reps]
    heatmap    = {s: [r["per_sensor"][s] for r in dg_reps] for s in SENSORS}
    f1_series  = {k: v["f1"] for k, v in res.items()}
    triage_rows = [
        {"w": r["t"], "cause": r["cause"], "impact": round(r["impact"], 2), "v": r["version"]}
        for r in dg_reps if r["alert"]
    ]
    log_rows = [
        {"w": int(w), "event": ev, "msg": msg, "color": EVENT_COLOR.get(ev, "#64748b")}
        for w, ev, msg in res["DriftGuard"]["log"]
    ]

    html = _build_html(windows, f1_series, composite, thr_vals,
                       heatmap, onset, triage_rows, log_rows, weights)
    components.html(html, height=1040, scrolling=False)

else:
    _landing()
