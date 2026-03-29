"""
DataOps AI — Production SaaS Interface
Built by Mohammed Amine Goumri
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, io, re
import anthropic
import plotly.express as px
import plotly.graph_objects as go
import sqlalchemy
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DataOps AI",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⬡</text></svg>",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

STYLES = """
<style>
/* ── Google Fonts ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600&family=Geist+Mono:wght@400;500&display=swap');

/* ── Design tokens ────────────────────────────────────── */
:root {
    --bg:          #ffffff;
    --bg-subtle:   #f9fafb;
    --bg-muted:    #f3f4f6;
    --border:      #e5e7eb;
    --border-dark: #d1d5db;
    --text-strong: #111827;
    --text-base:   #374151;
    --text-muted:  #6b7280;
    --text-faint:  #9ca3af;
    --accent:      #1c64f2;
    --accent-hover:#1a56db;
    --accent-light:#eff4ff;
    --success:     #057a55;
    --success-bg:  #f0fff4;
    --success-bd:  #c6f6d5;
    --warning:     #92400e;
    --warning-bg:  #fffbeb;
    --warning-bd:  #fde68a;
    --danger:      #9b1c1c;
    --danger-bg:   #fef2f2;
    --danger-bd:   #fecaca;
    --info:        #1e40af;
    --info-bg:     #eff6ff;
    --info-bd:     #bfdbfe;
    --radius-sm:   6px;
    --radius-md:   10px;
    --radius-lg:   14px;
    --shadow-sm:   0 1px 2px rgba(0,0,0,0.05);
    --shadow-md:   0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -2px rgba(0,0,0,0.05);
    --font:        'Geist', -apple-system, sans-serif;
    --mono:        'Geist Mono', 'SF Mono', monospace;
}

/* ── Base reset ───────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: var(--font) !important;
    color: var(--text-base);
    background: var(--bg);
    -webkit-font-smoothing: antialiased;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="collapsedControl"] { display: none !important; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── App shell ────────────────────────────────────────── */
.app-shell {
    max-width: 1180px;
    margin: 0 auto;
    padding: 0 2rem 4rem;
}

/* ── Topbar ───────────────────────────────────────────── */
.topbar {
    max-width: 1180px;
    margin: 0 auto;
    padding: 0 2rem;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
    position: sticky;
    top: 0;
    z-index: 100;
}
.topbar-brand {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-strong);
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.topbar-brand-dot {
    width: 8px; height: 8px;
    background: var(--accent);
    border-radius: 50%;
    display: inline-block;
}
.topbar-meta {
    font-size: 0.75rem;
    color: var(--text-faint);
    font-family: var(--mono);
}

/* ── Tab navigation ───────────────────────────────────── */
.tab-nav {
    max-width: 1180px;
    margin: 0 auto;
    padding: 0 2rem;
    display: flex;
    align-items: center;
    gap: 0;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
    position: sticky;
    top: 56px;
    z-index: 99;
}
.tab-item {
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--text-muted);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    white-space: nowrap;
    transition: color 0.15s, border-color 0.15s;
    text-decoration: none;
}
.tab-item:hover { color: var(--text-strong); }
.tab-item.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
    font-weight: 600;
}
.tab-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 18px;
    height: 18px;
    background: var(--bg-muted);
    color: var(--text-muted);
    font-size: 0.65rem;
    font-weight: 600;
    border-radius: 999px;
    padding: 0 5px;
    margin-left: 6px;
    font-family: var(--mono);
    vertical-align: middle;
}
.tab-badge.active {
    background: var(--accent-light);
    color: var(--accent);
}

/* ── Page sections ────────────────────────────────────── */
.section {
    padding: 2rem 0 1rem;
}
.section-header {
    margin-bottom: 1.5rem;
}
.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-strong);
    letter-spacing: -0.02em;
    margin: 0 0 0.25rem;
}
.section-desc {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin: 0;
    line-height: 1.5;
}
.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

/* ── KPI cards ────────────────────────────────────────── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.kpi-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.25rem 1.5rem;
    box-shadow: var(--shadow-sm);
}
.kpi-card-label {
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-faint);
    margin-bottom: 0.6rem;
}
.kpi-card-value {
    font-size: 1.75rem;
    font-weight: 600;
    color: var(--text-strong);
    letter-spacing: -0.03em;
    line-height: 1;
    font-family: var(--mono);
}
.kpi-card-sub {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
}
.kpi-card-delta {
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 0.3rem;
}
.delta-up   { color: var(--success); }
.delta-down { color: var(--danger);  }
.delta-flat { color: var(--text-faint); }

/* ── Status badges ────────────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    border: 1px solid transparent;
    font-family: var(--mono);
}
.badge-success { background: var(--success-bg); color: var(--success); border-color: var(--success-bd); }
.badge-warning { background: var(--warning-bg); color: var(--warning); border-color: var(--warning-bd); }
.badge-danger  { background: var(--danger-bg);  color: var(--danger);  border-color: var(--danger-bd);  }
.badge-info    { background: var(--info-bg);    color: var(--info);    border-color: var(--info-bd);    }
.badge-neutral { background: var(--bg-muted);   color: var(--text-muted); border-color: var(--border); }

/* ── Data table wrapper ───────────────────────────────── */
.table-wrap {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}

/* ── Issue rows ───────────────────────────────────────── */
.issue-list { display: flex; flex-direction: column; gap: 0.5rem; margin: 0.5rem 0; }
.issue-item {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    border-radius: var(--radius-md);
    border: 1px solid transparent;
    font-size: 0.82rem;
}
.issue-item.error   { background: var(--danger-bg);  border-color: var(--danger-bd);  }
.issue-item.warning { background: var(--warning-bg); border-color: var(--warning-bd); }
.issue-item.info    { background: var(--info-bg);    border-color: var(--info-bd);    }
.issue-indicator {
    width: 6px; height: 6px;
    border-radius: 50%;
    margin-top: 5px;
    flex-shrink: 0;
}
.issue-indicator.error   { background: #ef4444; }
.issue-indicator.warning { background: #f59e0b; }
.issue-indicator.info    { background: #3b82f6; }
.issue-col {
    font-weight: 600;
    color: var(--text-strong);
    font-family: var(--mono);
    font-size: 0.78rem;
}
.issue-detail { color: var(--text-muted); font-size: 0.78rem; margin-top: 1px; }

/* ── Score display ────────────────────────────────────── */
.score-display {
    display: flex;
    align-items: baseline;
    gap: 0.3rem;
    margin: 0.5rem 0;
}
.score-number {
    font-size: 2.5rem;
    font-weight: 600;
    letter-spacing: -0.04em;
    font-family: var(--mono);
    line-height: 1;
}
.score-denom {
    font-size: 1rem;
    color: var(--text-faint);
    font-family: var(--mono);
}

/* ── AI output block ──────────────────────────────────── */
.ai-block {
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 0 var(--radius-md) var(--radius-md) 0;
    padding: 1rem 1.25rem;
    font-size: 0.85rem;
    line-height: 1.75;
    color: var(--text-base);
    white-space: pre-wrap;
}
.ai-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--accent);
    margin-bottom: 0.6rem;
    font-family: var(--mono);
}

/* ── Code block ───────────────────────────────────────── */
.code-block {
    background: #0d1117;
    color: #e6edf3;
    font-family: var(--mono);
    font-size: 0.78rem;
    padding: 1rem 1.25rem;
    border-radius: var(--radius-md);
    white-space: pre-wrap;
    line-height: 1.65;
    overflow-x: auto;
    border: 1px solid #21262d;
}

/* ── Log entries ──────────────────────────────────────── */
.log-table {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: hidden;
}
.log-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.6rem 1rem;
    border-bottom: 1px solid var(--border);
    font-size: 0.78rem;
}
.log-row:last-child { border-bottom: none; }
.log-row:hover { background: var(--bg-subtle); }
.log-num { color: var(--text-faint); font-family: var(--mono); min-width: 2rem; }
.log-text { flex: 1; color: var(--text-base); }
.log-text.etl  { color: var(--info); }
.log-text.err  { color: var(--danger); }
.log-text.doc  { color: #7c3aed; }
.log-time { color: var(--text-faint); font-family: var(--mono); font-size: 0.7rem; }

/* ── Empty state ──────────────────────────────────────── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3.5rem 2rem;
    text-align: center;
    background: var(--bg-subtle);
    border: 1px dashed var(--border-dark);
    border-radius: var(--radius-lg);
    gap: 0.5rem;
}
.empty-icon {
    font-size: 1.5rem;
    opacity: 0.4;
    margin-bottom: 0.25rem;
}
.empty-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-muted);
}
.empty-desc {
    font-size: 0.8rem;
    color: var(--text-faint);
    max-width: 320px;
}

/* ── Doc dictionary ───────────────────────────────────── */
.dict-row {
    display: grid;
    grid-template-columns: 180px 100px 1fr 100px;
    gap: 1rem;
    padding: 0.7rem 1rem;
    border-bottom: 1px solid var(--border);
    font-size: 0.8rem;
    align-items: start;
}
.dict-row:hover { background: var(--bg-subtle); }
.dict-header {
    background: var(--bg-subtle);
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-faint);
}
.dict-col-name {
    font-family: var(--mono);
    font-weight: 500;
    color: var(--text-strong);
    font-size: 0.78rem;
}
.dict-type {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-muted);
}
.dict-def { color: var(--text-base); line-height: 1.5; }

/* ── Pipeline progress ────────────────────────────────── */
.pipeline {
    display: flex;
    align-items: center;
    gap: 0;
    padding: 1rem 0;
}
.pipeline-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
    position: relative;
}
.pipeline-step::after {
    content: '';
    position: absolute;
    top: 12px;
    left: 50%;
    width: 100%;
    height: 2px;
    background: var(--border);
    z-index: 0;
}
.pipeline-step:last-child::after { display: none; }
.pipeline-step.done::after { background: #16a34a; }
.pipeline-dot {
    width: 24px; height: 24px;
    border-radius: 50%;
    background: var(--bg-muted);
    border: 2px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.6rem;
    font-family: var(--mono);
    font-weight: 600;
    color: var(--text-faint);
    z-index: 1;
    position: relative;
}
.pipeline-dot.done  { background: #16a34a; border-color: #16a34a; color: #fff; }
.pipeline-dot.active { background: var(--accent); border-color: var(--accent); color: #fff; }
.pipeline-label {
    font-size: 0.7rem;
    color: var(--text-faint);
    margin-top: 0.4rem;
    font-weight: 500;
}
.pipeline-label.done   { color: #16a34a; font-weight: 600; }
.pipeline-label.active { color: var(--accent); font-weight: 600; }

/* ── Streamlit overrides ──────────────────────────────── */
/* Buttons */
.stButton > button {
    font-family: var(--font) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.45rem 1rem !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    background: var(--bg) !important;
    color: var(--text-base) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.15s !important;
    height: auto !important;
    line-height: 1.4 !important;
}
.stButton > button:hover {
    border-color: var(--border-dark) !important;
    background: var(--bg-subtle) !important;
}
div[data-testid="stButton"] button[kind="primary"],
.primary-btn button {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #fff !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover,
.primary-btn button:hover {
    background: var(--accent-hover) !important;
    border-color: var(--accent-hover) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 4px !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    background: transparent !important;
    padding: 0.4rem 0.9rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg) !important;
    color: var(--text-strong) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* Inputs */
.stTextInput input, .stTextArea textarea, .stSelectbox select {
    font-family: var(--font) !important;
    font-size: 0.85rem !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-base) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(28,100,242,0.08) !important;
    outline: none !important;
}

/* Labels */
.stTextInput label, .stTextArea label, .stSelectbox label,
.stRadio label, .stCheckbox label, [data-testid="stWidgetLabel"] {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--text-base) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg-subtle) !important;
    transition: border-color 0.15s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] { font-size: 0.72rem !important; color: var(--text-faint) !important; }
[data-testid="stMetricValue"] { font-family: var(--mono) !important; font-size: 1.5rem !important; color: var(--text-strong) !important; }

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg) !important;
}

/* Radio / checkbox */
.stRadio [data-baseweb="radio"] { font-size: 0.82rem !important; }

/* Alerts */
.stAlert {
    border-radius: var(--radius-md) !important;
    font-size: 0.82rem !important;
}

/* Spinner */
.stSpinner { font-size: 0.82rem !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-dark); border-radius: 3px; }

/* Data editor */
[data-testid="stDataEditor"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}

/* Download button */
[data-testid="stDownloadButton"] button {
    font-size: 0.82rem !important;
    border-radius: var(--radius-sm) !important;
}
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & PALETTE
# ─────────────────────────────────────────────────────────────────────────────

CHART_COLORS = [
    "#1c64f2", "#7e3af2", "#057a55", "#c27803",
    "#c81e1e", "#1e96c8", "#6c2bd9", "#047481",
]

CHART_LAYOUT = dict(
    font_family="Geist, -apple-system, sans-serif",
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=12, r=12, t=36, b=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=11)),
    title_font=dict(size=13, color="#111827"),
    colorway=CHART_COLORS,
)

TABS = ["Overview", "Data Source", "Data Quality", "Transformation", "Analytics", "Documentation"]

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

_STATE_DEFAULTS = {
    "tab": "Overview",
    "df": None,
    "df_original": None,
    "filename": None,
    "clean_log": [],
    "quality": None,
    "ai_diagnosis": None,
    "ai_suggestions": [],
    "kpis": [],
    "charts": [],
    "exec_report": None,
    "etl_sql": None,
    "etl_preview": None,
    "etl_plan": None,
    "custom_chart": None,
    "doc_dictionary": None,
    "doc_changelog": None,
    "doc_readme": None,
}

for k, v in _STATE_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def set_tab(name: str):
    st.session_state["tab"] = name


# ─────────────────────────────────────────────────────────────────────────────
# API CLIENT
# ─────────────────────────────────────────────────────────────────────────────

def claude(prompt: str, system: str, max_tokens: int = 1500) -> str:
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def parse_json(text: str):
    m = re.search(r"```json\s*([\s\S]+?)\s*```", text)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    try: return json.loads(text)
    except: return None


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_file(f) -> pd.DataFrame:
    return pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)


def load_sql(conn_str: str, query: str) -> pd.DataFrame:
    engine = sqlalchemy.create_engine(conn_str)
    with engine.connect() as c:
        return pd.read_sql(query, c)


def register_dataset(df: pd.DataFrame, filename: str):
    st.session_state.update({
        "df": df.copy(), "df_original": df.copy(), "filename": filename,
        "clean_log": [], "quality": None, "ai_diagnosis": None,
        "ai_suggestions": [], "kpis": [], "charts": [],
        "exec_report": None, "doc_dictionary": None,
        "doc_changelog": None, "doc_readme": None,
    })


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_quality(df: pd.DataFrame) -> dict:
    issues, score = [], 100
    null_counts = df.isnull().sum()

    for col, n in null_counts.items():
        if n > 0:
            pct = round(n / len(df) * 100, 1)
            sev = "error" if pct > 20 else "warning"
            issues.append({"column": col, "type": "Missing values",
                           "detail": f"{n} null values ({pct}%)", "severity": sev})
            score -= min(15, pct * 0.4)

    dup = int(df.duplicated().sum())
    if dup > 0:
        pct = round(dup / len(df) * 100, 1)
        issues.append({"column": "— all rows —", "type": "Duplicate rows",
                       "detail": f"{dup} duplicates ({pct}%)",
                       "severity": "error" if pct > 5 else "warning"})
        score -= min(20, pct * 2)

    for col in df.select_dtypes(include="object").columns:
        if pd.to_numeric(df[col].dropna(), errors="coerce").notna().mean() > 0.7:
            issues.append({"column": col, "type": "Type mismatch",
                           "detail": "Numeric values stored as text", "severity": "warning"})
            score -= 5

    for col in df.select_dtypes(include="number").columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            n_out = int(((df[col] < q1 - 3*iqr) | (df[col] > q3 + 3*iqr)).sum())
            if n_out > 0:
                issues.append({"column": col, "type": "Statistical outliers",
                               "detail": f"{n_out} values beyond 3×IQR", "severity": "info"})

    for col in df.select_dtypes(include="object").columns:
        ws = int(df[col].dropna().apply(lambda x: str(x) != str(x).strip()).sum())
        if ws > 0:
            issues.append({"column": col, "type": "Whitespace",
                           "detail": f"{ws} values with leading/trailing spaces", "severity": "info"})

    return {
        "score": max(0, round(score)),
        "issues": issues,
        "null_total": int(null_counts.sum()),
        "dup_total": dup,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLEANING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def apply_ops(df: pd.DataFrame, operations: list) -> tuple:
    df, log = df.copy(), []
    for op in operations:
        action, col = op.get("action"), op.get("column")
        try:
            if action == "drop_nulls" and col:
                b = len(df); df = df.dropna(subset=[col])
                log.append(f"Dropped {b - len(df)} rows with nulls in {col}")

            elif action == "fill_nulls" and col:
                m, v = op.get("method", "value"), op.get("value")
                if m == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                    val = df[col].mean(); df[col] = df[col].fillna(val)
                    log.append(f"Filled nulls in {col} with mean ({round(val, 2)})")
                elif m == "median" and pd.api.types.is_numeric_dtype(df[col]):
                    val = df[col].median(); df[col] = df[col].fillna(val)
                    log.append(f"Filled nulls in {col} with median ({round(val, 2)})")
                elif m == "mode":
                    val = df[col].mode()[0]; df[col] = df[col].fillna(val)
                    log.append(f"Filled nulls in {col} with mode")
                else:
                    df[col] = df[col].fillna(v)
                    log.append(f"Filled nulls in {col} with constant '{v}'")

            elif action == "drop_duplicates":
                b = len(df); df = df.drop_duplicates()
                log.append(f"Removed {b - len(df)} duplicate rows")

            elif action == "trim_whitespace" and col:
                df[col] = df[col].str.strip()
                log.append(f"Trimmed whitespace in {col}")

            elif action == "convert_type" and col:
                t = op.get("target_type", "numeric")
                if t == "numeric": df[col] = pd.to_numeric(df[col], errors="coerce")
                elif t == "datetime": df[col] = pd.to_datetime(df[col], errors="coerce")
                else: df[col] = df[col].astype(str)
                log.append(f"Converted {col} to {t}")

            elif action == "rename_column" and col:
                nn = op.get("new_name")
                df = df.rename(columns={col: nn})
                log.append(f"Renamed {col} to {nn}")

            elif action == "drop_column" and col:
                df = df.drop(columns=[col])
                log.append(f"Dropped column {col}")

            elif action == "cap_outliers" and col:
                if pd.api.types.is_numeric_dtype(df[col]):
                    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                    iqr = q3 - q1; lo, hi = q1 - 3*iqr, q3 + 3*iqr
                    b = ((df[col] < lo) | (df[col] > hi)).sum()
                    df[col] = df[col].clip(lower=lo, upper=hi)
                    log.append(f"Capped {b} outliers in {col}")

            elif action == "uppercase" and col:
                df[col] = df[col].str.upper(); log.append(f"Uppercased {col}")

            elif action == "lowercase" and col:
                df[col] = df[col].str.lower(); log.append(f"Lowercased {col}")

            elif action == "replace_value" and col:
                ov, nv = op.get("old_value"), op.get("new_value")
                df[col] = df[col].replace(ov, nv)
                log.append(f"Replaced '{ov}' with '{nv}' in {col}")

        except Exception as e:
            log.append(f"[ERROR] {action} on {col}: {e}")

    return df, log


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_chart(cfg: dict, df: pd.DataFrame):
    x = cfg.get("x") if cfg.get("x") in df.columns else None
    y = cfg.get("y") if cfg.get("y") in df.columns else None
    color = cfg.get("color") if cfg.get("color") in df.columns else None
    agg, ctype, title = cfg.get("agg", "none"), cfg.get("type", "bar"), cfg.get("title", "")
    plot_df = df.copy()
    if agg != "none" and x and y:
        plot_df = getattr(plot_df.groupby(x)[y], agg)().reset_index()
        color = None
    kw = dict(color_discrete_sequence=CHART_COLORS, title=title, height=320)
    fig = None
    try:
        if ctype == "bar":       fig = px.bar(plot_df, x=x, y=y, color=color, **kw)
        elif ctype == "line":    fig = px.line(plot_df, x=x, y=y, color=color, **kw)
        elif ctype == "area":    fig = px.area(plot_df, x=x, y=y, color=color, **kw)
        elif ctype == "scatter": fig = px.scatter(plot_df, x=x, y=y, color=color, **kw)
        elif ctype == "pie":     fig = px.pie(plot_df, names=x, values=y, color_discrete_sequence=CHART_COLORS, title=title, height=320)
        elif ctype == "histogram": fig = px.histogram(plot_df, x=x or y, color=color, **kw)
        elif ctype == "box":     fig = px.box(plot_df, x=x, y=y, color=color, **kw)
        if fig:
            fig.update_layout(**CHART_LAYOUT)
            fig.update_xaxes(showgrid=False, linecolor="#e5e7eb", tickfont=dict(size=11))
            fig.update_yaxes(showgrid=True, gridcolor="#f3f4f6", linecolor="#e5e7eb", tickfont=dict(size=11))
    except: pass
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def render_badge(text: str, variant: str = "neutral") -> str:
    return f'<span class="badge badge-{variant}">{text}</span>'


def render_kpi_card(label: str, value: str, sub: str = "", delta: str = "",
                    delta_dir: str = "flat") -> str:
    delta_html = f'<div class="kpi-card-delta delta-{delta_dir}">{delta}</div>' if delta else ""
    sub_html = f'<div class="kpi-card-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-card-label">{label}</div>
        <div class="kpi-card-value">{value}</div>
        {delta_html}{sub_html}
    </div>"""


def render_issue(issue: dict) -> str:
    sev = issue["severity"]
    return f"""
    <div class="issue-item {sev}">
        <div class="issue-indicator {sev}"></div>
        <div>
            <div class="issue-col">{issue['column']}</div>
            <div class="issue-detail">{issue['type']} — {issue['detail']}</div>
        </div>
    </div>"""


def render_log_row(idx: int, text: str, ts: str) -> str:
    cls = "err" if "[ERROR]" in text else ("etl" if "[ETL]" in text else ("doc" if "[DOC]" in text else ""))
    return f"""
    <div class="log-row">
        <span class="log-num">{idx:02d}</span>
        <span class="log-text {cls}">{text}</span>
        <span class="log-time">{ts}</span>
    </div>"""


def render_section_header(title: str, desc: str = ""):
    desc_html = f'<p class="section-desc">{desc}</p>' if desc else ""
    st.markdown(f"""
    <div class="section-header">
        <h2 class="section-title">{title}</h2>
        {desc_html}
    </div>""", unsafe_allow_html=True)


def render_empty_state(title: str, desc: str = "", icon: str = "○"):
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-icon">{icon}</div>
        <div class="empty-title">{title}</div>
        <div class="empty-desc">{desc}</div>
    </div>""", unsafe_allow_html=True)


def render_divider():
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def tab_overview():
    df = st.session_state["df"]
    has_data = df is not None
    quality = st.session_state.get("quality")
    clean_log = st.session_state.get("clean_log", [])

    render_section_header("Overview", "Current state of your data pipeline.")

    # Pipeline progress
    steps = [
        ("Load", has_data),
        ("Quality", quality is not None),
        ("Clean", bool(clean_log)),
        ("Transform", any("[ETL]" in l for l in clean_log)),
        ("Report", st.session_state.get("exec_report") is not None),
        ("Docs", st.session_state.get("doc_dictionary") is not None),
    ]
    dots_html = ""
    for i, (lbl, done) in enumerate(steps):
        dot_cls = "done" if done else ""
        lbl_cls = "done" if done else ""
        symbol = "✓" if done else str(i + 1)
        connector = f'<div style="flex:1;height:2px;background:{"#16a34a" if done and i < len(steps)-1 else "#e5e7eb"};margin-bottom:28px;max-width:60px"></div>' if i < len(steps) - 1 else ""
        dots_html += f"""
        <div style="display:flex;flex-direction:column;align-items:center;flex:1">
            <div class="pipeline-dot {dot_cls}">{symbol}</div>
            <div class="pipeline-label {lbl_cls}">{lbl}</div>
        </div>
        {connector}"""
    st.markdown(f'<div style="display:flex;align-items:flex-start;padding:1.5rem 0 2rem">{dots_html}</div>', unsafe_allow_html=True)

    render_divider()

    # KPI grid
    if has_data:
        q = quality or {}
        score = q.get("score", "—")
        score_color = "#057a55" if isinstance(score, int) and score >= 80 else ("#c27803" if isinstance(score, int) and score >= 60 else "#c81e1e") if isinstance(score, int) else "#6b7280"

        cards_html = f"""
        <div class="kpi-grid">
            {render_kpi_card("Rows", f"{df.shape[0]:,}", f"{df.shape[1]} columns")}
            {render_kpi_card("Quality score", str(score), "out of 100" if score != "—" else "Run analysis")}
            {render_kpi_card("Issues found", str(len(q.get("issues", []))) if q else "—", "quality checks")}
            {render_kpi_card("Operations", str(len(clean_log)), "applied so far")}
        </div>"""
        st.markdown(cards_html, unsafe_allow_html=True)

        # Quick actions
        render_section_header("Quick actions")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("Go to Data Source", use_container_width=True):
                set_tab("Data Source"); st.rerun()
        with c2:
            if st.button("Run quality analysis", use_container_width=True):
                set_tab("Data Quality"); st.rerun()
        with c3:
            if st.button("Open transformations", use_container_width=True):
                set_tab("Transformation"); st.rerun()
        with c4:
            if st.button("View analytics", use_container_width=True):
                set_tab("Analytics"); st.rerun()

        # Dataset summary
        render_divider()
        render_section_header("Dataset summary")
        with st.expander("Column overview", expanded=False):
            summary_data = []
            for col in df.columns:
                null_pct = round(df[col].isnull().mean() * 100, 1)
                summary_data.append({
                    "Column": col,
                    "Type": str(df[col].dtype),
                    "Non-null": f"{df[col].notna().sum():,}",
                    "Null %": f"{null_pct}%",
                    "Unique": f"{df[col].nunique():,}",
                })
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    else:
        render_empty_state(
            "No dataset loaded",
            "Go to Data Source to upload a CSV, Excel file, or connect a SQL database.",
            "○"
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Load data"):
            set_tab("Data Source"); st.rerun()


def tab_data_source():
    render_section_header("Data source", "Upload a file or connect to a database to get started.")

    t_file, t_sql = st.tabs(["File upload", "SQL connection"])

    with t_file:
        st.markdown("<br>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop a CSV or Excel file here, or click to browse",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
        )
        if uploaded:
            try:
                df = load_file(uploaded)
                register_dataset(df, uploaded.name)
                st.success(f"Loaded {uploaded.name} — {df.shape[0]:,} rows, {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Could not read file: {e}")

    with t_sql:
        st.markdown("<br>", unsafe_allow_html=True)
        conn_str = st.text_input(
            "Connection string",
            placeholder="postgresql://user:password@host:5432/database",
        )
        query = st.text_area("SQL query", placeholder="SELECT * FROM table LIMIT 10000", height=100)
        if st.button("Connect and load"):
            try:
                with st.spinner("Connecting..."):
                    df = load_sql(conn_str, query)
                register_dataset(df, "sql_query.csv")
                st.success(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Connection failed: {e}")

    df = st.session_state["df"]
    if df is not None:
        render_divider()
        render_section_header("Data preview", f"{st.session_state['filename']} — {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.dataframe(df.head(50), use_container_width=True, hide_index=True)

        render_divider()
        render_section_header("Statistical summary")
        st.dataframe(df.describe().round(3), use_container_width=True)


def tab_data_quality():
    df = st.session_state["df"]
    if df is None:
        render_empty_state("No dataset loaded", "Load data in the Data Source tab first.")
        return

    render_section_header("Data quality", "Automated profiling and AI-powered diagnosis of your dataset.")

    quality = st.session_state.get("quality")

    if not quality:
        if st.button("Run quality analysis", type="primary"):
            with st.spinner("Profiling dataset..."):
                q = run_quality(df)
                st.session_state["quality"] = q

            with st.spinner("Generating AI diagnosis..."):
                diag = claude(
                    f"Shape: {df.shape}\nColumns: {list(df.columns)}\n"
                    f"Issues: {json.dumps(q['issues'])}\nSample:\n{df.head(3).to_string()}",
                    "You are a senior data analyst. Write a 3-paragraph quality diagnosis. "
                    "P1: what this dataset represents and overall health. "
                    "P2: the most critical issues and their business impact. "
                    "P3: prioritized actionable recommendations. "
                    "Max 200 words. Be specific and professional.",
                )
                st.session_state["ai_diagnosis"] = diag

            with st.spinner("Generating cleaning suggestions..."):
                raw = claude(
                    f"Columns: {list(df.columns)}\nTypes: {df.dtypes.to_dict()}\n"
                    f"Issues: {json.dumps(q['issues'])}\nSample:\n{df.head(3).to_string()}",
                    "Return ONLY a JSON array of cleaning operations — no markdown, no explanation:\n"
                    '[{"action":str,"column":str,"description":str,"method":str,"value":str}]\n'
                    "Valid actions: drop_nulls, fill_nulls, drop_duplicates, trim_whitespace, "
                    "convert_type, rename_column, drop_column, cap_outliers, uppercase, lowercase. "
                    "Only include actions that are clearly warranted.",
                )
                sugg = parse_json(raw)
                st.session_state["ai_suggestions"] = sugg if isinstance(sugg, list) else []

            st.rerun()
    else:
        if st.button("Re-run analysis"):
            st.session_state["quality"] = None
            st.session_state["ai_diagnosis"] = None
            st.session_state["ai_suggestions"] = []
            st.rerun()

    quality = st.session_state.get("quality")
    if not quality:
        return

    # Score + stats
    score = quality["score"]
    score_color = "#057a55" if score >= 80 else ("#c27803" if score >= 60 else "#c81e1e")
    sev_label = "Good" if score >= 80 else ("Fair" if score >= 60 else "Poor")
    badge_variant = "success" if score >= 80 else ("warning" if score >= 60 else "danger")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-card-label">Quality score</div>
            <div class="score-display">
                <span class="score-number" style="color:{score_color}">{score}</span>
                <span class="score-denom">/ 100</span>
            </div>
            {render_badge(sev_label, badge_variant)}
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(render_kpi_card("Missing values", f"{quality['null_total']:,}", "across all columns"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_kpi_card("Duplicate rows", f"{quality['dup_total']:,}", "exact matches"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_kpi_card("Checks flagged", str(len(quality["issues"])), "total issues"), unsafe_allow_html=True)

    # Issues
    if quality["issues"]:
        render_divider()
        render_section_header("Issues detected")
        issues_html = '<div class="issue-list">' + "".join(render_issue(i) for i in quality["issues"]) + '</div>'
        st.markdown(issues_html, unsafe_allow_html=True)

    # AI diagnosis
    diag = st.session_state.get("ai_diagnosis")
    if diag:
        render_divider()
        render_section_header("AI diagnosis")
        st.markdown(f'<div class="ai-block"><div class="ai-label">Analysis by Claude</div>{diag}</div>', unsafe_allow_html=True)

    # Cleaning panel
    render_divider()
    render_section_header("Cleaning operations", "Apply fixes to your dataset. All changes are logged and reversible.")

    ct = st.tabs(["AI suggestions", "Manual operations", "Formula", "Direct editing"])

    with ct[0]:
        sugg = st.session_state.get("ai_suggestions", [])
        if sugg:
            st.markdown("<br>", unsafe_allow_html=True)
            selected = []
            for i, s in enumerate(sugg):
                col_a, _ = st.columns([6, 1])
                with col_a:
                    action_label = s.get("action", "").replace("_", " ").title()
                    col_name = s.get("column", "")
                    desc = s.get("description", "")
                    if st.checkbox(f"**{action_label}** on `{col_name}` — {desc}", key=f"s_{i}", value=True):
                        selected.append(s)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Apply selected operations", type="primary"):
                nd, entries = apply_ops(df, selected)
                st.session_state["df"] = nd
                st.session_state["clean_log"].extend(entries)
                st.success(f"Applied {len(entries)} operations successfully.")
                st.rerun()
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("Run quality analysis to receive AI-generated suggestions.")

    with ct[1]:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("manual_ops"):
            r1, r2 = st.columns(2)
            with r1:
                action = st.selectbox("Operation", [
                    "fill_nulls", "drop_nulls", "drop_duplicates", "trim_whitespace",
                    "convert_type", "rename_column", "drop_column",
                    "cap_outliers", "uppercase", "lowercase", "replace_value",
                ])
            with r2:
                column = st.selectbox("Target column", ["(all)"] + list(df.columns))

            c1, c2, c3 = st.columns(3)
            with c1: method = st.selectbox("Fill method", ["mean", "median", "mode", "constant"])
            with c2: fill_val = st.text_input("Constant value", "")
            with c3: new_name = st.text_input("New column name", "")

            c4, c5 = st.columns(2)
            with c4: old_val = st.text_input("Old value (replace)", "")
            with c5: new_val = st.text_input("New value (replace)", "")

            target_type = st.selectbox("Convert to type", ["numeric", "datetime", "string"])

            if st.form_submit_button("Apply operation", type="primary"):
                op = {
                    "action": action,
                    "column": None if column == "(all)" else column,
                    "method": method.replace("constant", "value"),
                    "value": fill_val or None,
                    "new_name": new_name or None,
                    "old_value": old_val or None,
                    "new_value": new_val or None,
                    "target_type": target_type,
                }
                nd, entries = apply_ops(df, [op])
                st.session_state["df"] = nd
                st.session_state["clean_log"].extend(entries)
                for e in entries:
                    st.success(f"Done: {e}")
                st.rerun()

    with ct[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        fcol = st.selectbox("Target column", list(df.columns), key="formula_col")
        fexpr = st.text_area("Expression (pandas eval syntax)",
                             placeholder="e.g.   revenue * 0.2   or   price / quantity",
                             height=80, label_visibility="visible")
        if st.button("Apply formula", type="primary"):
            try:
                nd = df.copy()
                nd[fcol] = df.eval(fexpr)
                st.session_state["df"] = nd
                st.session_state["clean_log"].append(f"Formula on {fcol}: {fexpr}")
                st.success("Formula applied.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    with ct[3]:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Edit cells directly. Click Save to apply changes.")
        edited = st.data_editor(df, use_container_width=True, num_rows="dynamic", key="cell_editor")
        if st.button("Save cell edits", type="primary"):
            st.session_state["df"] = edited
            st.session_state["clean_log"].append("Manual cell edits applied")
            st.success("Changes saved.")
            st.rerun()

    # Transformation log
    clog = st.session_state["clean_log"]
    if clog:
        render_divider()
        render_section_header("Operation log")
        ts = datetime.now().strftime("%H:%M")
        rows_html = "".join(render_log_row(i + 1, e, ts) for i, e in enumerate(clog))
        st.markdown(f'<div class="log-table">{rows_html}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        rc, _ = st.columns([1, 5])
        with rc:
            if st.button("Reset to original dataset"):
                st.session_state["df"] = st.session_state["df_original"].copy()
                st.session_state["clean_log"] = []
                st.session_state["quality"] = None
                st.rerun()

    # Export
    dfc = st.session_state["df"]
    render_divider()
    render_section_header("Export")
    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button(
            "Download CSV",
            dfc.to_csv(index=False).encode(),
            f"cleaned_{st.session_state['filename']}.csv",
            "text/csv",
            use_container_width=True,
        )
    with e2:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            dfc.to_excel(w, index=False)
        st.download_button(
            "Download Excel",
            buf.getvalue(),
            f"cleaned_{st.session_state['filename']}.xlsx",
            use_container_width=True,
        )
    with e3:
        log_txt = "DataOps AI — Operation Log\n" + "=" * 40 + "\n"
        log_txt += "\n".join(f"{i+1}. {e}" for i, e in enumerate(clog))
        st.download_button("Download log", log_txt.encode(), "operations.txt", use_container_width=True)


def tab_transformation():
    df = st.session_state["df"]
    if df is None:
        render_empty_state("No dataset loaded", "Load data in the Data Source tab first.")
        return

    render_section_header("Transformation", "Reshape, aggregate and enrich your data.")

    et = st.tabs(["AI SQL", "Column operations", "Merge / Join", "Pivot & Reshape", "Planning"])

    with et[0]:
        st.markdown("<br>", unsafe_allow_html=True)
        schema = ", ".join([f"{c} ({df[c].dtype})" for c in df.columns])
        prompt = st.text_area(
            "Describe the transformation in plain language",
            placeholder="Examples:\n"
                        "— Calculate total revenue per region, sorted descending\n"
                        "— Keep only rows where status is 'active' and amount > 1000\n"
                        "— Add a profit_margin column = (revenue - cost) / revenue * 100",
            height=110,
        )
        if st.button("Generate code", type="primary"):
            if prompt.strip():
                with st.spinner("Generating transformation code..."):
                    raw = claude(
                        f"Schema: {schema}\nSample:\n{df.head(2).to_string()}\nRequest: {prompt}",
                        'Return ONLY a JSON object — no markdown:\n'
                        '{"sql": "pandas code using df as input variable, result must be assigned to result_df", '
                        '"explanation": "one sentence description"}',
                    )
                    parsed = parse_json(raw)
                if parsed:
                    st.session_state["etl_sql"] = parsed
                    st.markdown(f'<div class="code-block">{parsed.get("sql", "")}</div>', unsafe_allow_html=True)
                    st.caption(parsed.get("explanation", ""))
                    try:
                        g = {"df": df.copy(), "pd": pd, "np": np}
                        exec(parsed["sql"], g)
                        result = g.get("result_df", g.get("df"))
                        st.session_state["etl_preview"] = result
                        st.markdown(f"**Preview** — {result.shape[0]:,} rows × {result.shape[1]} columns")
                        st.dataframe(result.head(20), use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"Execution error: {e}")
                else:
                    st.error("Could not parse response. Try rephrasing.")

        if st.session_state.get("etl_preview") is not None:
            if st.button("Apply to dataset", type="primary"):
                st.session_state["df"] = st.session_state["etl_preview"]
                st.session_state["clean_log"].append(f"[ETL] Applied: {prompt[:80]}")
                st.session_state.pop("etl_preview", None)
                st.success("Transformation applied.")
                st.rerun()

    with et[1]:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("col_ops_form"):
            r1, r2 = st.columns(2)
            with r1:
                op = st.selectbox("Operation", [
                    "Add calculated column",
                    "Split column by delimiter",
                    "Merge two columns",
                    "Extract date part",
                    "Normalize (min-max 0–1)",
                    "Standardize (z-score)",
                    "Bin into categories",
                    "Map values",
                ])
            with r2:
                tcol = st.selectbox("Source column", list(df.columns))

            c1, c2 = st.columns(2)
            with c1: new_col_name = st.text_input("Output column name", placeholder="e.g. profit_margin")
            with c2: formula_or_delim = st.text_input("Formula / delimiter / second column", placeholder="e.g. revenue - cost")

            c3, c4 = st.columns(2)
            with c3: bins_str = st.text_input("Bin edges", placeholder="e.g. 0,100,500,1000")
            with c4: labels_str = st.text_input("Bin labels", placeholder="e.g. Low,Medium,High")

            map_json = st.text_area('Value mapping (JSON)', placeholder='{"Y": 1, "N": 0}', height=60)
            date_part = st.selectbox("Date part", ["year", "month", "day", "weekday", "quarter"])

            if st.form_submit_button("Apply", type="primary"):
                try:
                    nd = df.copy()
                    out_name = new_col_name or f"{tcol}_new"
                    entry = ""

                    if op == "Add calculated column":
                        nd[out_name] = nd.eval(formula_or_delim); entry = f"Added {out_name} = {formula_or_delim}"
                    elif op == "Split column by delimiter":
                        sp = nd[tcol].astype(str).str.split(formula_or_delim or ",", expand=True)
                        for i, c in enumerate(sp.columns): nd[f"{tcol}_part{i+1}"] = sp[c]
                        entry = f"Split {tcol} by '{formula_or_delim or ','}'"
                    elif op == "Merge two columns":
                        sec = bins_str
                        if sec in nd.columns:
                            nd[out_name] = nd[tcol].astype(str) + (formula_or_delim or " ") + nd[sec].astype(str)
                            entry = f"Merged {tcol} + {sec}"
                    elif op == "Extract date part":
                        nd[tcol] = pd.to_datetime(nd[tcol], errors="coerce")
                        nd[out_name] = getattr(nd[tcol].dt, date_part)
                        entry = f"Extracted {date_part} from {tcol}"
                    elif op == "Normalize (min-max 0–1)":
                        mn, mx = nd[tcol].min(), nd[tcol].max()
                        nd[out_name] = (nd[tcol] - mn) / (mx - mn)
                        entry = f"Normalized {tcol}"
                    elif op == "Standardize (z-score)":
                        mu, sigma = nd[tcol].mean(), nd[tcol].std()
                        nd[out_name] = (nd[tcol] - mu) / sigma
                        entry = f"Standardized {tcol}"
                    elif op == "Bin into categories":
                        edges = [float(x.strip()) for x in bins_str.split(",")]
                        lbls = [x.strip() for x in labels_str.split(",")]
                        nd[out_name] = pd.cut(nd[tcol], bins=edges,
                                              labels=lbls if len(lbls) == len(edges) - 1 else None)
                        entry = f"Binned {tcol} into categories"
                    elif op == "Map values":
                        mapping = json.loads(map_json)
                        nd[out_name] = nd[tcol].map(mapping).fillna(nd[tcol])
                        entry = f"Mapped values in {tcol}"

                    st.session_state["df"] = nd
                    st.session_state["clean_log"].append(f"[ETL] {entry}")
                    st.success(entry); st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    with et[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        sf = st.file_uploader("Upload second dataset", type=["csv", "xlsx"], key="merge_upload")
        if sf:
            df2 = load_file(sf)
            st.caption(f"Second dataset: {df2.shape[0]:,} rows × {df2.shape[1]} columns")
            st.dataframe(df2.head(5), use_container_width=True, hide_index=True)
            with st.form("merge_form"):
                c1, c2, c3 = st.columns(3)
                with c1: lk = st.selectbox("Left key", df.columns.tolist())
                with c2: rk = st.selectbox("Right key", df2.columns.tolist())
                with c3: how = st.selectbox("Join type", ["inner", "left", "right", "outer"])
                if st.form_submit_button("Merge", type="primary"):
                    try:
                        merged = pd.merge(df, df2, left_on=lk, right_on=rk, how=how)
                        st.session_state["df"] = merged
                        st.session_state["clean_log"].append(
                            f"[ETL] {how.upper()} JOIN on {lk} = {rk} → {merged.shape[0]:,} rows")
                        st.success(f"Merged: {merged.shape[0]:,} rows × {merged.shape[1]} columns")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Merge failed: {e}")

    with et[3]:
        st.markdown("<br>", unsafe_allow_html=True)
        rtype = st.radio("Operation type", ["Pivot table", "Melt (wide to long)", "Transpose"], horizontal=True)
        num_cols = df.select_dtypes(include="number").columns.tolist()

        if rtype == "Pivot table":
            with st.form("pivot_form"):
                c1, c2, c3 = st.columns(3)
                with c1: pi = st.selectbox("Index (rows)", list(df.columns))
                with c2: pc = st.selectbox("Pivot column", list(df.columns))
                with c3: pv = st.selectbox("Values", num_cols if num_cols else list(df.columns))
                agg = st.selectbox("Aggregation", ["sum", "mean", "count", "min", "max"])
                if st.form_submit_button("Create pivot", type="primary"):
                    try:
                        p = df.pivot_table(index=pi, columns=pc, values=pv, aggfunc=agg).reset_index()
                        p.columns = [str(c) for c in p.columns]
                        st.session_state["df"] = p
                        st.session_state["clean_log"].append(f"[ETL] Pivot: {pi} × {pc} ({agg} of {pv})")
                        st.success(f"Pivot: {p.shape[0]} rows × {p.shape[1]} columns"); st.rerun()
                    except Exception as e: st.error(f"Error: {e}")

        elif rtype == "Melt (wide to long)":
            with st.form("melt_form"):
                id_vars = st.multiselect("ID columns (keep)", list(df.columns))
                c1, c2 = st.columns(2)
                with c1: vn = st.text_input("Value column name", "value")
                with c2: vrn = st.text_input("Variable column name", "variable")
                if st.form_submit_button("Melt", type="primary"):
                    try:
                        m = df.melt(id_vars=id_vars, var_name=vrn, value_name=vn)
                        st.session_state["df"] = m
                        st.session_state["clean_log"].append(f"[ETL] Melt → {m.shape[0]:,} rows")
                        st.success(f"Melted: {m.shape[0]:,} rows"); st.rerun()
                    except Exception as e: st.error(f"Error: {e}")
        else:
            if st.button("Transpose dataset", type="primary"):
                t = df.T.reset_index(); t.columns = [f"col_{i}" for i in range(len(t.columns))]
                st.session_state["df"] = t
                st.session_state["clean_log"].append("[ETL] Transposed dataset")
                st.rerun()

    with et[4]:
        st.markdown("<br>", unsafe_allow_html=True)
        goal = st.text_area(
            "Describe your end goal",
            placeholder="Example: I need a monthly summary table with total revenue, average order value "
                        "and transaction count per region, ready to load into a Power BI dashboard.",
            height=110,
        )
        if st.button("Generate plan", type="primary"):
            if goal.strip():
                with st.spinner("Building transformation plan..."):
                    si = "\n".join([f"- {c}: {df[c].dtype} ({df[c].nunique()} unique values)" for c in df.columns])
                    plan = claude(
                        f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns\n"
                        f"Columns:\n{si}\nSample:\n{df.head(3).to_string()}\nGoal: {goal}",
                        "You are a senior BI consultant. Write a numbered step-by-step transformation plan. "
                        "Be specific about each operation. End with instructions for using the output in Power BI.",
                        max_tokens=1200,
                    )
                    st.session_state["etl_plan"] = plan
        if st.session_state.get("etl_plan"):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="ai-block"><div class="ai-label">Transformation plan by Claude</div>{st.session_state["etl_plan"]}</div>', unsafe_allow_html=True)


def tab_analytics():
    df = st.session_state["df"]
    if df is None:
        render_empty_state("No dataset loaded", "Load data in the Data Source tab first.")
        return

    render_section_header("Analytics", "Explore your data through interactive visualizations.")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    at = st.tabs(["KPI cards", "Smart charts", "Statistical analysis", "Report", "Custom chart"])

    with at[0]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Extract KPIs with AI", type="primary"):
            with st.spinner("Analyzing..."):
                raw = claude(
                    f"Columns: {list(df.columns)}\n"
                    f"Stats:\n{df.describe().to_string()}\n"
                    f"Sample:\n{df.head(5).to_string()}",
                    "You are a senior BI analyst. Extract 4–6 meaningful KPIs. "
                    "Return ONLY a JSON array — no markdown:\n"
                    '[{"label":str,"value":str,"delta":str,"direction":"up"|"down"|"flat","insight":str}]',
                )
                parsed = parse_json(raw)
                if isinstance(parsed, list):
                    st.session_state["kpis"] = parsed

        kpis = st.session_state.get("kpis", [])
        if kpis:
            st.markdown("<br>", unsafe_allow_html=True)
            cols = st.columns(min(len(kpis), 4))
            for i, kpi in enumerate(kpis):
                d = kpi.get("direction", "flat")
                arrow = "↑ " if d == "up" else ("↓ " if d == "down" else "")
                with cols[i % 4]:
                    st.markdown(
                        render_kpi_card(
                            kpi.get("label", ""),
                            kpi.get("value", "—"),
                            kpi.get("insight", ""),
                            f"{arrow}{kpi.get('delta', '')}",
                            d,
                        ),
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            render_empty_state("No KPIs yet", "Click 'Extract KPIs with AI' to generate insights.")

    with at[1]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Recommend charts with AI", type="primary"):
            with st.spinner("Selecting best visualizations..."):
                schema = ", ".join([f"{c} ({df[c].dtype})" for c in df.columns])
                raw = claude(
                    f"Columns: {schema}\nSample:\n{df.head(4).to_string()}",
                    "You are a data visualization expert. Recommend 4 charts for this dataset. "
                    "Return ONLY a JSON array:\n"
                    '[{"title":str,"type":"bar"|"line"|"area"|"scatter"|"pie"|"histogram"|"box",'
                    '"x":str|null,"y":str|null,"color":str|null,"agg":"sum"|"mean"|"count"|"none","rationale":str}]'
                    "\nOnly use column names that exist in the dataset.",
                )
                parsed = parse_json(raw)
                if isinstance(parsed, list):
                    st.session_state["charts"] = parsed

        charts = st.session_state.get("charts", [])
        if charts:
            for i in range(0, len(charts), 2):
                pair = charts[i:i+2]
                cols = st.columns(len(pair))
                for ci, cfg in enumerate(pair):
                    with cols[ci]:
                        fig = build_chart(cfg, df)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(cfg.get("rationale", ""))
        else:
            render_empty_state("No charts yet", "Click 'Recommend charts with AI' to generate visualizations.")

    with at[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        view = st.radio("View", ["Correlation matrix", "Distribution", "Value counts"], horizontal=True)

        if view == "Correlation matrix":
            if len(num_cols) >= 2:
                corr = df[num_cols].corr()
                fig = px.imshow(
                    corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale=[[0, "#c81e1e"], [0.5, "#f9fafb"], [1, "#1c64f2"]],
                    title="Pearson correlation matrix", zmin=-1, zmax=1,
                )
                fig.update_layout(**CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("At least 2 numeric columns are needed for a correlation matrix.")

        elif view == "Distribution":
            if num_cols:
                col_sel = st.selectbox("Select column", num_cols)
                with_box = st.checkbox("Overlay box plot", value=True)
                fig = px.histogram(
                    df, x=col_sel, marginal="box" if with_box else None,
                    color_discrete_sequence=[CHART_COLORS[0]],
                    title=f"Distribution — {col_sel}",
                )
                fig.update_layout(**CHART_LAYOUT)
                fig.update_xaxes(showgrid=False, linecolor="#e5e7eb")
                fig.update_yaxes(gridcolor="#f3f4f6", linecolor="#e5e7eb")
                st.plotly_chart(fig, use_container_width=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean", f"{df[col_sel].mean():.3f}")
                c2.metric("Median", f"{df[col_sel].median():.3f}")
                c3.metric("Std dev", f"{df[col_sel].std():.3f}")
                c4.metric("Skewness", f"{df[col_sel].skew():.3f}")
            else:
                st.info("No numeric columns available.")

        elif view == "Value counts":
            if cat_cols:
                col_sel = st.selectbox("Select column", cat_cols)
                top_n = st.slider("Top N values", 5, 30, 10)
                vc = df[col_sel].value_counts().head(top_n).reset_index()
                vc.columns = [col_sel, "count"]
                fig = px.bar(
                    vc, x=col_sel, y="count",
                    color_discrete_sequence=[CHART_COLORS[0]],
                    title=f"Top {top_n} values — {col_sel}",
                )
                fig.update_layout(**CHART_LAYOUT)
                fig.update_xaxes(showgrid=False, linecolor="#e5e7eb")
                fig.update_yaxes(gridcolor="#f3f4f6", linecolor="#e5e7eb")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categorical columns available.")

    with at[3]:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: lang = st.radio("Language", ["English", "French"], horizontal=True)
        with c2: aud = st.selectbox("Audience", ["Senior manager", "C-level / Executive", "Data team", "External client"])
        ctx = st.text_input("Context (optional)", placeholder="e.g. Monthly sales for the Morocco region, Q1 2025")

        if st.button("Generate executive report", type="primary"):
            with st.spinner("Writing report..."):
                report = claude(
                    f"Context: {ctx or 'Not provided'}\n"
                    f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n"
                    f"Columns: {list(df.columns)}\n"
                    f"Statistics:\n{df.describe().to_string()}\n"
                    f"KPIs: {json.dumps(st.session_state.get('kpis', []), ensure_ascii=False)}\n"
                    f"Operations applied:\n{chr(10).join(st.session_state['clean_log'][-10:])}",
                    f"Write a formal executive report in {lang} for a {aud}. "
                    "Structure: 1) Executive Summary, 2) Dataset Overview, 3) Key Findings, "
                    "4) Risks and Anomalies, 5) Recommendations, 6) Next Steps. "
                    "Be specific, cite numbers. Maximum 400 words.",
                    max_tokens=1500,
                )
                st.session_state["exec_report"] = report

        if st.session_state.get("exec_report"):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="ai-block"><div class="ai-label">Executive report</div>{st.session_state["exec_report"]}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button("Download report (.txt)", st.session_state["exec_report"].encode(), "executive_report.txt")

    with at[4]:
        st.markdown("<br>", unsafe_allow_html=True)
        all_cols = list(df.columns)
        with st.form("custom_chart_form"):
            r1, r2, r3 = st.columns(3)
            with r1: ct = st.selectbox("Chart type", ["Bar", "Line", "Area", "Scatter", "Pie", "Histogram", "Box"])
            with r2: cx = st.selectbox("X axis", ["(none)"] + all_cols)
            with r3: cy = st.selectbox("Y axis", ["(none)"] + all_cols)

            r4, r5, r6 = st.columns(3)
            with r4: cc = st.selectbox("Color by", ["(none)"] + all_cols)
            with r5: cagg = st.selectbox("Aggregate Y", ["none", "sum", "mean", "count", "max", "min"])
            with r6: ctitle = st.text_input("Chart title", "")

            height = st.slider("Height (px)", 300, 700, 400)
            if st.form_submit_button("Render chart", type="primary"):
                x = cx if cx != "(none)" else None
                y = cy if cy != "(none)" else None
                color = cc if cc != "(none)" else None
                pf = df.copy()
                if cagg != "none" and x and y:
                    pf = getattr(pf.groupby(x)[y], cagg)().reset_index(); color = None
                kw = dict(color_discrete_sequence=CHART_COLORS, title=ctitle, height=height)
                fig = None
                try:
                    if ct == "Bar":       fig = px.bar(pf, x=x, y=y, color=color, **kw)
                    elif ct == "Line":    fig = px.line(pf, x=x, y=y, color=color, **kw)
                    elif ct == "Area":    fig = px.area(pf, x=x, y=y, color=color, **kw)
                    elif ct == "Scatter": fig = px.scatter(pf, x=x, y=y, color=color, **kw)
                    elif ct == "Pie":     fig = px.pie(pf, names=x, values=y, color_discrete_sequence=CHART_COLORS, title=ctitle, height=height)
                    elif ct == "Histogram": fig = px.histogram(pf, x=x or y, color=color, **kw)
                    elif ct == "Box":     fig = px.box(pf, x=x, y=y, color=color, **kw)
                    if fig:
                        fig.update_layout(**CHART_LAYOUT)
                        fig.update_xaxes(showgrid=False, linecolor="#e5e7eb")
                        fig.update_yaxes(gridcolor="#f3f4f6", linecolor="#e5e7eb")
                        st.session_state["custom_chart"] = fig
                except Exception as e: st.error(f"Error: {e}")

        if st.session_state.get("custom_chart"):
            st.plotly_chart(st.session_state["custom_chart"], use_container_width=True)


def tab_documentation():
    df = st.session_state["df"]
    if df is None:
        render_empty_state("No dataset loaded", "Load data in the Data Source tab first.")
        return

    render_section_header("Documentation", "Auto-generated data dictionary, changelog and lineage.")

    dt = st.tabs(["Data dictionary", "Changelog", "README", "Export"])

    with dt[0]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Generate data dictionary", type="primary"):
            with st.spinner("Analyzing columns..."):
                profiles = [{
                    "column": c,
                    "dtype": str(df[c].dtype),
                    "null_pct": round(df[c].isnull().mean() * 100, 1),
                    "unique": df[c].nunique(),
                    "sample": [str(s) for s in df[c].dropna().head(3).tolist()],
                } for c in df.columns]

                raw = claude(
                    f"Dataset: {df.shape[0]:,} rows\n"
                    f"Profiles:\n{json.dumps(profiles, indent=2)}\n"
                    f"Sample:\n{df.head(3).to_string()}",
                    "Write a data dictionary entry for each column. Return ONLY a JSON array:\n"
                    '[{"column":str,"business_definition":str,"data_type":str,"format_notes":str,'
                    '"quality_status":"Good"|"Warning"|"Critical","quality_note":str,"example":str}]\n'
                    "business_definition: what this column means in business terms (1 sentence). "
                    "format_notes: expected format, units, constraints. "
                    "quality_note: specific concern if status is Warning/Critical, else leave blank.",
                    max_tokens=2000,
                )
                parsed = parse_json(raw)
                if isinstance(parsed, list):
                    st.session_state["doc_dictionary"] = parsed
                else:
                    st.error("Could not generate dictionary. Try again.")

        dictionary = st.session_state.get("doc_dictionary")
        if dictionary:
            st.markdown("<br>", unsafe_allow_html=True)
            # Header row
            st.markdown("""
            <div class="log-table">
                <div class="dict-row dict-header" style="grid-template-columns:180px 90px 1fr 90px 1fr">
                    <span>Column</span>
                    <span>Type</span>
                    <span>Definition</span>
                    <span>Status</span>
                    <span>Notes</span>
                </div>""", unsafe_allow_html=True)

            for e in dictionary:
                s = e.get("quality_status", "Good")
                badge_v = "success" if s == "Good" else ("warning" if s == "Warning" else "danger")
                st.markdown(f"""
                <div class="dict-row" style="grid-template-columns:180px 90px 1fr 90px 1fr">
                    <span class="dict-col-name">{e.get('column','')}</span>
                    <span class="dict-type">{e.get('data_type','')}</span>
                    <span class="dict-def">{e.get('business_definition','')}</span>
                    <span>{render_badge(s, badge_v)}</span>
                    <span class="dict-def" style="color:#6b7280">{e.get('quality_note','') or e.get('format_notes','')}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            render_empty_state("No dictionary yet", "Click 'Generate data dictionary' to define your columns.")

    with dt[1]:
        st.markdown("<br>", unsafe_allow_html=True)
        clog = st.session_state.get("clean_log", [])
        if not clog:
            render_empty_state("No operations recorded", "Apply cleaning or transformation operations first.")
        else:
            if st.button("Generate AI summary", type="primary"):
                with st.spinner("Summarizing..."):
                    orig = st.session_state.get("df_original")
                    summary = claude(
                        "Operations applied:\n" + "\n".join(f"{i+1}. {e}" for i, e in enumerate(clog)) +
                        f"\nShape: {orig.shape if orig is not None else 'unknown'} → {df.shape}",
                        "Summarize these data transformation operations as a professional changelog. "
                        "Format: ## Summary, ## Operations (numbered list), ## Shape Changes, ## Recommendations.",
                        max_tokens=800,
                    )
                    st.session_state["doc_changelog"] = summary

            st.markdown("<br>", unsafe_allow_html=True)
            ts = datetime.now().strftime("%H:%M")
            rows = "".join(render_log_row(i + 1, e, ts) for i, e in enumerate(clog))
            st.markdown(f'<div class="log-table">{rows}</div>', unsafe_allow_html=True)

            if st.session_state.get("doc_changelog"):
                render_divider()
                st.markdown(f'<div class="ai-block"><div class="ai-label">AI changelog summary</div>{st.session_state["doc_changelog"]}</div>', unsafe_allow_html=True)

    with dt[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        ctx = st.text_input("Project context (optional)", placeholder="e.g. Monthly sales data for Morocco, used in Power BI dashboard")

        if st.button("Generate README", type="primary"):
            with st.spinner("Writing README..."):
                d = st.session_state.get("doc_dictionary", [])
                col_summary = "\n".join([f"- `{e['column']}`: {e.get('business_definition', e.get('data_type',''))}" for e in d]) if d else "\n".join([f"- `{c}`: {df[c].dtype}" for c in df.columns])
                q = st.session_state.get("quality")
                clog = st.session_state.get("clean_log", [])

                readme = claude(
                    f"File: {st.session_state.get('filename', 'dataset')}\n"
                    f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n"
                    f"Context: {ctx or 'Not provided'}\n"
                    f"Columns:\n{col_summary}\n"
                    f"Quality score: {q['score'] if q else 'N/A'}\n"
                    f"Operations applied: {len(clog)}\n"
                    f"Author: Mohammed Amine Goumri",
                    "Write a professional GitHub-ready markdown README for this dataset. "
                    "Sections: # Title, ## Overview, ## Schema (markdown table: Column | Type | Description), "
                    "## Data Quality, ## Transformations Applied, ## Usage, ## Author.",
                    max_tokens=1200,
                )
                st.session_state["doc_readme"] = readme

        if st.session_state.get("doc_readme"):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="ai-block">{st.session_state["doc_readme"]}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button("Download README.md", st.session_state["doc_readme"].encode(), "README.md", "text/markdown")
        else:
            render_empty_state("No README yet", "Click 'Generate README' to create GitHub-ready documentation.")

    with dt[3]:
        st.markdown("<br>", unsafe_allow_html=True)
        render_section_header("Export documentation package")

        hd = st.session_state.get("doc_dictionary") is not None
        hr = st.session_state.get("doc_readme") is not None
        hc = st.session_state.get("doc_changelog") is not None
        clog = st.session_state.get("clean_log", [])

        # Status summary
        items = [
            ("Data dictionary", hd, f"{len(st.session_state.get('doc_dictionary') or [])} columns"),
            ("Changelog summary", hc, f"{len(clog)} operations"),
            ("README.md", hr, "GitHub-ready"),
        ]
        for name, ready, meta in items:
            variant = "success" if ready else "neutral"
            status = "Ready" if ready else "Not generated"
            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;
            padding:0.7rem 1rem;border:1px solid #e5e7eb;border-radius:8px;margin-bottom:0.5rem;background:#fff">
                <div>
                    <div style="font-size:0.85rem;font-weight:500;color:#111827">{name}</div>
                    <div style="font-size:0.75rem;color:#6b7280;margin-top:2px">{meta}</div>
                </div>
                {render_badge(status, variant)}
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Build package
        pkg = f"# DataOps AI — Documentation Package\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        pkg += f"Dataset: {st.session_state.get('filename', 'dataset')}\n"
        pkg += f"Author: Mohammed Amine Goumri\n\n" + "=" * 60 + "\n\n"

        if hd:
            pkg += "# DATA DICTIONARY\n\n"
            for e in st.session_state["doc_dictionary"]:
                pkg += f"## {e.get('column', '')}\n"
                pkg += f"- Type: {e.get('data_type', '')}\n"
                pkg += f"- Definition: {e.get('business_definition', '')}\n"
                pkg += f"- Format: {e.get('format_notes', '')}\n"
                pkg += f"- Quality: {e.get('quality_status', '')} — {e.get('quality_note', '')}\n"
                pkg += f"- Example: {e.get('example', '')}\n\n"

        if hc:
            pkg += "\n" + "=" * 60 + "\n\n# CHANGELOG SUMMARY\n\n"
            pkg += st.session_state["doc_changelog"] + "\n\n"

        if hr:
            pkg += "\n" + "=" * 60 + "\n\n# README\n\n"
            pkg += st.session_state["doc_readme"] + "\n\n"

        if clog:
            pkg += "\n" + "=" * 60 + "\n\n# RAW OPERATION LOG\n\n"
            pkg += "\n".join(f"{i+1:03d}. {e}" for i, e in enumerate(clog))

        e1, e2, e3 = st.columns(3)
        with e1:
            st.download_button("Download full package (.txt)", pkg.encode(), "dataops_documentation.txt", use_container_width=True)
        with e2:
            if hr:
                st.download_button("Download README.md", st.session_state["doc_readme"].encode(), "README.md", "text/markdown", use_container_width=True)
        with e3:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                st.session_state["df"].to_excel(w, index=False, sheet_name="Data")
                if hd:
                    pd.DataFrame(st.session_state["doc_dictionary"]).to_excel(w, index=False, sheet_name="Dictionary")
                if clog:
                    pd.DataFrame({"operation": clog}).to_excel(w, index=False, sheet_name="Changelog")
            st.download_button(
                "Download Excel + dictionary",
                buf.getvalue(),
                f"dataops_{st.session_state.get('filename', 'data')}.xlsx",
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# APP SHELL
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(STYLES, unsafe_allow_html=True)

df = st.session_state["df"]
has_data = df is not None
quality = st.session_state.get("quality")
filename = st.session_state.get("filename", "")
n_ops = len(st.session_state.get("clean_log", []))

# Topbar
topbar_meta = f"{filename}  ·  {df.shape[0]:,} rows" if has_data else "No dataset loaded"
st.markdown(f"""
<div class="topbar">
    <div class="topbar-brand">
        <div class="topbar-brand-dot"></div>
        DataOps AI
    </div>
    <div class="topbar-meta">{topbar_meta}</div>
</div>
""", unsafe_allow_html=True)

# Tab navigation using Streamlit buttons rendered as tab bar
current_tab = st.session_state["tab"]

tab_counts = {
    "Data Quality": len(quality["issues"]) if quality else None,
    "Transformation": n_ops if n_ops else None,
}

tabs_html = '<div class="tab-nav">'
for tab in TABS:
    active_cls = "active" if current_tab == tab else ""
    count = tab_counts.get(tab)
    badge_html = f'<span class="tab-badge {active_cls}">{count}</span>' if count else ""
    tabs_html += f'<div class="tab-item {active_cls}" id="tab-{tab}">{tab}{badge_html}</div>'
tabs_html += "</div>"
st.markdown(tabs_html, unsafe_allow_html=True)

# Tab buttons (invisible, keyboard-only navigation replacement)
cols = st.columns(len(TABS))
for i, (col, tab) in enumerate(zip(cols, TABS)):
    with col:
        if st.button(tab, key=f"tab_btn_{tab}", use_container_width=True,
                     help=f"Switch to {tab}",
                     type="primary" if current_tab == tab else "secondary"):
            set_tab(tab)
            st.rerun()

# Main content
st.markdown('<div class="app-shell">', unsafe_allow_html=True)

if current_tab == "Overview":       tab_overview()
elif current_tab == "Data Source":  tab_data_source()
elif current_tab == "Data Quality": tab_data_quality()
elif current_tab == "Transformation": tab_transformation()
elif current_tab == "Analytics":    tab_analytics()
elif current_tab == "Documentation": tab_documentation()

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="border-top:1px solid #e5e7eb;padding:1.5rem 0;margin-top:3rem;
text-align:center;font-size:0.72rem;color:#9ca3af">
DataOps AI — Built by Mohammed Amine Goumri — Powered by Claude AI
</div>
""", unsafe_allow_html=True)
