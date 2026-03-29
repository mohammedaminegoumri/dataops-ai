import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import re
import anthropic
import plotly.express as px
import plotly.graph_objects as go
import sqlalchemy

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataOps AI",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.block-container { padding: 2rem 2.5rem 3rem; max-width: 1400px; }

.app-header {
    border-bottom: 2px solid #0f172a;
    padding-bottom: 1rem;
    margin-bottom: 2rem;
}

.app-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 500;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin: 0;
}

.app-subtitle {
    font-size: 0.85rem;
    color: #64748b;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 0.2rem;
}

.module-badge {
    display: inline-block;
    background: #0f172a;
    color: #f8fafc;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
}

.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    border-left: 3px solid #0f172a;
    padding-left: 0.7rem;
    margin: 1.5rem 0 1rem;
}

.quality-score-box {
    background: #0f172a;
    color: #f8fafc;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-bottom: 1rem;
}

.quality-score-number {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3.5rem;
    font-weight: 500;
    line-height: 1;
    color: #f8fafc;
}

.quality-score-label {
    font-size: 0.75rem;
    color: #94a3b8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.3rem;
    font-family: 'IBM Plex Mono', monospace;
}

.issue-card {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    background: #fff;
    border-left: 4px solid #ef4444;
}

.issue-card.warning {
    border-left-color: #f59e0b;
}

.issue-card.info {
    border-left-color: #3b82f6;
}

.issue-card.ok {
    border-left-color: #10b981;
}

.issue-col {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    font-size: 0.85rem;
    color: #0f172a;
}

.issue-desc {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 0.15rem;
}

.stat-pill {
    display: inline-block;
    background: #f1f5f9;
    color: #475569;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    margin-right: 0.3rem;
    margin-top: 0.3rem;
}

.ai-narrative {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-top: 3px solid #0f172a;
    border-radius: 0 0 10px 10px;
    padding: 1.2rem 1.5rem;
    font-size: 0.9rem;
    line-height: 1.75;
    color: #334155;
    white-space: pre-wrap;
}

.transform-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}

.transform-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
    color: #0f172a;
    margin-bottom: 0.4rem;
}

.stButton > button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    border-radius: 8px;
    border: 1.5px solid #0f172a;
    background: #0f172a;
    color: #f8fafc;
    padding: 0.5rem 1.2rem;
    transition: all 0.15s;
}

.stButton > button:hover {
    background: #1e293b;
}

.stButton > button[kind="secondary"] {
    background: transparent;
    color: #0f172a;
}

[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

.diff-added { background: #dcfce7; }
.diff-removed { background: #fee2e2; }

.step-indicator {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
    align-items: center;
}

.step {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    border: 1px solid #cbd5e1;
    color: #94a3b8;
}

.step.active {
    background: #0f172a;
    color: #f8fafc;
    border-color: #0f172a;
}

.step.done {
    background: #dcfce7;
    color: #166534;
    border-color: #86efac;
}
</style>
""", unsafe_allow_html=True)

# ── Claude client ─────────────────────────────────────────────────────────────
def get_client():
    return anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

def call_claude(prompt: str, system: str, max_tokens: int = 1500) -> str:
    client = get_client()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text

def extract_json(text: str):
    match = re.search(r"```json\s*([\s\S]+?)\s*```", text)
    if match:
        try: return json.loads(match.group(1))
        except: pass
    try: return json.loads(text)
    except: return None

# ── Data loading ──────────────────────────────────────────────────────────────
def load_csv_excel(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

def load_from_sql(conn_str: str, query: str) -> pd.DataFrame:
    engine = sqlalchemy.create_engine(conn_str)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

# ── Quality analysis ──────────────────────────────────────────────────────────
def analyze_quality(df: pd.DataFrame) -> dict:
    issues = []
    score = 100
    total_cells = len(df) * len(df.columns)

    # Nulls
    null_counts = df.isnull().sum()
    for col, n in null_counts.items():
        if n > 0:
            pct = round(n / len(df) * 100, 1)
            severity = "error" if pct > 20 else "warning"
            issues.append({
                "column": col, "type": "Missing values",
                "detail": f"{n} nulls ({pct}%)",
                "severity": severity, "count": int(n), "pct": pct
            })
            score -= min(15, pct * 0.4)

    # Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        pct = round(dup_count / len(df) * 100, 1)
        issues.append({
            "column": "ALL ROWS", "type": "Duplicate rows",
            "detail": f"{dup_count} duplicate rows ({pct}%)",
            "severity": "error" if pct > 5 else "warning", "count": int(dup_count), "pct": pct
        })
        score -= min(20, pct * 2)

    # Type issues — numeric cols stored as object
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col].dropna(), errors="coerce")
        valid_pct = converted.notna().mean()
        if valid_pct > 0.7:
            issues.append({
                "column": col, "type": "Type mismatch",
                "detail": f"{round(valid_pct*100)}% of values look numeric but column is text",
                "severity": "warning", "count": 0, "pct": 0
            })
            score -= 5

    # Outliers in numeric cols (IQR method)
    for col in df.select_dtypes(include="number").columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            outliers = df[(df[col] < q1 - 3*iqr) | (df[col] > q3 + 3*iqr)]
            if len(outliers) > 0:
                issues.append({
                    "column": col, "type": "Outliers detected",
                    "detail": f"{len(outliers)} extreme values (3×IQR rule)",
                    "severity": "info", "count": len(outliers), "pct": 0
                })

    # Whitespace / formatting in string cols
    for col in df.select_dtypes(include="object").columns:
        has_ws = df[col].dropna().apply(lambda x: str(x) != str(x).strip()).sum()
        if has_ws > 0:
            issues.append({
                "column": col, "type": "Leading/trailing spaces",
                "detail": f"{has_ws} values have extra whitespace",
                "severity": "info", "count": int(has_ws), "pct": 0
            })

    score = max(0, round(score))
    return {
        "score": score,
        "issues": issues,
        "shape": df.shape,
        "null_total": int(null_counts.sum()),
        "dup_total": int(dup_count),
    }

def get_ai_diagnosis(df: pd.DataFrame, quality: dict) -> str:
    system = (
        "You are a senior data analyst. "
        "Write a concise 3-paragraph diagnosis of this dataset's quality. "
        "Paragraph 1: overall health & what the data seems to represent. "
        "Paragraph 2: the most critical issues and their business impact. "
        "Paragraph 3: prioritized cleaning recommendations. "
        "Be specific, use column names and numbers. Max 200 words total."
    )
    prompt = (
        f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        f"Columns: {list(df.columns)}\n"
        f"Quality score: {quality['score']}/100\n"
        f"Issues found: {json.dumps(quality['issues'], indent=2)}\n"
        f"Sample data (first 3 rows):\n{df.head(3).to_string()}"
    )
    return call_claude(prompt, system)

# ── Cleaning operations ───────────────────────────────────────────────────────
def apply_cleaning(df: pd.DataFrame, operations: list) -> tuple[pd.DataFrame, list]:
    df = df.copy()
    log = []
    for op in operations:
        try:
            action = op.get("action")
            col = op.get("column")

            if action == "drop_nulls" and col:
                before = len(df)
                df = df.dropna(subset=[col])
                log.append(f"Dropped {before - len(df)} rows with nulls in '{col}'")

            elif action == "fill_nulls" and col:
                value = op.get("value")
                method = op.get("method", "value")
                if method == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                    val = df[col].mean()
                    df[col] = df[col].fillna(val)
                    log.append(f"Filled nulls in '{col}' with mean ({round(val, 2)})")
                elif method == "median" and pd.api.types.is_numeric_dtype(df[col]):
                    val = df[col].median()
                    df[col] = df[col].fillna(val)
                    log.append(f"Filled nulls in '{col}' with median ({round(val, 2)})")
                elif method == "mode":
                    val = df[col].mode()[0]
                    df[col] = df[col].fillna(val)
                    log.append(f"Filled nulls in '{col}' with mode ('{val}')")
                else:
                    df[col] = df[col].fillna(value)
                    log.append(f"Filled nulls in '{col}' with '{value}'")

            elif action == "drop_duplicates":
                before = len(df)
                df = df.drop_duplicates()
                log.append(f"Removed {before - len(df)} duplicate rows")

            elif action == "trim_whitespace" and col:
                if df[col].dtype == object:
                    df[col] = df[col].str.strip()
                    log.append(f"Trimmed whitespace in '{col}'")

            elif action == "convert_type" and col:
                target = op.get("target_type", "numeric")
                if target == "numeric":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    log.append(f"Converted '{col}' to numeric")
                elif target == "datetime":
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    log.append(f"Converted '{col}' to datetime")
                elif target == "string":
                    df[col] = df[col].astype(str)
                    log.append(f"Converted '{col}' to string")

            elif action == "rename_column" and col:
                new_name = op.get("new_name")
                df = df.rename(columns={col: new_name})
                log.append(f"Renamed '{col}' → '{new_name}'")

            elif action == "drop_column" and col:
                df = df.drop(columns=[col])
                log.append(f"Dropped column '{col}'")

            elif action == "cap_outliers" and col:
                if pd.api.types.is_numeric_dtype(df[col]):
                    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                    iqr = q3 - q1
                    lo, hi = q1 - 3*iqr, q3 + 3*iqr
                    before = ((df[col] < lo) | (df[col] > hi)).sum()
                    df[col] = df[col].clip(lower=lo, upper=hi)
                    log.append(f"Capped {before} outliers in '{col}' to [{round(lo,2)}, {round(hi,2)}]")

            elif action == "uppercase" and col:
                df[col] = df[col].str.upper()
                log.append(f"Converted '{col}' to uppercase")

            elif action == "lowercase" and col:
                df[col] = df[col].str.lower()
                log.append(f"Converted '{col}' to lowercase")

            elif action == "replace_value" and col:
                old_val = op.get("old_value")
                new_val = op.get("new_value")
                df[col] = df[col].replace(old_val, new_val)
                log.append(f"Replaced '{old_val}' → '{new_val}' in '{col}'")

            elif action == "custom_formula" and col:
                formula = op.get("formula", "")
                df[col] = df.eval(formula)
                log.append(f"Applied formula to '{col}': {formula}")

        except Exception as e:
            log.append(f"[ERROR] {op.get('action')} on '{op.get('column')}': {str(e)}")

    return df, log

def get_ai_cleaning_suggestions(df: pd.DataFrame, quality: dict) -> list:
    system = (
        "You are a data cleaning expert. "
        "Based on the dataset issues, suggest specific cleaning operations. "
        "Return ONLY a JSON array — no markdown, no explanation:\n"
        '[{"action": str, "column": str, "description": str, "method": str, "value": str}]\n'
        "Possible actions: drop_nulls, fill_nulls, drop_duplicates, trim_whitespace, "
        "convert_type, rename_column, drop_column, cap_outliers, uppercase, lowercase.\n"
        "For fill_nulls include method: mean|median|mode|value. "
        "For convert_type include target_type: numeric|datetime|string. "
        "Only suggest what's clearly needed."
    )
    prompt = (
        f"Columns: {list(df.columns)}\n"
        f"Dtypes: {df.dtypes.to_dict()}\n"
        f"Issues: {json.dumps(quality['issues'], indent=2)}\n"
        f"Sample: {df.head(3).to_string()}"
    )
    raw = call_claude(prompt, system)
    parsed = extract_json(raw)
    return parsed if isinstance(parsed, list) else []

# ── Header ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    <div class="app-header">
        <div class="app-title">⚙ DataOps AI</div>
        <div class="app-subtitle">// BI Automation Agent — built by Mohammed Amine Goumri</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:right;font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#94a3b8;margin-top:0.8rem">
        MODULE 01<br>Data Quality & Cleaning
    </div>
    """, unsafe_allow_html=True)

# ── Step indicator ────────────────────────────────────────────────────────────
step = st.session_state.get("step", 1)
steps_html = ""
labels = ["01 · Load", "02 · Diagnose", "03 · Clean", "04 · Export"]
for i, label in enumerate(labels, 1):
    cls = "done" if step > i else ("active" if step == i else "step")
    steps_html += f'<div class="step {cls}">{label}</div>'
    if i < 4: steps_html += '<div style="color:#cbd5e1;font-size:0.8rem">→</div>'
st.markdown(f'<div class="step-indicator">{steps_html}</div>', unsafe_allow_html=True)

# ── Step 1: Load Data ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Data source</div>', unsafe_allow_html=True)

source = st.radio("Choose input", ["CSV / Excel file", "SQL Database"], horizontal=True, label_visibility="collapsed")

df_raw = None

if source == "CSV / Excel file":
    uploaded = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    if uploaded:
        try:
            df_raw = load_csv_excel(uploaded)
            st.session_state["filename"] = uploaded.name
        except Exception as e:
            st.error(f"Could not read file: {e}")

else:
    with st.expander("SQL Connection", expanded=True):
        conn_str = st.text_input("Connection string", placeholder="postgresql://user:pass@host:5432/dbname")
        query = st.text_area("SQL Query", placeholder="SELECT * FROM your_table LIMIT 5000", height=80)
        if st.button("Connect & Load"):
            try:
                df_raw = load_from_sql(conn_str, query)
                st.session_state["filename"] = "sql_query"
                st.success(f"Loaded {len(df_raw):,} rows from database")
            except Exception as e:
                st.error(f"Connection error: {e}")

# ── Store df in session state ─────────────────────────────────────────────────
if df_raw is not None:
    if "df" not in st.session_state or st.session_state.get("filename") != st.session_state.get("last_file"):
        st.session_state["df"] = df_raw.copy()
        st.session_state["df_original"] = df_raw.copy()
        st.session_state["last_file"] = st.session_state.get("filename")
        st.session_state["clean_log"] = []
        st.session_state["step"] = 1
        st.session_state.pop("quality", None)
        st.session_state.pop("ai_diagnosis", None)
        st.session_state.pop("ai_suggestions", None)

df = st.session_state.get("df")

if df is None:
    st.markdown("""
    <div style="margin-top:2rem;padding:2.5rem;background:#f8fafc;border:1px dashed #cbd5e1;
    border-radius:12px;text-align:center;color:#94a3b8;font-family:'IBM Plex Mono',monospace">
        <div style="font-size:1.8rem;margin-bottom:0.5rem">◻</div>
        <div style="font-size:0.85rem">No data loaded — upload a file or connect a database</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Preview
with st.expander(f"Data preview — {df.shape[0]:,} rows × {df.shape[1]} columns", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.caption(f"Rows: **{df.shape[0]:,}**")
    with col_b:
        st.caption(f"Columns: **{df.shape[1]}**")
    with col_c:
        st.caption(f"Memory: **{round(df.memory_usage(deep=True).sum()/1024/1024, 2)} MB**")

# ── Step 2: Diagnose ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Quality diagnosis</div>', unsafe_allow_html=True)

if st.button("Run AI Quality Analysis", use_container_width=False):
    with st.spinner("Analyzing data quality..."):
        quality = analyze_quality(df)
        st.session_state["quality"] = quality
    with st.spinner("Writing AI diagnosis..."):
        diagnosis = get_ai_diagnosis(df, quality)
        st.session_state["ai_diagnosis"] = diagnosis
    with st.spinner("Generating cleaning suggestions..."):
        suggestions = get_ai_cleaning_suggestions(df, quality)
        st.session_state["ai_suggestions"] = suggestions
    st.session_state["step"] = 2

quality = st.session_state.get("quality")
diagnosis = st.session_state.get("ai_diagnosis")
suggestions = st.session_state.get("ai_suggestions", [])

if quality:
    # Score + stats
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    score = quality["score"]
    score_color = "#10b981" if score >= 80 else ("#f59e0b" if score >= 60 else "#ef4444")

    with c1:
        st.markdown(f"""
        <div class="quality-score-box">
            <div class="quality-score-number" style="color:{score_color}">{score}</div>
            <div class="quality-score-label">Quality score / 100</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="background:#fef3c7;border-radius:10px;padding:1.2rem;text-align:center;height:100%">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:2rem;color:#92400e">{quality['null_total']:,}</div>
            <div style="font-size:0.72rem;color:#92400e;text-transform:uppercase;letter-spacing:0.08em">Missing values</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div style="background:#fee2e2;border-radius:10px;padding:1.2rem;text-align:center;height:100%">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:2rem;color:#991b1b">{quality['dup_total']:,}</div>
            <div style="font-size:0.72rem;color:#991b1b;text-transform:uppercase;letter-spacing:0.08em">Duplicate rows</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div style="background:#eff6ff;border-radius:10px;padding:1.2rem;text-align:center;height:100%">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:2rem;color:#1e40af">{len(quality['issues'])}</div>
            <div style="font-size:0.72rem;color:#1e40af;text-transform:uppercase;letter-spacing:0.08em">Issues found</div>
        </div>
        """, unsafe_allow_html=True)

    # Issues list
    if quality["issues"]:
        st.markdown('<div class="section-title">Issues detected</div>', unsafe_allow_html=True)
        sev_map = {"error": ("error", "🔴"), "warning": ("warning", "🟡"), "info": ("info", "🔵"), "ok": ("ok", "🟢")}
        for issue in quality["issues"]:
            sev, icon = sev_map.get(issue["severity"], ("info", "⚪"))
            st.markdown(f"""
            <div class="issue-card {sev}">
                <div class="issue-col">{icon} {issue['column']} — {issue['type']}</div>
                <div class="issue-desc">{issue['detail']}</div>
            </div>
            """, unsafe_allow_html=True)

    # AI narrative
    if diagnosis:
        st.markdown('<div class="section-title">AI diagnosis</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ai-narrative">{diagnosis}</div>', unsafe_allow_html=True)

# ── Step 3: Clean ─────────────────────────────────────────────────────────────
if quality:
    st.markdown('<div class="section-title">Data cleaning</div>', unsafe_allow_html=True)

    tabs = st.tabs(["AI suggestions", "Manual operations", "Custom formula", "Edit cells directly"])

    # ── Tab 1: AI suggestions ────────────────────────────────────────────────
    with tabs[0]:
        if suggestions:
            st.markdown("AI recommends these cleaning operations. Select what to apply:")
            selected_ops = []
            for i, s in enumerate(suggestions):
                col_a, col_b = st.columns([5, 1])
                with col_a:
                    checked = st.checkbox(
                        f"**{s.get('action','').replace('_',' ').title()}** on `{s.get('column','')}`  \n{s.get('description','')}",
                        key=f"sug_{i}", value=True
                    )
                if checked:
                    selected_ops.append(s)

            if st.button("Apply selected AI suggestions"):
                new_df, log = apply_cleaning(df, selected_ops)
                st.session_state["df"] = new_df
                st.session_state["clean_log"].extend(log)
                st.session_state["step"] = 3
                st.success(f"Applied {len(log)} operations")
                for entry in log:
                    st.caption(f"✓ {entry}")
                st.rerun()
        else:
            st.info("Run the quality analysis first to get AI suggestions.")

    # ── Tab 2: Manual operations ─────────────────────────────────────────────
    with tabs[1]:
        st.markdown("Build your own cleaning operations:")
        cols = list(df.columns)

        with st.form("manual_op"):
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                action = st.selectbox("Action", [
                    "fill_nulls", "drop_nulls", "drop_duplicates",
                    "trim_whitespace", "convert_type", "rename_column",
                    "drop_column", "cap_outliers", "uppercase", "lowercase",
                    "replace_value"
                ])
            with r1c2:
                column = st.selectbox("Column", ["(all columns)"] + cols)

            r2c1, r2c2, r2c3 = st.columns(3)
            with r2c1:
                method = st.selectbox("Method (for fill_nulls)", ["mean", "median", "mode", "value"])
            with r2c2:
                fill_value = st.text_input("Value (if method = value)", "")
            with r2c3:
                new_name = st.text_input("New name (for rename)", "")

            old_val = st.text_input("Old value (for replace_value)", "")
            new_val = st.text_input("New value (for replace_value)", "")
            target_type = st.selectbox("Target type (for convert_type)", ["numeric", "datetime", "string"])

            if st.form_submit_button("Apply operation"):
                op = {
                    "action": action,
                    "column": None if column == "(all columns)" else column,
                    "method": method,
                    "value": fill_value or None,
                    "new_name": new_name or None,
                    "old_value": old_val or None,
                    "new_value": new_val or None,
                    "target_type": target_type,
                }
                new_df, log = apply_cleaning(df, [op])
                st.session_state["df"] = new_df
                st.session_state["clean_log"].extend(log)
                st.session_state["step"] = 3
                for entry in log:
                    st.success(f"✓ {entry}")
                st.rerun()

    # ── Tab 3: Custom formula ────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("Write a pandas expression to transform a column:")
        formula_col = st.selectbox("Target column", cols, key="formula_col")
        formula_expr = st.text_area(
            "Formula (use column names directly)",
            placeholder="e.g.  revenue * 1.2   or   price / quantity",
            height=80
        )
        if st.button("Apply formula"):
            try:
                new_df = df.copy()
                new_df[formula_col] = df.eval(formula_expr)
                st.session_state["df"] = new_df
                st.session_state["clean_log"].append(f"Applied formula to '{formula_col}': {formula_expr}")
                st.session_state["step"] = 3
                st.success(f"✓ Applied formula to '{formula_col}'")
                st.rerun()
            except Exception as e:
                st.error(f"Formula error: {e}")

    # ── Tab 4: Edit cells directly ───────────────────────────────────────────
    with tabs[3]:
        st.markdown("Edit cells directly in the table below. Changes are saved when you click outside a cell.")
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            key="cell_editor"
        )
        if st.button("Save cell edits"):
            st.session_state["df"] = edited_df
            st.session_state["clean_log"].append("Manual cell edits applied via table editor")
            st.session_state["step"] = 3
            st.success("✓ Cell edits saved")
            st.rerun()

    # ── Cleaning log ─────────────────────────────────────────────────────────
    clean_log = st.session_state.get("clean_log", [])
    if clean_log:
        st.markdown('<div class="section-title">Cleaning log</div>', unsafe_allow_html=True)
        for entry in clean_log:
            color = "#fee2e2" if "[ERROR]" in entry else "#dcfce7"
            icon = "✗" if "[ERROR]" in entry else "✓"
            st.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.78rem;'
                f'background:{color};padding:0.3rem 0.7rem;border-radius:4px;margin-bottom:0.3rem">'
                f'{icon} {entry}</div>',
                unsafe_allow_html=True
            )

        # Undo last operation
        c_undo, c_reset = st.columns([1, 4])
        with c_undo:
            if st.button("↩ Reset to original"):
                st.session_state["df"] = st.session_state["df_original"].copy()
                st.session_state["clean_log"] = []
                st.session_state["step"] = 2
                st.rerun()

    # ── Before / After diff ───────────────────────────────────────────────────
    df_orig = st.session_state.get("df_original")
    df_current = st.session_state.get("df")
    if df_orig is not None and df_current is not None:
        orig_shape = df_orig.shape
        curr_shape = df_current.shape
        if orig_shape != curr_shape or not df_orig.equals(df_current):
            st.markdown('<div class="section-title">Before vs after</div>', unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            with b1:
                st.caption(f"Original: {orig_shape[0]:,} rows × {orig_shape[1]} cols")
                st.dataframe(df_orig.head(10), use_container_width=True)
            with b2:
                st.caption(f"Cleaned: {curr_shape[0]:,} rows × {curr_shape[1]} cols")
                st.dataframe(df_current.head(10), use_container_width=True)

# ── Step 4: Export ────────────────────────────────────────────────────────────
df_current = st.session_state.get("df")
if df_current is not None and st.session_state.get("clean_log"):
    st.markdown('<div class="section-title">Export cleaned data</div>', unsafe_allow_html=True)

    exp1, exp2, exp3 = st.columns(3)

    with exp1:
        csv_bytes = df_current.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download CSV",
            data=csv_bytes,
            file_name=f"cleaned_{st.session_state.get('filename','data')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with exp2:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_current.to_excel(writer, index=False, sheet_name="Cleaned Data")
        st.download_button(
            "⬇ Download Excel",
            data=buffer.getvalue(),
            file_name=f"cleaned_{st.session_state.get('filename','data')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with exp3:
        log_text = "DataOps AI — Cleaning Log\n" + "="*40 + "\n"
        log_text += f"Original: {st.session_state['df_original'].shape}\n"
        log_text += f"Cleaned:  {df_current.shape}\n\n"
        log_text += "Operations applied:\n"
        for i, entry in enumerate(st.session_state["clean_log"], 1):
            log_text += f"  {i}. {entry}\n"
        st.download_button(
            "⬇ Download cleaning log",
            data=log_text.encode("utf-8"),
            file_name="cleaning_log.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.session_state["step"] = 4

# ── MODULE 2: ETL & TRANSFORMATION ───────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:1.5rem">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;background:#7c3aed;
    color:#fff;padding:0.2rem 0.6rem;border-radius:4px;letter-spacing:0.05em">MODULE 02</span>
    <span style="font-family:'IBM Plex Mono',monospace;font-size:1rem;font-weight:500;color:#0f172a">
    ETL &amp; Data Transformation</span>
</div>
""", unsafe_allow_html=True)

# Only available if data is loaded
df_etl = st.session_state.get("df")
if df_etl is None:
    st.info("Load data in Module 01 first to use ETL features.")
else:
    etl_tabs = st.tabs([
        "AI SQL Generator",
        "Column operations",
        "Merge / Join",
        "Pivot & Reshape",
        "AI transformation plan"
    ])

    # ── ETL Tab 1: AI SQL Generator ───────────────────────────────────────────
    with etl_tabs[0]:
        st.markdown('<div class="section-title">Describe what you want — get SQL</div>', unsafe_allow_html=True)
        st.caption("Describe a transformation in plain language. Claude writes the SQL and applies it.")

        col_schema = ", ".join([f"{c} ({str(df_etl[c].dtype)})" for c in df_etl.columns])

        sql_prompt = st.text_area(
            "What do you want to do?",
            placeholder="e.g. Calculate total revenue per region and sort by highest first\n"
                        "e.g. Keep only rows where status = 'active' and amount > 1000\n"
                        "e.g. Add a new column 'profit_margin' = (revenue - cost) / revenue * 100",
            height=100,
            label_visibility="collapsed"
        )

        if st.button("Generate & Preview SQL", key="gen_sql"):
            if sql_prompt.strip():
                with st.spinner("Claude is writing the SQL..."):
                    system = (
                        "You are an expert SQL analyst. The user has a pandas DataFrame. "
                        "Write a pandas query or transformation based on their request. "
                        "Return ONLY a JSON object — no markdown:\n"
                        '{"sql": "the pandas code string using df as variable name", '
                        '"explanation": "what this does in plain language", '
                        '"result_description": "what the output will look like"}\n'
                        "Use pandas syntax (df.query(), df.groupby(), df.assign(), etc). "
                        "The result should always be assigned back to a variable called result_df."
                    )
                    user_msg = f"DataFrame schema: {col_schema}\nSample (2 rows):\n{df_etl.head(2).to_string()}\n\nUser request: {sql_prompt}"
                    raw = call_claude(user_msg, system)
                    parsed = extract_json(raw)

                if parsed:
                    st.session_state["etl_sql"] = parsed
                    st.markdown(f"""
                    <div style="background:#1e1e2e;color:#cdd6f4;font-family:'IBM Plex Mono',monospace;
                    font-size:0.82rem;padding:1.2rem;border-radius:10px;white-space:pre-wrap;margin:0.8rem 0">
{parsed.get('sql','')}
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption(f"💡 {parsed.get('explanation','')}")

                    # Preview execution
                    try:
                        exec_globals = {"df": df_etl.copy(), "pd": pd, "np": np}
                        exec(parsed["sql"], exec_globals)
                        result_df = exec_globals.get("result_df", exec_globals.get("df"))
                        st.session_state["etl_preview"] = result_df
                        st.markdown(f"**Preview** — {result_df.shape[0]:,} rows × {result_df.shape[1]} cols")
                        st.dataframe(result_df.head(20), use_container_width=True)
                    except Exception as e:
                        st.error(f"Execution error: {e}")
                else:
                    st.error("Could not parse Claude's response. Try rephrasing.")

        # Apply SQL result
        if st.session_state.get("etl_preview") is not None:
            c1, c2 = st.columns([1, 4])
            with c1:
                if st.button("Apply to dataset", key="apply_sql"):
                    st.session_state["df"] = st.session_state["etl_preview"]
                    sql_code = st.session_state.get("etl_sql", {}).get("sql", "")
                    st.session_state["clean_log"].append(f"ETL SQL applied: {sql_prompt[:80]}")
                    st.session_state.pop("etl_preview", None)
                    st.success("✓ Transformation applied to dataset")
                    st.rerun()

    # ── ETL Tab 2: Column operations ──────────────────────────────────────────
    with etl_tabs[1]:
        st.markdown('<div class="section-title">Column-level transformations</div>', unsafe_allow_html=True)
        cols_etl = list(df_etl.columns)

        with st.form("col_ops_form"):
            r1, r2 = st.columns(2)
            with r1:
                op_type = st.selectbox("Operation", [
                    "Add calculated column",
                    "Split column by delimiter",
                    "Merge two columns",
                    "Extract date part",
                    "Normalize (0–1 scale)",
                    "Standardize (z-score)",
                    "Bin numeric into categories",
                    "Map values (dict replace)",
                ])
            with r2:
                target_col = st.selectbox("Column", cols_etl, key="col_op_target")

            c1, c2, c3 = st.columns(3)
            with c1:
                new_col_name = st.text_input("New column name", placeholder="e.g. profit_margin")
            with c2:
                formula_input = st.text_input("Formula / delimiter / second column",
                                              placeholder="e.g. revenue - cost  or  ,  or  last_name")
            with c3:
                bins_input = st.text_input("Bins (for binning, e.g. 0,100,500,1000)",
                                           placeholder="0,100,500,1000")

            labels_input = st.text_input("Bin labels (e.g. Low,Medium,High)", placeholder="Low,Medium,High")
            map_input = st.text_area("Value mapping JSON (e.g. {\"Y\": 1, \"N\": 0})",
                                     placeholder='{"Y": 1, "N": 0}', height=60)

            date_part = st.selectbox("Date part (for extract)", ["year", "month", "day", "weekday", "quarter"])

            if st.form_submit_button("Apply column operation"):
                try:
                    new_df = df_etl.copy()
                    col_name = new_col_name or f"{target_col}_transformed"
                    log_entry = ""

                    if op_type == "Add calculated column":
                        new_df[col_name] = new_df.eval(formula_input)
                        log_entry = f"Added column '{col_name}' = {formula_input}"

                    elif op_type == "Split column by delimiter":
                        delim = formula_input or ","
                        split_df = new_df[target_col].astype(str).str.split(delim, expand=True)
                        for i, c in enumerate(split_df.columns):
                            new_df[f"{target_col}_part{i+1}"] = split_df[c]
                        log_entry = f"Split '{target_col}' by '{delim}'"

                    elif op_type == "Merge two columns":
                        sep = formula_input or " "
                        second = bins_input or ""
                        if second in new_df.columns:
                            new_df[col_name] = new_df[target_col].astype(str) + sep + new_df[second].astype(str)
                            log_entry = f"Merged '{target_col}' + '{second}' → '{col_name}'"

                    elif op_type == "Extract date part":
                        new_df[target_col] = pd.to_datetime(new_df[target_col], errors="coerce")
                        new_df[col_name] = getattr(new_df[target_col].dt, date_part)
                        log_entry = f"Extracted {date_part} from '{target_col}' → '{col_name}'"

                    elif op_type == "Normalize (0–1 scale)":
                        mn, mx = new_df[target_col].min(), new_df[target_col].max()
                        new_df[col_name] = (new_df[target_col] - mn) / (mx - mn)
                        log_entry = f"Normalized '{target_col}' to 0–1 → '{col_name}'"

                    elif op_type == "Standardize (z-score)":
                        mu, sigma = new_df[target_col].mean(), new_df[target_col].std()
                        new_df[col_name] = (new_df[target_col] - mu) / sigma
                        log_entry = f"Standardized '{target_col}' (z-score) → '{col_name}'"

                    elif op_type == "Bin numeric into categories":
                        bin_edges = [float(x.strip()) for x in bins_input.split(",")]
                        bin_labels = [x.strip() for x in labels_input.split(",")]
                        new_df[col_name] = pd.cut(new_df[target_col], bins=bin_edges,
                                                   labels=bin_labels if len(bin_labels) == len(bin_edges)-1 else None)
                        log_entry = f"Binned '{target_col}' into categories → '{col_name}'"

                    elif op_type == "Map values (dict replace)":
                        mapping = json.loads(map_input)
                        new_df[col_name] = new_df[target_col].map(mapping).fillna(new_df[target_col])
                        log_entry = f"Mapped values in '{target_col}' → '{col_name}'"

                    st.session_state["df"] = new_df
                    st.session_state["clean_log"].append(f"[ETL] {log_entry}")
                    st.success(f"✓ {log_entry}")
                    st.rerun()

                except Exception as e:
                    st.error(f"Operation failed: {e}")

    # ── ETL Tab 3: Merge / Join ───────────────────────────────────────────────
    with etl_tabs[2]:
        st.markdown('<div class="section-title">Join with another dataset</div>', unsafe_allow_html=True)
        st.caption("Upload a second file to join with your current dataset.")

        second_file = st.file_uploader("Upload second dataset", type=["csv", "xlsx"], key="merge_file")
        if second_file:
            df2 = load_csv_excel(second_file)
            st.caption(f"Second dataset: {df2.shape[0]:,} rows × {df2.shape[1]} cols")
            st.dataframe(df2.head(5), use_container_width=True)

            with st.form("merge_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    left_key = st.selectbox("Left key (your data)", df_etl.columns.tolist())
                with c2:
                    right_key = st.selectbox("Right key (second file)", df2.columns.tolist())
                with c3:
                    how = st.selectbox("Join type", ["inner", "left", "right", "outer"])

                if st.form_submit_button("Merge datasets"):
                    try:
                        merged = pd.merge(df_etl, df2, left_on=left_key, right_on=right_key, how=how)
                        st.session_state["df"] = merged
                        st.session_state["clean_log"].append(
                            f"[ETL] {how.upper()} JOIN on '{left_key}' = '{right_key}' → {merged.shape[0]:,} rows"
                        )
                        st.success(f"✓ Merged: {merged.shape[0]:,} rows × {merged.shape[1]} cols")
                        st.dataframe(merged.head(10), use_container_width=True)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Merge failed: {e}")

    # ── ETL Tab 4: Pivot & Reshape ────────────────────────────────────────────
    with etl_tabs[3]:
        st.markdown('<div class="section-title">Pivot table / Reshape</div>', unsafe_allow_html=True)
        cols_etl2 = list(df_etl.columns)
        num_cols = df_etl.select_dtypes(include="number").columns.tolist()

        reshape_type = st.radio("Operation", ["Pivot table", "Melt (wide → long)", "Transpose"], horizontal=True)

        if reshape_type == "Pivot table":
            with st.form("pivot_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    pivot_index = st.selectbox("Row (index)", cols_etl2)
                with c2:
                    pivot_cols = st.selectbox("Columns", cols_etl2)
                with c3:
                    pivot_vals = st.selectbox("Values", num_cols if num_cols else cols_etl2)
                pivot_agg = st.selectbox("Aggregation", ["sum", "mean", "count", "min", "max", "median"])

                if st.form_submit_button("Create pivot table"):
                    try:
                        pivoted = df_etl.pivot_table(
                            index=pivot_index, columns=pivot_cols,
                            values=pivot_vals, aggfunc=pivot_agg
                        ).reset_index()
                        pivoted.columns = [str(c) for c in pivoted.columns]
                        st.session_state["df"] = pivoted
                        st.session_state["clean_log"].append(
                            f"[ETL] Pivot: index={pivot_index}, cols={pivot_cols}, values={pivot_vals} ({pivot_agg})"
                        )
                        st.success(f"✓ Pivot created: {pivoted.shape[0]} rows × {pivoted.shape[1]} cols")
                        st.dataframe(pivoted.head(15), use_container_width=True)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Pivot failed: {e}")

        elif reshape_type == "Melt (wide → long)":
            with st.form("melt_form"):
                id_vars = st.multiselect("ID columns (keep as-is)", cols_etl2)
                value_name = st.text_input("Value column name", "value")
                var_name = st.text_input("Variable column name", "variable")

                if st.form_submit_button("Melt dataset"):
                    try:
                        melted = df_etl.melt(id_vars=id_vars, var_name=var_name, value_name=value_name)
                        st.session_state["df"] = melted
                        st.session_state["clean_log"].append(
                            f"[ETL] Melt: id_vars={id_vars} → {melted.shape[0]:,} rows"
                        )
                        st.success(f"✓ Melted: {melted.shape[0]:,} rows × {melted.shape[1]} cols")
                        st.dataframe(melted.head(15), use_container_width=True)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Melt failed: {e}")

        elif reshape_type == "Transpose":
            if st.button("Transpose dataset"):
                transposed = df_etl.T.reset_index()
                transposed.columns = [f"col_{i}" for i in range(len(transposed.columns))]
                st.session_state["df"] = transposed
                st.session_state["clean_log"].append("[ETL] Transposed dataset (rows ↔ columns)")
                st.success(f"✓ Transposed: {transposed.shape[0]} rows × {transposed.shape[1]} cols")
                st.rerun()

    # ── ETL Tab 5: AI Transformation Plan ────────────────────────────────────
    with etl_tabs[4]:
        st.markdown('<div class="section-title">AI full transformation plan</div>', unsafe_allow_html=True)
        st.caption("Describe your end goal. Claude writes a complete step-by-step transformation plan.")

        goal = st.text_area(
            "What is the final dataset you need?",
            placeholder="e.g. I need a monthly summary table showing total revenue, average order value, "
                        "and number of transactions per region, ready for a Power BI dashboard.",
            height=120,
            label_visibility="collapsed"
        )

        if st.button("Generate transformation plan", key="gen_plan"):
            if goal.strip():
                with st.spinner("Claude is building your transformation plan..."):
                    system = (
                        "You are a senior BI consultant. "
                        "Given a dataset schema and a user's goal, produce a clear step-by-step "
                        "data transformation plan. Each step should be specific and actionable. "
                        "Format your response as a numbered list. "
                        "End with a note on what the final dataset will look like and how to use it in Power BI."
                    )
                    schema_info = "\n".join([f"- {c}: {df_etl[c].dtype} ({df_etl[c].nunique()} unique values)" for c in df_etl.columns])
                    user_msg = (
                        f"Dataset: {df_etl.shape[0]:,} rows × {df_etl.shape[1]} columns\n"
                        f"Columns:\n{schema_info}\n\n"
                        f"Sample:\n{df_etl.head(3).to_string()}\n\n"
                        f"Goal: {goal}"
                    )
                    plan = call_claude(user_msg, system, max_tokens=1500)
                    st.session_state["etl_plan"] = plan

        if st.session_state.get("etl_plan"):
            st.markdown(f'<div class="ai-narrative">{st.session_state["etl_plan"]}</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:1rem;border-top:1px solid #e2e8f0;
font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#cbd5e1;text-align:center">
DataOps AI · Module 01 + 02 — Data Quality, Cleaning &amp; ETL · Built by Mohammed Amine Goumri
</div>
""", unsafe_allow_html=True)
