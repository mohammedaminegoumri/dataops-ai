"""
DataOps AI - Modern SaaS Dashboard
A production-ready data application with tab-based navigation and premium UX.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from utils import (
    get_anthropic_client,
    call_claude,
    extract_json,
    load_file,
    load_sql,
    analyze_data_quality,
    create_chart,
    apply_type_conversion,
    apply_text_operation,
    initialize_session_state,
    add_log_entry,
    format_number,
    format_percentage,
    get_severity_color,
    CHART_COLORS
)


# ── PAGE CONFIGURATION ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="DataOps AI",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CUSTOM CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* { margin: 0; padding: 0; box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none !important; }

.main-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 3rem 2.5rem;
}

.page-header {
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #e2e8f0;
}

.page-eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #1d4ed8;
    margin-bottom: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
}

.page-title {
    font-size: 2rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin: 0;
}

.page-description {
    font-size: 1rem;
    color: #475569;
    margin-top: 0.5rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem !important;
    background-color: #f8fafc !important;
    border-radius: 12px !important;
    padding: 0.3rem !important;
    border: 1px solid #e2e8f0 !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: #475569 !important;
    background-color: transparent !important;
    transition: all 150ms ease-in-out !important;
}

.stTabs [aria-selected="true"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}

.card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    transition: all 250ms ease-in-out;
}

.card:hover {
    border-color: #cbd5e1;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.section-header {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
    margin: 1.6rem 0 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e2e8f0;
}

.kpi-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
    position: relative;
    overflow: hidden;
    height: 100%;
    transition: all 250ms ease-in-out;
}

.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 3px;
}

.kpi-card.blue::after { background: #1d4ed8; }
.kpi-card.green::after { background: #16a34a; }
.kpi-card.red::after { background: #dc2626; }
.kpi-card.amber::after { background: #d97706; }
.kpi-card.purple::after { background: #7c3aed; }

.kpi-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
    margin-bottom: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
}

.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9rem;
    font-weight: 500;
    color: #0f172a;
    line-height: 1;
    margin-bottom: 0.4rem;
}

.kpi-delta {
    font-size: 0.78rem;
    font-weight: 500;
}

.kpi-delta.up { color: #16a34a; }
.kpi-delta.down { color: #dc2626; }
.kpi-delta.flat { color: #94a3b8; }

.kpi-insight {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 0.4rem;
    line-height: 1.5;
}

.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    border-radius: 9px !important;
    padding: 0.5rem 1.3rem !important;
    border: 1.5px solid #1d4ed8 !important;
    background-color: #1d4ed8 !important;
    color: #ffffff !important;
    transition: all 150ms ease-in-out !important;
}

.stButton > button:hover {
    background-color: #1e40af !important;
    border-color: #1e40af !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
}

.issue-row {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    padding: 0.9rem 1rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    border: 1px solid transparent;
}

.issue-row.error { background: #fef2f2; border-color: #fecaca; }
.issue-row.warning { background: #fffbeb; border-color: #fde68a; }
.issue-row.info { background: #eff6ff; border-color: #bfdbfe; }

.issue-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-top: 4px;
    flex-shrink: 0;
}

.issue-dot.error { background: #dc2626; }
.issue-dot.warning { background: #d97706; }
.issue-dot.info { background: #1d4ed8; }

.issue-column {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 500;
    color: #0f172a;
}

.issue-type {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.1rem;
}

.ai-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
    font-size: 0.88rem;
    line-height: 1.8;
    color: #334155;
    white-space: pre-wrap;
    word-break: break-word;
}

.ai-box-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.8rem;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #1d4ed8;
    font-family: 'JetBrains Mono', monospace;
}

.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    background: #f8fafc;
    border: 2px dashed #e2e8f0;
    border-radius: 16px;
    color: #94a3b8;
}

.empty-icon { font-size: 2.5rem; margin-bottom: 0.8rem; }
.empty-title { font-size: 1rem; font-weight: 600; color: #64748b; margin-bottom: 0.3rem; }
.empty-desc { font-size: 0.82rem; }

.tag {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 5px;
    margin-right: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}

.tag.blue { background: #dbeafe; color: #1e40af; }
.tag.green { background: #dcfce7; color: #166534; }
.tag.purple { background: #ede9fe; color: #5b21b6; }
.tag.amber { background: #fef3c7; color: #92400e; }
.tag.red { background: #fee2e2; color: #991b1b; }

.log-entry {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 0.35rem 0.7rem;
    border-radius: 6px;
    margin-bottom: 0.3rem;
}

.log-entry.ok { background: #f0fdf4; color: #166534; }
.log-entry.err { background: #fef2f2; color: #991b1b; }
.log-entry.etl { background: #eff6ff; color: #1e40af; }
.log-entry.doc { background: #faf5ff; color: #5b21b6; }

[data-testid="stFileUploader"] {
    border: 2px dashed #cbd5e1 !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
    background: #f8fafc !important;
}

.stMetric {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
}

.stDataFrame {
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
}

</style>
""", unsafe_allow_html=True)


# ── INITIALIZATION ──────────────────────────────────────────────────────────

initialize_session_state()


# ── MAIN APPLICATION ───────────────────────────────────────────────────────

def render_page_header(title: str, description: str, eyebrow: str = ""):
    """Render page header with title and description."""
    html = f"""
    <div class="page-header">
        {f'<div class="page-eyebrow">{eyebrow}</div>' if eyebrow else ''}
        <div class="page-title">{title}</div>
        <div class="page-description">{description}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, delta: str = "", direction: str = "neutral", insight: str = "", color: str = "blue"):
    """Render a KPI card."""
    arrow = "↑" if direction == "up" else ("↓" if direction == "down" else "→")
    delta_class = "up" if direction == "up" else ("down" if direction == "down" else "flat")
    
    html = f"""
    <div class="kpi-card {color}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-delta {delta_class}">{arrow} {delta}</div>
        <div class="kpi-insight">{insight}</div>
    </div>
    """
    return html


def render_issue_row(column: str, issue_type: str, detail: str, severity: str):
    """Render an issue row."""
    html = f"""
    <div class="issue-row {severity}">
        <div class="issue-dot {severity}"></div>
        <div>
            <div class="issue-column">{column}</div>
            <div class="issue-type">{issue_type} — {detail}</div>
        </div>
    </div>
    """
    return html


def main():
    """Main application logic."""
    
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        # ── MAIN TABS ───────────────────────────────────────────────────────
        
        tab_overview, tab_data_source, tab_quality, tab_transform, tab_analytics, tab_docs = st.tabs([
            "Overview",
            "Data Source",
            "Data Quality",
            "Transformation",
            "Analytics",
            "Documentation"
        ])
        
        # ── TAB: OVERVIEW ───────────────────────────────────────────────────
        
        with tab_overview:
            render_page_header(
                "Dashboard Overview",
                "High-level metrics and dataset status",
                "Module 01"
            )
            
            if st.session_state["df"] is None:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-icon">📊</div>
                    <div class="empty-title">No data loaded</div>
                    <div class="empty-desc">Navigate to the Data Source tab to load your dataset</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                df = st.session_state["df"]
                
                # KPI Cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(render_kpi_card(
                        "Total Rows",
                        format_number(len(df), 0),
                        "",
                        "neutral",
                        "Dataset size",
                        "blue"
                    ), unsafe_allow_html=True)
                
                with col2:
                    st.markdown(render_kpi_card(
                        "Columns",
                        str(len(df.columns)),
                        "",
                        "neutral",
                        "Number of features",
                        "purple"
                    ), unsafe_allow_html=True)
                
                with col3:
                    quality_score = st.session_state.get("quality_score", 0)
                    st.markdown(render_kpi_card(
                        "Quality Score",
                        f"{quality_score:.0f}%",
                        "",
                        "neutral",
                        "Data completeness",
                        "green" if quality_score > 80 else "amber"
                    ), unsafe_allow_html=True)
                
                with col4:
                    num_transforms = len(st.session_state.get("transformation_log", []))
                    st.markdown(render_kpi_card(
                        "Transformations",
                        str(num_transforms),
                        "",
                        "neutral",
                        "Operations applied",
                        "blue"
                    ), unsafe_allow_html=True)
                
                # Dataset Info
                st.markdown('<div class="section-header">Dataset Information</div>', unsafe_allow_html=True)
                
                info_col1, info_col2, info_col3 = st.columns(3)
                
                with info_col1:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                with info_col2:
                    numeric_cols = len(df.select_dtypes(include="number").columns)
                    st.metric("Numeric Columns", numeric_cols)
                
                with info_col3:
                    categorical_cols = len(df.select_dtypes(include="object").columns)
                    st.metric("Categorical Columns", categorical_cols)
                
                # Recent Transformations
                st.markdown('<div class="section-header">Recent Transformations</div>', unsafe_allow_html=True)
                
                logs = st.session_state.get("transformation_log", [])
                if logs:
                    for log in logs[-5:]:
                        log_type = "etl" if "[ETL]" in log else "ok"
                        st.markdown(f'<div class="log-entry {log_type}">{log}</div>', unsafe_allow_html=True)
                else:
                    st.info("No transformations applied yet")
        
        # ── TAB: DATA SOURCE ────────────────────────────────────────────────
        
        with tab_data_source:
            render_page_header(
                "Data Source",
                "Load data from files or SQL database",
                "Module 02"
            )
            
            source_type = st.radio("Select source type", ["File Upload", "SQL Connection"], horizontal=True, label_visibility="collapsed")
            
            if source_type == "File Upload":
                st.markdown('<div class="section-header">Upload File</div>', unsafe_allow_html=True)
                
                uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
                
                if uploaded_file:
                    df = load_file(uploaded_file)
                    if df is not None:
                        st.session_state["df"] = df
                        st.success(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
                        
                        st.markdown('<div class="section-header">Data Preview</div>', unsafe_allow_html=True)
                        st.dataframe(df.head(10), use_container_width=True)
            
            else:  # SQL Connection
                st.markdown('<div class="section-header">SQL Connection</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    conn_string = st.text_input("Connection string", placeholder="postgresql://user:pass@host/db")
                
                with col2:
                    query = st.text_area("SQL Query", placeholder="SELECT * FROM table LIMIT 1000", height=100)
                
                if st.button("Execute Query"):
                    if conn_string and query:
                        df = load_sql(conn_string, query)
                        if df is not None:
                            st.session_state["df"] = df
                            st.success(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
                            
                            st.markdown('<div class="section-header">Data Preview</div>', unsafe_allow_html=True)
                            st.dataframe(df.head(10), use_container_width=True)
        
        # ── TAB: DATA QUALITY ───────────────────────────────────────────────
        
        with tab_quality:
            render_page_header(
                "Data Quality",
                "Analyze data completeness, duplicates, and anomalies",
                "Module 03"
            )
            
            if st.session_state["df"] is None:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-icon">🔍</div>
                    <div class="empty-title">No data loaded</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                df = st.session_state["df"]
                
                if st.button("Run Quality Analysis"):
                    with st.spinner("Analyzing data quality..."):
                        score, issues = analyze_data_quality(df)
                        st.session_state["quality_score"] = score
                        st.session_state["quality_issues"] = issues
                
                # Quality Score
                score = st.session_state.get("quality_score", 0)
                issues = st.session_state.get("quality_issues", [])
                
                if score > 0 or issues:
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 2rem; background: #0f172a; border-radius: 12px;">
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 3.8rem; font-weight: 500; line-height: 1; color: #ffffff;">
                                {score:.0f}%
                            </div>
                            <div style="font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; color: #475569; margin-top: 0.4rem; font-family: 'JetBrains Mono', monospace;">
                                Quality Score
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if issues:
                            st.markdown('<div class="section-header">Issues Found</div>', unsafe_allow_html=True)
                            for issue in issues[:10]:
                                severity = issue.get("severity", "info")
                                st.markdown(render_issue_row(
                                    issue.get("column", ""),
                                    issue.get("type", ""),
                                    issue.get("detail", ""),
                                    severity
                                ), unsafe_allow_html=True)
                        else:
                            st.success("✓ No issues detected")
        
        # ── TAB: TRANSFORMATION ────────────────────────────────────────────
        
        with tab_transform:
            render_page_header(
                "Data Transformation",
                "Apply manual transformations and generate ETL plans",
                "Module 04"
            )
            
            if st.session_state["df"] is None:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-icon">⚙</div>
                    <div class="empty-title">No data loaded</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                df = st.session_state["df"]
                
                transform_type = st.radio(
                    "Select transformation",
                    ["Type Conversion", "Text Operations", "Remove Duplicates", "AI-Powered Plan"],
                    horizontal=True,
                    label_visibility="collapsed"
                )
                
                if transform_type == "Type Conversion":
                    st.markdown('<div class="section-header">Convert Column Type</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        column = st.selectbox("Select column", df.columns)
                    
                    with col2:
                        target_type = st.selectbox("Target type", ["int", "float", "str", "datetime"])
                    
                    if st.button("Convert"):
                        result_df = apply_type_conversion(df, column, target_type)
                        if result_df is not None:
                            st.session_state["df"] = result_df
                            add_log_entry("ETL", f"Type conversion: {column} → {target_type}")
                            st.success(f"✓ Converted {column} to {target_type}")
                            st.rerun()
                
                elif transform_type == "Text Operations":
                    st.markdown('<div class="section-header">Apply Text Operation</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        column = st.selectbox("Select column", df.select_dtypes(include="object").columns)
                    
                    with col2:
                        operation = st.selectbox("Operation", ["upper", "lower", "trim", "replace"])
                    
                    param = ""
                    if operation == "replace":
                        param = st.text_input("Enter replacement (old|new)")
                    
                    if st.button("Apply"):
                        result_df = apply_text_operation(df, column, operation, param)
                        if result_df is not None:
                            st.session_state["df"] = result_df
                            add_log_entry("ETL", f"Text operation: {column} → {operation}")
                            st.success(f"✓ Applied {operation} to {column}")
                            st.rerun()
                
                elif transform_type == "Remove Duplicates":
                    st.markdown('<div class="section-header">Remove Duplicate Rows</div>', unsafe_allow_html=True)
                    
                    before_count = len(df)
                    df_clean = df.drop_duplicates()
                    after_count = len(df_clean)
                    removed = before_count - after_count
                    
                    st.info(f"Found {removed} duplicate rows")
                    
                    if removed > 0 and st.button("Remove Duplicates"):
                        st.session_state["df"] = df_clean
                        add_log_entry("ETL", f"Removed {removed} duplicate rows")
                        st.success(f"✓ Removed {removed} duplicates")
                        st.rerun()
                
                else:  # AI-Powered Plan
                    st.markdown('<div class="section-header">Generate ETL Plan</div>', unsafe_allow_html=True)
                    
                    goal = st.text_area(
                        "Describe your transformation goal",
                        placeholder="e.g., Monthly revenue per region, ready for Power BI",
                        height=100,
                        label_visibility="collapsed"
                    )
                    
                    if st.button("Generate Plan"):
                        if goal.strip():
                            with st.spinner("Building transformation plan..."):
                                schema = "\n".join([f"- {c}: {df[c].dtype} ({df[c].nunique()} unique)" for c in df.columns])
                                prompt = f"""Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns
Columns:
{schema}

Sample data:
{df.head(3).to_string()}

Goal: {goal}"""
                                
                                plan = call_claude(
                                    prompt,
                                    "Write a numbered step-by-step transformation plan. End with Power BI usage instructions.",
                                    max_tokens=1200
                                )
                                st.session_state["etl_plan"] = plan
                    
                    if st.session_state.get("etl_plan"):
                        st.markdown(f"""
                        <div class="ai-box">
                            <div class="ai-box-header">◆ Transformation Plan</div>
                            {st.session_state["etl_plan"]}
                        </div>
                        """, unsafe_allow_html=True)
        
        # ── TAB: ANALYTICS ──────────────────────────────────────────────────
        
        with tab_analytics:
            render_page_header(
                "Analytics & Visualization",
                "Interactive charts and statistical exploration",
                "Module 05"
            )
            
            if st.session_state["df"] is None:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-icon">📈</div>
                    <div class="empty-title">No data loaded</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                df = st.session_state["df"]
                
                analytics_type = st.radio(
                    "Select analysis type",
                    ["KPI Cards", "Smart Charts", "Correlation", "Distribution"],
                    horizontal=True,
                    label_visibility="collapsed"
                )
                
                if analytics_type == "KPI Cards":
                    st.markdown('<div class="section-header">Generate KPIs</div>', unsafe_allow_html=True)
                    
                    if st.button("Generate KPIs"):
                        with st.spinner("Extracting KPIs..."):
                            schema = "\n".join([f"{c}: {df[c].dtype}" for c in df.columns])
                            prompt = f"""Columns: {schema}
Stats:
{df.describe().to_string()}

Sample:
{df.head(5).to_string()}"""
                            
                            raw = call_claude(
                                prompt,
                                'Extract 4-6 KPIs. Return ONLY JSON:\n[{"label":str,"value":str,"delta":str,"direction":"up"|"down"|"neutral","insight":str,"color":"blue"|"green"|"red"|"amber"|"purple"}]'
                            )
                            kpis = extract_json(raw)
                            if isinstance(kpis, list):
                                st.session_state["kpis"] = kpis
                    
                    kpis = st.session_state.get("kpis", [])
                    if kpis:
                        cols = st.columns(min(3, len(kpis)))
                        for i, kpi in enumerate(kpis):
                            with cols[i % 3]:
                                st.markdown(render_kpi_card(
                                    kpi.get("label", ""),
                                    kpi.get("value", "—"),
                                    kpi.get("delta", ""),
                                    kpi.get("direction", "neutral"),
                                    kpi.get("insight", ""),
                                    kpi.get("color", "blue")
                                ), unsafe_allow_html=True)
                    else:
                        st.info("Click 'Generate KPIs' to create KPI cards")
                
                elif analytics_type == "Smart Charts":
                    st.markdown('<div class="section-header">Generate Smart Charts</div>', unsafe_allow_html=True)
                    
                    if st.button("Generate Charts"):
                        with st.spinner("Choosing best charts..."):
                            schema = ", ".join([f"{c}({df[c].dtype})" for c in df.columns])
                            prompt = f"""Columns: {schema}
Sample:
{df.head(4).to_string()}"""
                            
                            raw = call_claude(
                                prompt,
                                'Recommend 4 charts. Return ONLY JSON:\n[{"title":str,"type":"bar"|"line"|"area"|"pie"|"scatter"|"histogram"|"box","x":str|null,"y":str|null,"color":str|null,"agg":"sum"|"mean"|"count"|"none","rationale":str}]'
                            )
                            charts = extract_json(raw)
                            if isinstance(charts, list):
                                st.session_state["charts"] = charts
                    
                    charts = st.session_state.get("charts", [])
                    if charts:
                        for i in range(0, len(charts), 2):
                            pair = charts[i:i+2]
                            cols = st.columns(len(pair))
                            for ci, cfg in enumerate(pair):
                                with cols[ci]:
                                    fig = create_chart(cfg, df)
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                                        st.caption(f"💡 {cfg.get('rationale', '')}")
                    else:
                        st.info("Click 'Generate Charts' to create visualizations")
                
                elif analytics_type == "Correlation":
                    st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
                    
                    numeric_cols = df.select_dtypes(include="number").columns.tolist()
                    if len(numeric_cols) >= 2:
                        fig = px.imshow(
                            df[numeric_cols].corr(),
                            text_auto=".2f",
                            aspect="auto",
                            color_continuous_scale="RdBu_r",
                            title="Correlation Matrix",
                            zmin=-1,
                            zmax=1
                        )
                        fig.update_layout(font_family="Inter", paper_bgcolor="white", margin=dict(l=8, r=8, t=40, b=8))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Need at least 2 numeric columns")
                
                else:  # Distribution
                    st.markdown('<div class="section-header">Distribution Analysis</div>', unsafe_allow_html=True)
                    
                    numeric_cols = df.select_dtypes(include="number").columns.tolist()
                    if numeric_cols:
                        column = st.selectbox("Select column", numeric_cols)
                        
                        fig = px.histogram(
                            df,
                            x=column,
                            marginal="box",
                            color_discrete_sequence=[CHART_COLORS[0]],
                            title=f"Distribution of {column}"
                        )
                        fig.update_layout(font_family="Inter", paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=8, r=8, t=40, b=8))
                        fig.update_xaxes(showgrid=False)
                        fig.update_yaxes(gridcolor="#f1f5f9")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Mean", f"{df[column].mean():.2f}")
                        col2.metric("Median", f"{df[column].median():.2f}")
                        col3.metric("Std Dev", f"{df[column].std():.2f}")
                        col4.metric("Skewness", f"{df[column].skew():.2f}")
                    else:
                        st.info("No numeric columns available")
        
        # ── TAB: DOCUMENTATION ──────────────────────────────────────────────
        
        with tab_docs:
            render_page_header(
                "Documentation",
                "Data dictionary, transformation logs, and metadata",
                "Module 06"
            )
            
            if st.session_state["df"] is None:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-icon">📚</div>
                    <div class="empty-title">No data loaded</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                df = st.session_state["df"]
                
                doc_type = st.radio(
                    "Select documentation",
                    ["Data Dictionary", "Transformation Log"],
                    horizontal=True,
                    label_visibility="collapsed"
                )
                
                if doc_type == "Data Dictionary":
                    st.markdown('<div class="section-header">Column Information</div>', unsafe_allow_html=True)
                    
                    data_dict = []
                    for col in df.columns:
                        data_dict.append({
                            "Column": col,
                            "Type": str(df[col].dtype),
                            "Non-Null": f"{df[col].notna().sum():,}",
                            "Unique": df[col].nunique(),
                            "Missing": df[col].isna().sum()
                        })
                    
                    dict_df = pd.DataFrame(data_dict)
                    st.dataframe(dict_df, use_container_width=True)
                
                else:  # Transformation Log
                    st.markdown('<div class="section-header">Transformation History</div>', unsafe_allow_html=True)
                    
                    logs = st.session_state.get("transformation_log", [])
                    if logs:
                        for log in reversed(logs):
                            log_type = "etl" if "[ETL]" in log else "ok"
                            st.markdown(f'<div class="log-entry {log_type}">{log}</div>', unsafe_allow_html=True)
                    else:
                        st.info("No transformations applied yet")
        
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
