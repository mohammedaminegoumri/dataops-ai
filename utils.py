"""
Utility functions for data processing, AI integration, and charting.
Modular, reusable, and production-ready.
"""

import json
import re
import pandas as pd
import numpy as np
import anthropic
import plotly.express as px
import plotly.graph_objects as go
import sqlalchemy
import streamlit as st
from typing import Optional, Dict, List, Any, Tuple


# ── AI INTEGRATION ──────────────────────────────────────────────────────────

def get_anthropic_client() -> anthropic.Anthropic:
    """Initialize and return Anthropic client."""
    return anthropic.Anthropic(api_key=st.secrets.get("ANTHROPIC_API_KEY"))


def call_claude(
    prompt: str,
    system: str,
    max_tokens: int = 1500,
    model: str = "claude-haiku-4-5-20251001"
) -> str:
    """
    Call Claude API with given prompt and system message.
    
    Args:
        prompt: User message
        system: System prompt for context
        max_tokens: Maximum tokens in response
        model: Model identifier
        
    Returns:
        Response text from Claude
    """
    try:
        client = get_anthropic_client()
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        st.error(f"AI Error: {str(e)}")
        return ""


def extract_json(text: str) -> Optional[Dict | List]:
    """
    Extract JSON from markdown code blocks or raw JSON text.
    
    Args:
        text: Text containing JSON
        
    Returns:
        Parsed JSON object or None
    """
    # Try markdown code block first
    match = re.search(r"```json\s*([\s\S]+?)\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try raw JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ── DATA LOADING ────────────────────────────────────────────────────────────

def load_file(file_obj) -> Optional[pd.DataFrame]:
    """
    Load CSV or Excel file into DataFrame.
    
    Args:
        file_obj: Uploaded file object
        
    Returns:
        DataFrame or None if error
    """
    try:
        if file_obj.name.lower().endswith(".csv"):
            return pd.read_csv(file_obj)
        elif file_obj.name.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(file_obj)
        else:
            st.error("Unsupported file format. Use CSV or Excel.")
            return None
    except Exception as e:
        st.error(f"File loading error: {str(e)}")
        return None


def load_sql(connection_string: str, query: str) -> Optional[pd.DataFrame]:
    """
    Load data from SQL database.
    
    Args:
        connection_string: SQLAlchemy connection string
        query: SQL query to execute
        
    Returns:
        DataFrame or None if error
    """
    try:
        engine = sqlalchemy.create_engine(connection_string)
        with engine.connect() as connection:
            return pd.read_sql(query, connection)
    except Exception as e:
        st.error(f"SQL Error: {str(e)}")
        return None


# ── DATA QUALITY ANALYSIS ───────────────────────────────────────────────────

def analyze_data_quality(df: pd.DataFrame) -> Tuple[float, List[Dict]]:
    """
    Comprehensive data quality analysis.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Tuple of (quality_score, issues_list)
    """
    issues = []
    score = 100.0
    
    # Missing values
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            pct = round(count / len(df) * 100, 1)
            severity = "error" if pct > 20 else "warning"
            issues.append({
                "column": col,
                "type": "Missing values",
                "detail": f"{count} nulls ({pct}%)",
                "severity": severity,
                "count": int(count)
            })
            score -= min(15, pct * 0.4)
    
    # Duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        pct = round(duplicates / len(df) * 100, 1)
        issues.append({
            "column": "ALL ROWS",
            "type": "Duplicate rows",
            "detail": f"{duplicates} duplicates ({pct}%)",
            "severity": "error" if pct > 5 else "warning",
            "count": int(duplicates)
        })
        score -= min(20, pct * 2)
    
    # Type mismatches
    for col in df.select_dtypes(include="object").columns:
        numeric_pct = pd.to_numeric(df[col].dropna(), errors="coerce").notna().mean()
        if numeric_pct > 0.7:
            issues.append({
                "column": col,
                "type": "Type mismatch",
                "detail": "Numeric values stored as text",
                "severity": "warning",
                "count": 0
            })
            score -= 5
    
    # Outliers
    for col in df.select_dtypes(include="number").columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            outlier_count = int(((df[col] < q1 - 3*iqr) | (df[col] > q3 + 3*iqr)).sum())
            if outlier_count > 0:
                issues.append({
                    "column": col,
                    "type": "Outliers",
                    "detail": f"{outlier_count} extreme values detected",
                    "severity": "info",
                    "count": outlier_count
                })
    
    # Whitespace issues
    for col in df.select_dtypes(include="object").columns:
        ws_count = df[col].dropna().apply(lambda x: str(x) != str(x).strip()).sum()
        if ws_count > 0:
            issues.append({
                "column": col,
                "type": "Whitespace",
                "detail": f"{ws_count} values with leading/trailing spaces",
                "severity": "info",
                "count": int(ws_count)
            })
    
    return max(0, score), issues


# ── CHARTING ────────────────────────────────────────────────────────────────

CHART_COLORS = [
    "#1d4ed8", "#7c3aed", "#0ea5e9", "#16a34a",
    "#d97706", "#dc2626", "#ec4899", "#14b8a6"
]


def create_chart(config: Dict, df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Create Plotly chart based on configuration.
    
    Args:
        config: Chart configuration dict with keys: x, y, color, type, agg, title
        df: DataFrame to plot
        
    Returns:
        Plotly Figure or None if error
    """
    try:
        x = config.get("x") if config.get("x") in df.columns else None
        y = config.get("y") if config.get("y") in df.columns else None
        color = config.get("color") if config.get("color") in df.columns else None
        agg = config.get("agg", "none")
        chart_type = config.get("type", "bar")
        title = config.get("title", "Chart")
        
        plot_df = df.copy()
        
        # Apply aggregation if needed
        if agg != "none" and x and y:
            plot_df = getattr(plot_df.groupby(x)[y], agg)().reset_index()
            color = None
        
        fig = None
        kw = dict(color_discrete_sequence=CHART_COLORS, title=title, height=300)
        
        if chart_type == "bar":
            fig = px.bar(plot_df, x=x, y=y, color=color, **kw)
        elif chart_type == "line":
            fig = px.line(plot_df, x=x, y=y, color=color, **kw)
        elif chart_type == "area":
            fig = px.area(plot_df, x=x, y=y, color=color, **kw)
        elif chart_type == "pie":
            fig = px.pie(plot_df, names=x, values=y, color_discrete_sequence=CHART_COLORS, title=title, height=300)
        elif chart_type == "scatter":
            fig = px.scatter(plot_df, x=x, y=y, color=color, **kw)
        elif chart_type == "histogram":
            fig = px.histogram(plot_df, x=x or y, color=color, **kw)
        elif chart_type == "box":
            fig = px.box(plot_df, x=x, y=y, color=color, **kw)
        
        if fig:
            fig.update_layout(
                font_family="Inter",
                title_font_family="JetBrains Mono",
                title_font_size=12,
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=8, r=8, t=40, b=8),
                hovermode="x unified"
            )
            fig.update_xaxes(showgrid=False, linecolor="#e2e8f0")
            fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0")
        
        return fig
    except Exception as e:
        st.error(f"Chart error: {str(e)}")
        return None


# ── DATA TRANSFORMATION ─────────────────────────────────────────────────────

def apply_type_conversion(df: pd.DataFrame, column: str, target_type: str) -> Optional[pd.DataFrame]:
    """
    Convert column to target type.
    
    Args:
        df: DataFrame
        column: Column name
        target_type: Target data type (int, float, str, datetime)
        
    Returns:
        Modified DataFrame or None if error
    """
    try:
        df_copy = df.copy()
        if target_type == "int":
            df_copy[column] = pd.to_numeric(df_copy[column], errors="coerce").astype("Int64")
        elif target_type == "float":
            df_copy[column] = pd.to_numeric(df_copy[column], errors="coerce")
        elif target_type == "str":
            df_copy[column] = df_copy[column].astype(str)
        elif target_type == "datetime":
            df_copy[column] = pd.to_datetime(df_copy[column], errors="coerce")
        return df_copy
    except Exception as e:
        st.error(f"Type conversion error: {str(e)}")
        return None


def apply_text_operation(df: pd.DataFrame, column: str, operation: str, param: str = "") -> Optional[pd.DataFrame]:
    """
    Apply text operation to column.
    
    Args:
        df: DataFrame
        column: Column name
        operation: Operation type (upper, lower, trim, replace)
        param: Parameter for operation (e.g., replacement text)
        
    Returns:
        Modified DataFrame or None if error
    """
    try:
        df_copy = df.copy()
        if operation == "upper":
            df_copy[column] = df_copy[column].str.upper()
        elif operation == "lower":
            df_copy[column] = df_copy[column].str.lower()
        elif operation == "trim":
            df_copy[column] = df_copy[column].str.strip()
        elif operation == "replace" and param:
            old, new = param.split("|")
            df_copy[column] = df_copy[column].str.replace(old, new)
        return df_copy
    except Exception as e:
        st.error(f"Text operation error: {str(e)}")
        return None


# ── SESSION STATE HELPERS ───────────────────────────────────────────────────

def initialize_session_state():
    """Initialize session state variables if not present."""
    defaults = {
        "df": None,
        "quality_score": 0,
        "quality_issues": [],
        "transformation_log": [],
        "kpis": [],
        "charts": [],
        "etl_plan": None,
        "executive_report": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_log_entry(log_type: str, message: str):
    """Add entry to transformation log."""
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [{log_type}] {message}"
    if "transformation_log" not in st.session_state:
        st.session_state["transformation_log"] = []
    st.session_state["transformation_log"].append(entry)


# ── FORMATTING HELPERS ──────────────────────────────────────────────────────

def format_number(value: float, decimals: int = 2) -> str:
    """Format number with thousands separator."""
    return f"{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format as percentage."""
    return f"{value:.{decimals}f}%"


def get_severity_color(severity: str) -> str:
    """Get color class for severity level."""
    severity_map = {
        "error": "red",
        "warning": "amber",
        "info": "blue"
    }
    return severity_map.get(severity, "blue")
