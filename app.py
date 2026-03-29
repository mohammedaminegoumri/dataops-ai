import streamlit as st
import pandas as pd
import numpy as np
import json, io, re
import anthropic
import plotly.express as px
import plotly.graph_objects as go
import sqlalchemy
from datetime import datetime

st.set_page_config(page_title="DataOps AI", page_icon="⚙", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Plus Jakarta Sans',sans-serif!important}
.block-container{padding:0!important;max-width:100%!important}
section[data-testid="stSidebar"]{background:#0f1117!important;border-right:1px solid #1e2330!important}
section[data-testid="stSidebar"]>div{padding:0!important}
.sb-logo{padding:1.5rem 1.4rem 1rem;border-bottom:1px solid #1e2330}
.sb-logo-text{font-family:'JetBrains Mono',monospace;font-size:1.1rem;font-weight:500;color:#f8fafc;letter-spacing:-.02em}
.sb-logo-sub{font-size:.7rem;color:#4b5563;margin-top:.2rem;font-family:'JetBrains Mono',monospace}
.sb-sec{padding:1rem 1rem .5rem}
.sb-sec-lbl{font-size:.62rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:#374151;margin-bottom:.4rem;padding:0 .4rem}
.sb-divider{border:none;border-top:1px solid #1e2330;margin:.8rem 1rem}
.sb-dataset{margin:.5rem 1rem 1rem;background:#111827;border:1px solid #1e2330;border-radius:10px;padding:.8rem 1rem}
.sdi-lbl{font-size:.65rem;color:#4b5563;text-transform:uppercase;letter-spacing:.08em;font-family:'JetBrains Mono',monospace}
.sdi-val{font-size:.82rem;color:#e5e7eb;font-family:'JetBrains Mono',monospace;margin-top:.15rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sdi-stat{display:inline-block;font-size:.7rem;color:#6b7280;font-family:'JetBrains Mono',monospace;margin-top:.3rem}
.main-wrap{padding:2rem 2.5rem;max-width:1300px}
.pg-eyebrow{font-size:.7rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:#1d4ed8;font-family:'JetBrains Mono',monospace;margin-bottom:.4rem}
.pg-title{font-size:1.6rem;font-weight:700;color:#0f172a;letter-spacing:-.03em;line-height:1.2;margin:0}
.pg-desc{font-size:.9rem;color:#64748b;margin-top:.4rem}
.page-header{margin-bottom:2rem}
.kpi-card{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:1.3rem 1.5rem;position:relative;overflow:hidden;height:100%}
.kpi-card::after{content:'';position:absolute;bottom:0;left:0;right:0;height:3px}
.kpi-card.blue::after{background:#1d4ed8}.kpi-card.green::after{background:#16a34a}
.kpi-card.red::after{background:#dc2626}.kpi-card.amber::after{background:#d97706}
.kpi-card.purple::after{background:#7c3aed}
.kpi-label{font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:#94a3b8;margin-bottom:.5rem;font-family:'JetBrains Mono',monospace}
.kpi-value{font-family:'JetBrains Mono',monospace;font-size:1.9rem;font-weight:500;color:#0f172a;line-height:1;margin-bottom:.4rem}
.kpi-delta{font-size:.78rem;font-weight:500}
.kpi-delta.up{color:#16a34a}.kpi-delta.down{color:#dc2626}.kpi-delta.flat{color:#94a3b8}
.kpi-insight{font-size:.75rem;color:#94a3b8;margin-top:.4rem;line-height:1.5}
.score-wrap{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:1.5rem;background:#0f172a;border-radius:14px}
.score-big{font-family:'JetBrains Mono',monospace;font-size:3.8rem;font-weight:500;line-height:1}
.score-lbl{font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:#475569;margin-top:.4rem;font-family:'JetBrains Mono',monospace}
.issue-row{display:flex;align-items:flex-start;gap:.8rem;padding:.9rem 1rem;border-radius:10px;margin-bottom:.5rem;border:1px solid transparent}
.issue-row.error{background:#fef2f2;border-color:#fecaca}
.issue-row.warning{background:#fffbeb;border-color:#fde68a}
.issue-row.info{background:#eff6ff;border-color:#bfdbfe}
.issue-dot{width:8px;height:8px;border-radius:50%;margin-top:4px;flex-shrink:0}
.issue-dot.error{background:#dc2626}.issue-dot.warning{background:#d97706}.issue-dot.info{background:#1d4ed8}
.issue-col{font-family:'JetBrains Mono',monospace;font-size:.82rem;font-weight:500;color:#0f172a}
.issue-type{font-size:.75rem;color:#64748b;margin-top:.1rem}
.sec-hd{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:#94a3b8;margin:1.6rem 0 .8rem;font-family:'JetBrains Mono',monospace}
.ai-box{background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:1.3rem 1.5rem;font-size:.88rem;line-height:1.8;color:#334155;white-space:pre-wrap}
.ai-box-hd{display:flex;align-items:center;gap:.5rem;margin-bottom:.8rem;font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:#1d4ed8;font-family:'JetBrains Mono',monospace}
.code-block{background:#1e1e2e;color:#cdd6f4;font-family:'JetBrains Mono',monospace;font-size:.8rem;padding:1.1rem 1.3rem;border-radius:10px;white-space:pre-wrap;line-height:1.6;overflow-x:auto}
.log-entry{font-family:'JetBrains Mono',monospace;font-size:.75rem;padding:.35rem .7rem;border-radius:6px;margin-bottom:.3rem}
.log-entry.ok{background:#f0fdf4;color:#166534}.log-entry.err{background:#fef2f2;color:#991b1b}
.log-entry.etl{background:#eff6ff;color:#1e40af}.log-entry.doc{background:#faf5ff;color:#5b21b6}
.doc-card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:1.2rem 1.4rem;margin-bottom:.8rem}
.doc-card-title{font-weight:600;font-size:.9rem;color:#0f172a;margin-bottom:.3rem}
.doc-card-meta{font-size:.75rem;color:#94a3b8;font-family:'JetBrains Mono',monospace}
.tag{display:inline-block;font-size:.68rem;font-weight:600;padding:.15rem .5rem;border-radius:5px;margin-right:.3rem;font-family:'JetBrains Mono',monospace}
.tag.blue{background:#dbeafe;color:#1e40af}.tag.green{background:#dcfce7;color:#166534}
.tag.purple{background:#ede9fe;color:#5b21b6}.tag.amber{background:#fef3c7;color:#92400e}
.tag.red{background:#fee2e2;color:#991b1b}
.empty-state{text-align:center;padding:3rem 2rem;background:#f8fafc;border:2px dashed #e2e8f0;border-radius:16px;color:#94a3b8}
.empty-icon{font-size:2.5rem;margin-bottom:.8rem}
.empty-title{font-size:1rem;font-weight:600;color:#64748b;margin-bottom:.3rem}
.empty-desc{font-size:.82rem}
.stButton>button{font-family:'Plus Jakarta Sans',sans-serif!important;font-weight:500!important;font-size:.875rem!important;border-radius:9px!important;padding:.5rem 1.3rem!important;border:1.5px solid #1d4ed8!important;background:#1d4ed8!important;color:#fff!important;transition:all .15s!important}
.stButton>button:hover{background:#1e40af!important;border-color:#1e40af!important}
.stTabs [data-baseweb="tab-list"]{gap:.2rem;background:#f8fafc;border-radius:10px;padding:.3rem;border:1px solid #e2e8f0}
.stTabs [data-baseweb="tab"]{border-radius:8px!important;padding:.4rem 1rem!important;font-size:.82rem!important;font-weight:500!important;color:#64748b!important;background:transparent!important}
.stTabs [aria-selected="true"]{background:#fff!important;color:#0f172a!important;box-shadow:0 1px 3px rgba(0,0,0,.08)!important}
[data-testid="stFileUploader"]{border:2px dashed #cbd5e1!important;border-radius:12px!important;padding:.5rem!important;background:#f8fafc!important}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_client(): return anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

def call_claude(prompt, system, max_tokens=1500):
    msg = get_client().messages.create(model="claude-haiku-4-5-20251001", max_tokens=max_tokens,
        system=system, messages=[{"role":"user","content":prompt}])
    return msg.content[0].text

def extract_json(text):
    m = re.search(r"```json\s*([\s\S]+?)\s*```", text)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    try: return json.loads(text)
    except: return None

def load_file(f):
    return pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)

def load_sql(conn_str, query):
    engine = sqlalchemy.create_engine(conn_str)
    with engine.connect() as c: return pd.read_sql(query, c)

COLORS = ["#1d4ed8","#7c3aed","#0ea5e9","#16a34a","#d97706","#dc2626","#ec4899","#14b8a6"]

def make_chart(cfg, df):
    x = cfg.get("x") if cfg.get("x") in df.columns else None
    y = cfg.get("y") if cfg.get("y") in df.columns else None
    color = cfg.get("color") if cfg.get("color") in df.columns else None
    agg, ctype, title = cfg.get("agg","none"), cfg.get("type","bar"), cfg.get("title","Chart")
    plot_df = df.copy()
    if agg != "none" and x and y:
        plot_df = getattr(plot_df.groupby(x)[y], agg)().reset_index(); color = None
    kw = dict(color_discrete_sequence=COLORS, title=title, height=300)
    fig = None
    try:
        if ctype=="bar": fig=px.bar(plot_df,x=x,y=y,color=color,**kw)
        elif ctype=="line": fig=px.line(plot_df,x=x,y=y,color=color,**kw)
        elif ctype=="area": fig=px.area(plot_df,x=x,y=y,color=color,**kw)
        elif ctype=="pie": fig=px.pie(plot_df,names=x,values=y,color_discrete_sequence=COLORS,title=title,height=300)
        elif ctype=="scatter": fig=px.scatter(plot_df,x=x,y=y,color=color,**kw)
        elif ctype=="histogram": fig=px.histogram(plot_df,x=x or y,color=color,**kw)
        elif ctype=="box": fig=px.box(plot_df,x=x,y=y,color=color,**kw)
        if fig:
            fig.update_layout(font_family="Plus Jakarta Sans",title_font_family="JetBrains Mono",
                title_font_size=12,plot_bgcolor="white",paper_bgcolor="white",margin=dict(l=8,r=8,t=40,b=8))
            fig.update_xaxes(showgrid=False,linecolor="#e2e8f0")
            fig.update_yaxes(showgrid=True,gridcolor="#f1f5f9",linecolor="#e2e8f0")
    except: pass
    return fig

def analyze_quality(df):
    issues=[]; score=100
    nc=df.isnull().sum()
    for col,n in nc.items():
        if n>0:
            pct=round(n/len(df)*100,1); sev="error" if pct>20 else "warning"
            issues.append({"column":col,"type":"Missing values","detail":f"{n} nulls ({pct}%)","severity":sev,"count":int(n)})
            score-=min(15,pct*0.4)
    dup=df.duplicated().sum()
    if dup>0:
        pct=round(dup/len(df)*100,1)
        issues.append({"column":"ALL ROWS","type":"Duplicate rows","detail":f"{dup} duplicates ({pct}%)","severity":"error" if pct>5 else "warning","count":int(dup)})
        score-=min(20,pct*2)
    for col in df.select_dtypes(include="object").columns:
        if pd.to_numeric(df[col].dropna(),errors="coerce").notna().mean()>0.7:
            issues.append({"column":col,"type":"Type mismatch","detail":"Numeric stored as text","severity":"warning","count":0})
            score-=5
    for col in df.select_dtypes(include="number").columns:
        q1,q3=df[col].quantile(0.25),df[col].quantile(0.75); iqr=q3-q1
        if iqr>0:
            n_out=int(((df[col]<q1-3*iqr)|(df[col]>q3+3*iqr)).sum())
            if n_out>0: issues.append({"column":col,"type":"Outliers","detail":f"{n_out} extreme values","severity":"info","count":n_out})
    for col in df.select_dtypes(include="object").columns:
        ws=df[col].dropna().apply(lambda x:str(x)!=str(x).strip()).sum()
        if ws>0: issues.append({"column":col,"type":"Whitespace","detail":f"{ws} values with extra spaces","severity":"info","count":int(ws)})
    return {"score":max(0,round(score)),"issues":issues,"null_total":int(nc.sum()),"dup_total":int(dup)}

def apply_cleaning(df, ops):
    df=df.copy(); log=[]
    for op in ops:
        try:
            a,col=op.get("action"),op.get("column")
            if a=="drop_nulls" and col:
                b=len(df);df=df.dropna(subset=[col]);log.append(f"Dropped {b-len(df)} null rows in '{col}'")
            elif a=="fill_nulls" and col:
                m,v=op.get("method","value"),op.get("value")
                if m=="mean" and pd.api.types.is_numeric_dtype(df[col]):
                    val=df[col].mean();df[col]=df[col].fillna(val);log.append(f"Filled '{col}' with mean ({round(val,2)})")
                elif m=="median" and pd.api.types.is_numeric_dtype(df[col]):
                    val=df[col].median();df[col]=df[col].fillna(val);log.append(f"Filled '{col}' with median ({round(val,2)})")
                elif m=="mode":
                    val=df[col].mode()[0];df[col]=df[col].fillna(val);log.append(f"Filled '{col}' with mode")
                else:
                    df[col]=df[col].fillna(v);log.append(f"Filled '{col}' with '{v}'")
            elif a=="drop_duplicates":
                b=len(df);df=df.drop_duplicates();log.append(f"Removed {b-len(df)} duplicates")
            elif a=="trim_whitespace" and col:
                df[col]=df[col].str.strip();log.append(f"Trimmed whitespace in '{col}'")
            elif a=="convert_type" and col:
                t=op.get("target_type","numeric")
                if t=="numeric": df[col]=pd.to_numeric(df[col],errors="coerce")
                elif t=="datetime": df[col]=pd.to_datetime(df[col],errors="coerce")
                else: df[col]=df[col].astype(str)
                log.append(f"Converted '{col}' to {t}")
            elif a=="rename_column" and col:
                nn=op.get("new_name");df=df.rename(columns={col:nn});log.append(f"Renamed '{col}' → '{nn}'")
            elif a=="drop_column" and col:
                df=df.drop(columns=[col]);log.append(f"Dropped column '{col}'")
            elif a=="cap_outliers" and col:
                if pd.api.types.is_numeric_dtype(df[col]):
                    q1,q3=df[col].quantile(0.25),df[col].quantile(0.75);iqr=q3-q1
                    b=((df[col]<q1-3*iqr)|(df[col]>q3+3*iqr)).sum()
                    df[col]=df[col].clip(lower=q1-3*iqr,upper=q3+3*iqr);log.append(f"Capped {b} outliers in '{col}'")
            elif a=="uppercase" and col:
                df[col]=df[col].str.upper();log.append(f"Uppercased '{col}'")
            elif a=="lowercase" and col:
                df[col]=df[col].str.lower();log.append(f"Lowercased '{col}'")
            elif a=="replace_value" and col:
                ov,nv=op.get("old_value"),op.get("new_value")
                df[col]=df[col].replace(ov,nv);log.append(f"Replaced '{ov}'→'{nv}' in '{col}'")
        except Exception as e: log.append(f"[ERROR] {op.get('action')} on '{op.get('column')}': {e}")
    return df, log

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {"page":"home","df":None,"df_original":None,"filename":None,"clean_log":[],
    "quality":None,"ai_diagnosis":None,"ai_suggestions":[],"kpis_rep":[],"charts_rep":[],
    "exec_report":None,"etl_sql":None,"etl_preview":None,"etl_plan":None,"custom_chart":None,
    "doc_dictionary":None,"doc_changelog":None,"doc_readme":None,"doc_lineage":None}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k]=v

def nav(p): st.session_state["page"]=p

df=st.session_state["df"]; has_data=df is not None; log=st.session_state["clean_log"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-logo"><div class="sb-logo-text">⚙ DataOps AI</div><div class="sb-logo-sub">// BI Automation Agent</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sec"><div class="sb-sec-lbl">Workspace</div></div>', unsafe_allow_html=True)

    nav_items = [("🏠  Home","home"),("📂  Load Data","load"),
        ("🔍  Quality & Cleaning","quality"),("⚡  ETL & Transform","etl"),
        ("📊  Reports & Charts","reports"),("📄  Documentation","docs")]
    for label,key in nav_items:
        active = st.session_state["page"]==key
        prefix = "▸ " if active else "   "
        if st.button(f"{prefix}{label}", key=f"nav_{key}", use_container_width=True):
            nav(key)

    if has_data:
        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sb-dataset">
            <div class="sdi-lbl">Active dataset</div>
            <div class="sdi-val">{st.session_state['filename'] or 'dataset'}</div>
            <div class="sdi-stat">{df.shape[0]:,} rows · {df.shape[1]} cols</div>
            {"<br><div class='sdi-stat' style='color:#16a34a'>✓ "+str(len(log))+" operations</div>" if log else ""}
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    q=st.session_state.get("quality")
    checks = [("Data loaded",has_data),("Quality analyzed",q is not None),
        ("Transformations",bool(log)),("Report ready",st.session_state.get("exec_report") is not None),
        ("Docs created",st.session_state.get("doc_dictionary") is not None)]
    status_md = "".join([f"{'🟢' if ok else '⚪'} {lbl}<br>" for lbl,ok in checks])
    st.markdown(f'<div style="padding:0 1rem 1rem;font-size:.72rem;color:#4b5563;font-family:JetBrains Mono,monospace">{status_md}</div>', unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="main-wrap">', unsafe_allow_html=True)
    page = st.session_state["page"]

    # ── HOME ──────────────────────────────────────────────────────────────────
    if page == "home":
        st.markdown('<div class="page-header"><div class="pg-eyebrow">BI Automation Agent</div><div class="pg-title">DataOps AI</div><div class="pg-desc">End-to-end data pipeline — quality, cleaning, ETL, reporting & documentation. Powered by Claude.</div></div>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        mods = [("01","🔍","Quality & Cleaning","quality","Null checks, outlier detection, AI diagnosis and interactive cleaning."),
                ("02","⚡","ETL & Transform","etl","AI SQL, column ops, joins, pivot and full transformation planning."),
                ("03","📊","Reports & Charts","reports","KPI cards, smart charts, heatmaps and a downloadable executive report."),
                ("04","📄","Documentation","docs","Data dictionary, changelog, README and lineage map — auto-generated.")]
        for i,(num,icon,name,key,desc) in enumerate(mods):
            col = c1 if i%2==0 else c2
            with col:
                if st.button(f"{icon}  Module {num} — {name}", key=f"hn_{key}", use_container_width=True): nav(key)
                st.markdown(f'<div style="font-size:.8rem;color:#64748b;margin:-.4rem 0 1rem;padding:0 .2rem">{desc}</div>', unsafe_allow_html=True)

        if not has_data:
            st.markdown('<div class="empty-state"><div class="empty-icon">📂</div><div class="empty-title">No data loaded yet</div><div class="empty-desc">Go to <strong>Load Data</strong> to get started.</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sec-hd">Pipeline status</div>', unsafe_allow_html=True)
            steps=[("Load",has_data),("Quality",q is not None),("Clean",bool(log)),
                   ("Transform",any("[ETL]" in l for l in log)),
                   ("Report",st.session_state.get("exec_report") is not None),
                   ("Docs",st.session_state.get("doc_dictionary") is not None)]
            sc=st.columns(len(steps))
            for i,(lbl,done) in enumerate(steps):
                with sc[i]:
                    bg="#16a34a" if done else "#e2e8f0"; tc="#fff" if done else "#94a3b8"
                    st.markdown(f'<div style="text-align:center"><div style="width:36px;height:36px;border-radius:50%;background:{bg};color:{tc};display:flex;align-items:center;justify-content:center;margin:0 auto;font-size:.7rem;font-family:JetBrains Mono,monospace">{"✓" if done else str(i+1)}</div><div style="font-size:.72rem;color:{"#16a34a" if done else "#94a3b8"};margin-top:.3rem;font-weight:{"600" if done else "400"}">{lbl}</div></div>', unsafe_allow_html=True)

    # ── LOAD ──────────────────────────────────────────────────────────────────
    elif page == "load":
        st.markdown('<div class="page-header"><div class="pg-eyebrow">Step 1</div><div class="pg-title">Load your data</div><div class="pg-desc">Upload a file or connect to a SQL database.</div></div>', unsafe_allow_html=True)
        t1,t2=st.tabs(["📁  File upload","🗄  SQL database"])
        with t1:
            uploaded=st.file_uploader("Drop your file here",type=["csv","xlsx","xls"],label_visibility="collapsed")
            if uploaded:
                try:
                    raw=load_file(uploaded)
                    st.session_state.update({"df":raw.copy(),"df_original":raw.copy(),"filename":uploaded.name,
                        "clean_log":[],"quality":None,"ai_diagnosis":None,"ai_suggestions":[],
                        "kpis_rep":[],"charts_rep":[],"exec_report":None})
                    st.success(f"✓ {uploaded.name} — {raw.shape[0]:,} rows × {raw.shape[1]} columns")
                    st.dataframe(raw.head(10),use_container_width=True)
                    if st.button("Continue to Quality Analysis →"): nav("quality")
                except Exception as e: st.error(f"Error: {e}")
        with t2:
            conn_str=st.text_input("Connection string",placeholder="postgresql://user:pass@host:5432/db")
            query=st.text_area("Query",placeholder="SELECT * FROM table LIMIT 10000",height=90)
            if st.button("Connect & Load"):
                try:
                    raw=load_sql(conn_str,query)
                    st.session_state.update({"df":raw.copy(),"df_original":raw.copy(),"filename":"sql_query.csv","clean_log":[],"quality":None})
                    st.success(f"✓ {raw.shape[0]:,} rows × {raw.shape[1]} cols")
                    st.dataframe(raw.head(10),use_container_width=True)
                except Exception as e: st.error(f"Connection error: {e}")

    # ── QUALITY ───────────────────────────────────────────────────────────────
    elif page == "quality":
        st.markdown('<div class="page-header"><div class="pg-eyebrow">Module 01</div><div class="pg-title">Quality & Cleaning</div><div class="pg-desc">Diagnose every issue and fix it with full control.</div></div>', unsafe_allow_html=True)
        if not has_data:
            st.markdown('<div class="empty-state"><div class="empty-icon">🔍</div><div class="empty-title">No data loaded</div><div class="empty-desc">Go to Load Data first.</div></div>', unsafe_allow_html=True)
        else:
            df=st.session_state["df"]; q=st.session_state.get("quality")
            if not q:
                st.info(f"Dataset ready: **{df.shape[0]:,} rows × {df.shape[1]} columns** — {st.session_state['filename']}")
                if st.button("🔍  Run quality analysis"):
                    with st.spinner("Analyzing data quality..."): q=analyze_quality(df); st.session_state["quality"]=q
                    with st.spinner("AI diagnosis..."):
                        diag=call_claude(f"Shape:{df.shape}\nColumns:{list(df.columns)}\nIssues:{json.dumps(q['issues'])}\nSample:\n{df.head(3).to_string()}",
                            "Write a 3-paragraph quality diagnosis. P1: what data is + overall health. P2: critical issues + business impact. P3: prioritized recommendations. Max 180 words.")
                        st.session_state["ai_diagnosis"]=diag
                    with st.spinner("Generating suggestions..."):
                        raw=call_claude(f"Columns:{list(df.columns)}\nDtypes:{df.dtypes.to_dict()}\nIssues:{json.dumps(q['issues'])}\nSample:\n{df.head(3).to_string()}",
                            'Suggest cleaning operations. Return ONLY JSON array:\n[{"action":str,"column":str,"description":str,"method":str,"value":str}]\nActions: drop_nulls,fill_nulls,drop_duplicates,trim_whitespace,convert_type,rename_column,drop_column,cap_outliers,uppercase,lowercase')
                        sugg=extract_json(raw); st.session_state["ai_suggestions"]=sugg if isinstance(sugg,list) else []
                    st.rerun()

            q=st.session_state.get("quality")
            if q:
                score=q["score"]; sc_color="#16a34a" if score>=80 else ("#d97706" if score>=60 else "#dc2626")
                c1,c2,c3,c4=st.columns(4)
                with c1: st.markdown(f'<div class="score-wrap"><div class="score-big" style="color:{sc_color}">{score}</div><div class="score-lbl">quality score / 100</div></div>', unsafe_allow_html=True)
                with c2: st.markdown(f'<div class="kpi-card amber"><div class="kpi-label">Missing values</div><div class="kpi-value">{q["null_total"]:,}</div><div class="kpi-insight">across all columns</div></div>', unsafe_allow_html=True)
                with c3: st.markdown(f'<div class="kpi-card red"><div class="kpi-label">Duplicate rows</div><div class="kpi-value">{q["dup_total"]:,}</div><div class="kpi-insight">exact matches</div></div>', unsafe_allow_html=True)
                with c4: st.markdown(f'<div class="kpi-card blue"><div class="kpi-label">Issues found</div><div class="kpi-value">{len(q["issues"])}</div><div class="kpi-insight">total checks flagged</div></div>', unsafe_allow_html=True)

                if q["issues"]:
                    st.markdown('<div class="sec-hd">Issues detected</div>', unsafe_allow_html=True)
                    for issue in q["issues"]:
                        sev=issue["severity"]
                        st.markdown(f'<div class="issue-row {sev}"><div class="issue-dot {sev}"></div><div><div class="issue-col">{issue["column"]}</div><div class="issue-type">{issue["type"]} — {issue["detail"]}</div></div></div>', unsafe_allow_html=True)

                if st.session_state.get("ai_diagnosis"):
                    st.markdown('<div class="sec-hd">AI diagnosis</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="ai-box"><div class="ai-box-hd">◆ Claude analysis</div>{st.session_state["ai_diagnosis"]}</div>', unsafe_allow_html=True)

                st.markdown('<div class="sec-hd">Cleaning operations</div>', unsafe_allow_html=True)
                ct=st.tabs(["AI suggestions","Manual","Formula","Edit cells"])

                with ct[0]:
                    sugg=st.session_state.get("ai_suggestions",[])
                    if sugg:
                        sel=[]
                        for i,s in enumerate(sugg):
                            if st.checkbox(f"**{s.get('action','').replace('_',' ').title()}** on `{s.get('column','')}` — {s.get('description','')}",key=f"s_{i}",value=True): sel.append(s)
                        if st.button("Apply selected"):
                            nd,entries=apply_cleaning(df,sel)
                            st.session_state["df"]=nd; st.session_state["clean_log"].extend(entries)
                            st.success(f"✓ {len(entries)} operations applied"); st.rerun()
                    else: st.info("Run quality analysis first.")

                with ct[1]:
                    with st.form("manual_clean"):
                        r1,r2=st.columns(2)
                        with r1: action=st.selectbox("Action",["fill_nulls","drop_nulls","drop_duplicates","trim_whitespace","convert_type","rename_column","drop_column","cap_outliers","uppercase","lowercase","replace_value"])
                        with r2: column=st.selectbox("Column",["(all)"]+list(df.columns))
                        c1,c2,c3=st.columns(3)
                        with c1: method=st.selectbox("Method",["mean","median","mode","value"])
                        with c2: fval=st.text_input("Fill value","")
                        with c3: nname=st.text_input("New name","")
                        ov=st.text_input("Old value (replace)",""); nv=st.text_input("New value (replace)","")
                        ttype=st.selectbox("Target type",["numeric","datetime","string"])
                        if st.form_submit_button("Apply operation"):
                            op={"action":action,"column":None if column=="(all)" else column,"method":method,
                                "value":fval or None,"new_name":nname or None,"old_value":ov or None,"new_value":nv or None,"target_type":ttype}
                            nd,entries=apply_cleaning(df,[op])
                            st.session_state["df"]=nd; st.session_state["clean_log"].extend(entries)
                            for e in entries: st.success(f"✓ {e}")
                            st.rerun()

                with ct[2]:
                    fcol=st.selectbox("Target column",list(df.columns),key="fcol")
                    fexpr=st.text_area("Expression",placeholder="e.g. revenue * 0.2",height=70,label_visibility="collapsed")
                    if st.button("Apply formula"):
                        try:
                            nd=df.copy(); nd[fcol]=df.eval(fexpr)
                            st.session_state["df"]=nd; st.session_state["clean_log"].append(f"Formula on '{fcol}': {fexpr}")
                            st.success("✓ Applied"); st.rerun()
                        except Exception as e: st.error(f"Error: {e}")

                with ct[3]:
                    edited=st.data_editor(df,use_container_width=True,num_rows="dynamic",key="cell_ed")
                    if st.button("Save edits"):
                        st.session_state["df"]=edited; st.session_state["clean_log"].append("Manual cell edits saved")
                        st.success("✓ Saved"); st.rerun()

                clog=st.session_state["clean_log"]
                if clog:
                    st.markdown('<div class="sec-hd">Cleaning log</div>', unsafe_allow_html=True)
                    for entry in clog[-15:]:
                        cls="err" if "[ERROR]" in entry else ("etl" if "[ETL]" in entry else "ok")
                        st.markdown(f'<div class="log-entry {cls}">{"✗" if cls=="err" else "✓"} {entry}</div>', unsafe_allow_html=True)
                    rc,_=st.columns([1,4])
                    with rc:
                        if st.button("↩ Reset to original"):
                            st.session_state["df"]=st.session_state["df_original"].copy()
                            st.session_state["clean_log"]=[]; st.session_state["quality"]=None; st.rerun()

                dfc=st.session_state["df"]
                st.markdown('<div class="sec-hd">Export</div>', unsafe_allow_html=True)
                e1,e2,e3=st.columns(3)
                with e1: st.download_button("⬇ CSV",dfc.to_csv(index=False).encode(),f"cleaned_{st.session_state['filename']}.csv","text/csv",use_container_width=True)
                with e2:
                    buf=io.BytesIO()
                    with pd.ExcelWriter(buf,engine="openpyxl") as w: dfc.to_excel(w,index=False)
                    st.download_button("⬇ Excel",buf.getvalue(),f"cleaned_{st.session_state['filename']}.xlsx",use_container_width=True)
                with e3:
                    lt="DataOps AI Log\n"+"="*40+"\n"+"\n".join(f"{i+1}. {e}" for i,e in enumerate(clog))
                    st.download_button("⬇ Log",lt.encode(),"cleaning_log.txt",use_container_width=True)

    # ── ETL ───────────────────────────────────────────────────────────────────
    elif page == "etl":
        st.markdown('<div class="page-header"><div class="pg-eyebrow">Module 02</div><div class="pg-title">ETL & Transformation</div><div class="pg-desc">AI SQL generator, column operations, joins and pivots.</div></div>', unsafe_allow_html=True)
        if not has_data:
            st.markdown('<div class="empty-state"><div class="empty-icon">⚡</div><div class="empty-title">No data loaded</div></div>', unsafe_allow_html=True)
        else:
            df=st.session_state["df"]
            et=st.tabs(["AI SQL","Column ops","Merge / Join","Pivot & Reshape","AI Plan"])

            with et[0]:
                st.markdown('<div class="sec-hd">Describe what you want — get code</div>', unsafe_allow_html=True)
                schema=", ".join([f"{c}({df[c].dtype})" for c in df.columns])
                sp=st.text_area("What transformation?",placeholder="e.g. Total revenue per region sorted by highest\ne.g. Keep rows where status = active",height=90,label_visibility="collapsed")
                if st.button("Generate code",key="gen_sql"):
                    if sp.strip():
                        with st.spinner("Writing code..."):
                            raw=call_claude(f"Schema:{schema}\nSample:\n{df.head(2).to_string()}\nRequest:{sp}",
                                'Return ONLY JSON: {"sql":"pandas code using df, result must be result_df","explanation":"plain language"}')
                            parsed=extract_json(raw)
                        if parsed:
                            st.session_state["etl_sql"]=parsed
                            st.markdown(f'<div class="code-block">{parsed.get("sql","")}</div>', unsafe_allow_html=True)
                            st.caption(f"💡 {parsed.get('explanation','')}")
                            try:
                                g={"df":df.copy(),"pd":pd,"np":np}; exec(parsed["sql"],g)
                                result=g.get("result_df",g.get("df")); st.session_state["etl_preview"]=result
                                st.markdown(f"**Preview** — {result.shape[0]:,} rows × {result.shape[1]} cols")
                                st.dataframe(result.head(15),use_container_width=True)
                            except Exception as e: st.error(f"Execution error: {e}")
                        else: st.error("Could not parse. Try rephrasing.")
                if st.session_state.get("etl_preview") is not None:
                    if st.button("✓ Apply to dataset"):
                        st.session_state["df"]=st.session_state["etl_preview"]
                        st.session_state["clean_log"].append(f"[ETL] SQL: {sp[:80]}")
                        st.session_state.pop("etl_preview",None); st.success("✓ Applied"); st.rerun()

            with et[1]:
                with st.form("col_ops"):
                    r1,r2=st.columns(2)
                    with r1: op_type=st.selectbox("Operation",["Add calculated column","Split by delimiter","Merge two columns","Extract date part","Normalize (0–1)","Standardize (z-score)","Bin into categories","Map values"])
                    with r2: tcol=st.selectbox("Column",list(df.columns))
                    c1,c2,c3=st.columns(3)
                    with c1: new_col=st.text_input("New column name","")
                    with c2: formula_in=st.text_input("Formula / delimiter / 2nd col","")
                    with c3: bins_in=st.text_input("Bins (e.g. 0,100,500)","")
                    labels_in=st.text_input("Bin labels (e.g. Low,Med,High)","")
                    map_in=st.text_area('Value map JSON (e.g. {"Y":1,"N":0})','',height=55)
                    dpart=st.selectbox("Date part",["year","month","day","weekday","quarter"])
                    if st.form_submit_button("Apply"):
                        try:
                            nd=df.copy(); col_nm=new_col or f"{tcol}_new"; entry=""
                            if op_type=="Add calculated column": nd[col_nm]=nd.eval(formula_in); entry=f"Added '{col_nm}' = {formula_in}"
                            elif op_type=="Split by delimiter":
                                sp2=nd[tcol].astype(str).str.split(formula_in or ",",expand=True)
                                for i2,c2 in enumerate(sp2.columns): nd[f"{tcol}_p{i2+1}"]=sp2[c2]
                                entry=f"Split '{tcol}'"
                            elif op_type=="Merge two columns":
                                if bins_in in nd.columns: nd[col_nm]=nd[tcol].astype(str)+(formula_in or " ")+nd[bins_in].astype(str); entry=f"Merged '{tcol}'+"
                            elif op_type=="Extract date part":
                                nd[tcol]=pd.to_datetime(nd[tcol],errors="coerce"); nd[col_nm]=getattr(nd[tcol].dt,dpart); entry=f"Extracted {dpart}"
                            elif op_type=="Normalize (0–1)":
                                mn,mx=nd[tcol].min(),nd[tcol].max(); nd[col_nm]=(nd[tcol]-mn)/(mx-mn); entry=f"Normalized '{tcol}'"
                            elif op_type=="Standardize (z-score)":
                                mu,sigma=nd[tcol].mean(),nd[tcol].std(); nd[col_nm]=(nd[tcol]-mu)/sigma; entry=f"Standardized '{tcol}'"
                            elif op_type=="Bin into categories":
                                edges=[float(x.strip()) for x in bins_in.split(",")]; lbls=[x.strip() for x in labels_in.split(",")]
                                nd[col_nm]=pd.cut(nd[tcol],bins=edges,labels=lbls if len(lbls)==len(edges)-1 else None); entry=f"Binned '{tcol}'"
                            elif op_type=="Map values":
                                mapping=json.loads(map_in); nd[col_nm]=nd[tcol].map(mapping).fillna(nd[tcol]); entry=f"Mapped '{tcol}'"
                            st.session_state["df"]=nd; st.session_state["clean_log"].append(f"[ETL] {entry}")
                            st.success(f"✓ {entry}"); st.rerun()
                        except Exception as e: st.error(f"Error: {e}")

            with et[2]:
                sf=st.file_uploader("Upload second dataset",type=["csv","xlsx"],key="merge_file")
                if sf:
                    df2=load_file(sf); st.dataframe(df2.head(5),use_container_width=True)
                    with st.form("merge_form"):
                        c1,c2,c3=st.columns(3)
                        with c1: lk=st.selectbox("Left key",df.columns.tolist())
                        with c2: rk=st.selectbox("Right key",df2.columns.tolist())
                        with c3: how=st.selectbox("Join type",["inner","left","right","outer"])
                        if st.form_submit_button("Merge"):
                            try:
                                merged=pd.merge(df,df2,left_on=lk,right_on=rk,how=how)
                                st.session_state["df"]=merged; st.session_state["clean_log"].append(f"[ETL] {how.upper()} JOIN → {merged.shape[0]:,} rows")
                                st.success(f"✓ {merged.shape[0]:,} rows"); st.rerun()
                            except Exception as e: st.error(f"Error: {e}")

            with et[3]:
                rtype=st.radio("",["Pivot table","Melt (wide→long)","Transpose"],horizontal=True,label_visibility="collapsed")
                nc2=df.select_dtypes(include="number").columns.tolist()
                if rtype=="Pivot table":
                    with st.form("pivot"):
                        c1,c2,c3=st.columns(3)
                        with c1: pi=st.selectbox("Index",list(df.columns))
                        with c2: pc=st.selectbox("Columns",list(df.columns))
                        with c3: pv=st.selectbox("Values",nc2 if nc2 else list(df.columns))
                        agg=st.selectbox("Aggregation",["sum","mean","count","min","max","median"])
                        if st.form_submit_button("Create pivot"):
                            try:
                                p=df.pivot_table(index=pi,columns=pc,values=pv,aggfunc=agg).reset_index()
                                p.columns=[str(c) for c in p.columns]
                                st.session_state["df"]=p; st.session_state["clean_log"].append(f"[ETL] Pivot {pi}×{pc} ({agg})")
                                st.success(f"✓ {p.shape[0]} rows × {p.shape[1]} cols"); st.rerun()
                            except Exception as e: st.error(f"Error: {e}")
                elif rtype=="Melt (wide→long)":
                    with st.form("melt"):
                        idv=st.multiselect("ID columns",list(df.columns)); vn=st.text_input("Value name","value"); vrn=st.text_input("Variable name","variable")
                        if st.form_submit_button("Melt"):
                            try:
                                m=df.melt(id_vars=idv,var_name=vrn,value_name=vn)
                                st.session_state["df"]=m; st.session_state["clean_log"].append(f"[ETL] Melt → {m.shape[0]:,} rows")
                                st.success(f"✓ {m.shape[0]:,} rows"); st.rerun()
                            except Exception as e: st.error(f"Error: {e}")
                else:
                    if st.button("Transpose"):
                        t=df.T.reset_index(); t.columns=[f"col_{i}" for i in range(len(t.columns))]
                        st.session_state["df"]=t; st.session_state["clean_log"].append("[ETL] Transposed"); st.rerun()

            with et[4]:
                goal=st.text_area("Describe your end goal",placeholder="e.g. Monthly revenue per region, ready for Power BI",height=100,label_visibility="collapsed")
                if st.button("Generate transformation plan"):
                    if goal.strip():
                        with st.spinner("Building plan..."):
                            si="\n".join([f"- {c}: {df[c].dtype} ({df[c].nunique()} unique)" for c in df.columns])
                            plan=call_claude(f"Dataset:{df.shape[0]:,}×{df.shape[1]}\nColumns:\n{si}\nSample:\n{df.head(3).to_string()}\nGoal:{goal}",
                                "Write a numbered step-by-step transformation plan. End with Power BI usage instructions.",max_tokens=1200)
                            st.session_state["etl_plan"]=plan
                if st.session_state.get("etl_plan"):
                    st.markdown(f'<div class="ai-box"><div class="ai-box-hd">◆ Transformation plan</div>{st.session_state["etl_plan"]}</div>', unsafe_allow_html=True)

    # ── REPORTS ───────────────────────────────────────────────────────────────
    elif page == "reports":
        st.markdown('<div class="page-header"><div class="pg-eyebrow">Module 03</div><div class="pg-title">Reports & Charts</div><div class="pg-desc">KPI cards, smart charts, statistical exploration and executive report.</div></div>', unsafe_allow_html=True)
        if not has_data:
            st.markdown('<div class="empty-state"><div class="empty-icon">📊</div><div class="empty-title">No data loaded</div></div>', unsafe_allow_html=True)
        else:
            df=st.session_state["df"]
            nc=df.select_dtypes(include="number").columns.tolist()
            cc=df.select_dtypes(include=["object","category"]).columns.tolist()
            rt=st.tabs(["KPI Cards","Smart Charts","Explore","Executive Report","Custom Chart"])

            with rt[0]:
                if st.button("Generate KPIs",key="gen_kpis"):
                    with st.spinner("Extracting KPIs..."):
                        raw=call_claude(f"Columns:{list(df.columns)}\nStats:\n{df.describe().to_string()}\nSample:\n{df.head(5).to_string()}",
                            'Extract 4-6 KPIs. Return ONLY JSON:\n[{"label":str,"value":str,"delta":str,"direction":"up"|"down"|"neutral","insight":str,"color":"blue"|"green"|"red"|"amber"|"purple"}]')
                        p=extract_json(raw)
                        if isinstance(p,list): st.session_state["kpis_rep"]=p
                kpis=st.session_state.get("kpis_rep",[])
                if kpis:
                    ck=st.columns(min(3,len(kpis)))
                    for i,kpi in enumerate(kpis):
                        d=kpi.get("direction","neutral"); arrow="↑" if d=="up" else ("↓" if d=="down" else "→"); dcls="up" if d=="up" else ("down" if d=="down" else "flat")
                        with ck[i%3]: st.markdown(f'<div class="kpi-card {kpi.get("color","blue")}"><div class="kpi-label">{kpi.get("label","")}</div><div class="kpi-value">{kpi.get("value","—")}</div><div class="kpi-delta {dcls}">{arrow} {kpi.get("delta","")}</div><div class="kpi-insight">{kpi.get("insight","")}</div></div>', unsafe_allow_html=True)
                else: st.markdown('<div class="empty-state"><div class="empty-icon">📈</div><div class="empty-title">Click Generate KPIs</div></div>', unsafe_allow_html=True)

            with rt[1]:
                if st.button("Generate smart charts",key="gen_charts"):
                    with st.spinner("Choosing best charts..."):
                        schema=", ".join([f"{c}({df[c].dtype})" for c in df.columns])
                        raw=call_claude(f"Columns:{schema}\nSample:\n{df.head(4).to_string()}",
                            'Recommend 4 charts. Return ONLY JSON:\n[{"title":str,"type":"bar"|"line"|"area"|"pie"|"scatter"|"histogram"|"box","x":str|null,"y":str|null,"color":str|null,"agg":"sum"|"mean"|"count"|"none","rationale":str}]\nOnly use existing columns.')
                        p=extract_json(raw)
                        if isinstance(p,list): st.session_state["charts_rep"]=p
                charts=st.session_state.get("charts_rep",[])
                if charts:
                    for i in range(0,len(charts),2):
                        pair=charts[i:i+2]; cols_ch=st.columns(len(pair))
                        for ci,cfg in enumerate(pair):
                            with cols_ch[ci]:
                                fig=make_chart(cfg,df)
                                if fig: st.plotly_chart(fig,use_container_width=True); st.caption(f"💡 {cfg.get('rationale','')}")
                else: st.markdown('<div class="empty-state"><div class="empty-icon">📊</div><div class="empty-title">Click Generate smart charts</div></div>', unsafe_allow_html=True)

            with rt[2]:
                etype=st.radio("",["Correlation heatmap","Distribution","Value counts"],horizontal=True,label_visibility="collapsed")
                if etype=="Correlation heatmap":
                    if len(nc)>=2:
                        fig=px.imshow(df[nc].corr(),text_auto=".2f",aspect="auto",color_continuous_scale="RdBu_r",title="Correlation matrix",zmin=-1,zmax=1)
                        fig.update_layout(font_family="Plus Jakarta Sans",paper_bgcolor="white",margin=dict(l=8,r=8,t=40,b=8)); st.plotly_chart(fig,use_container_width=True)
                    else: st.info("Need 2+ numeric columns.")
                elif etype=="Distribution":
                    if nc:
                        cs=st.selectbox("Column",nc)
                        fig=px.histogram(df,x=cs,marginal="box",color_discrete_sequence=[COLORS[0]],title=f"Distribution of {cs}")
                        fig.update_layout(font_family="Plus Jakarta Sans",paper_bgcolor="white",plot_bgcolor="white",margin=dict(l=8,r=8,t=40,b=8))
                        fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor="#f1f5f9"); st.plotly_chart(fig,use_container_width=True)
                        c1,c2,c3,c4=st.columns(4)
                        c1.metric("Mean",f"{df[cs].mean():.2f}"); c2.metric("Median",f"{df[cs].median():.2f}")
                        c3.metric("Std dev",f"{df[cs].std():.2f}"); c4.metric("Skewness",f"{df[cs].skew():.2f}")
                elif etype=="Value counts":
                    if cc:
                        cs=st.selectbox("Column",cc); topn=st.slider("Top N",5,30,10)
                        vc=df[cs].value_counts().head(topn).reset_index(); vc.columns=[cs,"count"]
                        fig=px.bar(vc,x=cs,y="count",color_discrete_sequence=[COLORS[1]],title=f"Top {topn} in '{cs}'")
                        fig.update_layout(font_family="Plus Jakarta Sans",paper_bgcolor="white",plot_bgcolor="white",margin=dict(l=8,r=8,t=40,b=8))
                        fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor="#f1f5f9"); st.plotly_chart(fig,use_container_width=True)

            with rt[3]:
                c1,c2=st.columns(2)
                with c1: lang=st.radio("Language",["French","English"],horizontal=True)
                with c2: aud=st.selectbox("Audience",["Senior manager","C-level","Data team","Client"])
                ctx=st.text_input("Context (optional)",placeholder="e.g. Q1 2025 sales data, Morocco")
                if st.button("Generate report",key="gen_rep"):
                    with st.spinner("Writing executive report..."):
                        report=call_claude(
                            f"Context:{ctx or 'N/A'}\nShape:{df.shape}\nColumns:{list(df.columns)}\nStats:\n{df.describe().to_string()}\nKPIs:{json.dumps(st.session_state.get('kpis_rep',[]))}\nOps:{chr(10).join(st.session_state['clean_log'][-10:])}",
                            f"Write a formal executive report in {lang} for {aud}. Sections: 1)Executive Summary 2)Dataset Overview 3)Key Findings 4)Risks & Anomalies 5)Recommendations 6)Next Steps. Max 400 words.",max_tokens=1500)
                        st.session_state["exec_report"]=report
                if st.session_state.get("exec_report"):
                    st.markdown(f'<div class="ai-box"><div class="ai-box-hd">◆ Executive Report</div>{st.session_state["exec_report"]}</div>', unsafe_allow_html=True)
                    st.download_button("⬇ Download report",st.session_state["exec_report"].encode(),"executive_report.txt")

            with rt[4]:
                all_c=list(df.columns)
                with st.form("custom_chart"):
                    r1,r2,r3=st.columns(3)
                    with r1: ctype=st.selectbox("Type",["Bar","Line","Area","Scatter","Pie","Histogram","Box","Funnel"])
                    with r2: cx=st.selectbox("X",["(none)"]+all_c)
                    with r3: cy=st.selectbox("Y",["(none)"]+all_c)
                    r4,r5,r6=st.columns(3)
                    with r4: ccolor=st.selectbox("Color by",["(none)"]+all_c)
                    with r5: cagg=st.selectbox("Aggregate Y",["none","sum","mean","count","max","min"])
                    with r6: ctitle=st.text_input("Title","My chart")
                    ch=st.slider("Height",300,700,400)
                    if st.form_submit_button("Build chart"):
                        try:
                            x=cx if cx!="(none)" else None; y=cy if cy!="(none)" else None; color=ccolor if ccolor!="(none)" else None
                            pf=df.copy()
                            if cagg!="none" and x and y: pf=getattr(pf.groupby(x)[y],cagg)().reset_index(); color=None
                            kw=dict(color_discrete_sequence=COLORS,title=ctitle,height=ch); fig=None
                            if ctype=="Bar": fig=px.bar(pf,x=x,y=y,color=color,**kw)
                            elif ctype=="Line": fig=px.line(pf,x=x,y=y,color=color,**kw)
                            elif ctype=="Area": fig=px.area(pf,x=x,y=y,color=color,**kw)
                            elif ctype=="Scatter": fig=px.scatter(pf,x=x,y=y,color=color,**kw)
                            elif ctype=="Pie": fig=px.pie(pf,names=x,values=y,color_discrete_sequence=COLORS,title=ctitle,height=ch)
                            elif ctype=="Histogram": fig=px.histogram(pf,x=x or y,color=color,**kw)
                            elif ctype=="Box": fig=px.box(pf,x=x,y=y,color=color,**kw)
                            elif ctype=="Funnel": fig=px.funnel(pf,x=y,y=x,color=color,**kw)
                            if fig:
                                fig.update_layout(font_family="Plus Jakarta Sans",paper_bgcolor="white",plot_bgcolor="white",margin=dict(l=8,r=8,t=50,b=8))
                                fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor="#f1f5f9")
                                st.session_state["custom_chart"]=fig
                        except Exception as e: st.error(f"Error: {e}")
                if st.session_state.get("custom_chart"): st.plotly_chart(st.session_state["custom_chart"],use_container_width=True)

    # ── DOCUMENTATION ─────────────────────────────────────────────────────────
    elif page == "docs":
        st.markdown('<div class="page-header"><div class="pg-eyebrow">Module 04</div><div class="pg-title">Documentation</div><div class="pg-desc">Auto-generate a data dictionary, transformation changelog, README and lineage map.</div></div>', unsafe_allow_html=True)
        if not has_data:
            st.markdown('<div class="empty-state"><div class="empty-icon">📄</div><div class="empty-title">No data loaded</div></div>', unsafe_allow_html=True)
        else:
            df=st.session_state["df"]
            dt=st.tabs(["Data Dictionary","Changelog","README","Lineage Map","Export All"])

            with dt[0]:
                st.markdown('<div class="sec-hd">Column-by-column business definitions</div>', unsafe_allow_html=True)
                if st.button("Generate data dictionary",key="gen_dict"):
                    with st.spinner("Analyzing columns..."):
                        profiles=[{"column":c,"dtype":str(df[c].dtype),"null_count":int(df[c].isnull().sum()),
                            "unique_count":int(df[c].nunique()),"sample_values":[str(s) for s in df[c].dropna().head(3).tolist()]}
                            for c in df.columns]
                        raw=call_claude(
                            f"Dataset:{df.shape[0]:,} rows\nProfiles:\n{json.dumps(profiles,indent=2)}\nSample:\n{df.head(3).to_string()}",
                            'Write a data dictionary entry for each column. Return ONLY JSON:\n[{"column":str,"business_definition":str,"data_type":str,"format_notes":str,"quality_status":"Good"|"Warning"|"Critical","quality_note":str,"example":str}]',
                            max_tokens=2000)
                        p=extract_json(raw)
                        if isinstance(p,list): st.session_state["doc_dictionary"]=p
                        else: st.error("Could not parse. Try again.")

                dictionary=st.session_state.get("doc_dictionary")
                if dictionary:
                    for e in dictionary:
                        s=e.get("quality_status","Good"); tc={"Good":"green","Warning":"amber","Critical":"red"}.get(s,"green")
                        st.markdown(f"""
                        <div class="doc-card">
                            <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.5rem">
                                <span style="font-family:'JetBrains Mono',monospace;font-weight:500;font-size:.88rem;color:#0f172a">{e.get('column','')}</span>
                                <span class="tag blue">{e.get('data_type','')}</span>
                                <span class="tag {tc}">{s}</span>
                            </div>
                            <div style="font-size:.85rem;color:#334155;margin-bottom:.5rem">{e.get('business_definition','')}</div>
                            <div style="display:flex;gap:2rem;flex-wrap:wrap">
                                <div><span style="font-size:.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;font-family:'JetBrains Mono',monospace">Format</span><div style="font-size:.78rem;color:#64748b;margin-top:.1rem">{e.get('format_notes','')}</div></div>
                                <div><span style="font-size:.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;font-family:'JetBrains Mono',monospace">Example</span><div style="font-family:'JetBrains Mono',monospace;font-size:.78rem;color:#0f172a;margin-top:.1rem">{e.get('example','')}</div></div>
                                <div><span style="font-size:.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;font-family:'JetBrains Mono',monospace">Quality note</span><div style="font-size:.78rem;color:#64748b;margin-top:.1rem">{e.get('quality_note','')}</div></div>
                            </div>
                        </div>""", unsafe_allow_html=True)
                else: st.markdown('<div class="empty-state"><div class="empty-icon">📋</div><div class="empty-title">Click Generate data dictionary</div><div class="empty-desc">Claude will define every column with business context.</div></div>', unsafe_allow_html=True)

            with dt[1]:
                st.markdown('<div class="sec-hd">Every transformation — timestamped</div>', unsafe_allow_html=True)
                clog=st.session_state.get("clean_log",[])
                if not clog:
                    st.markdown('<div class="empty-state"><div class="empty-icon">📝</div><div class="empty-title">No operations yet</div><div class="empty-desc">Apply cleaning or ETL operations first.</div></div>', unsafe_allow_html=True)
                else:
                    if st.button("Generate AI changelog summary",key="gen_clog"):
                        with st.spinner("Summarizing..."):
                            orig=st.session_state.get("df_original")
                            cl=call_claude(
                                "Operations:\n"+"\n".join(f"{i+1}. {e}" for i,e in enumerate(clog))+f"\nShape: {orig.shape if orig is not None else 'unknown'} → {df.shape}",
                                "Summarize these data transformations as a professional changelog. Format: ## Summary, ## Operations (numbered), ## Shape Changes, ## Recommendations. Be concise.",max_tokens=800)
                            st.session_state["doc_changelog"]=cl
                    ts=datetime.now().strftime("%H:%M")
                    for i,entry in enumerate(clog):
                        cls="err" if "[ERROR]" in entry else ("etl" if "[ETL]" in entry else ("doc" if "[DOC]" in entry else "ok"))
                        st.markdown(f'<div class="log-entry {cls}" style="display:flex;gap:.8rem"><span style="color:#94a3b8;min-width:2rem">{i+1:02d}</span><span style="flex:1">{entry}</span><span style="color:#94a3b8;font-size:.65rem">{ts}</span></div>', unsafe_allow_html=True)
                    if st.session_state.get("doc_changelog"):
                        st.markdown('<div class="sec-hd">AI summary</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="ai-box">{st.session_state["doc_changelog"]}</div>', unsafe_allow_html=True)

            with dt[2]:
                st.markdown('<div class="sec-hd">GitHub-ready README for your dataset</div>', unsafe_allow_html=True)
                rctx=st.text_input("Project context (optional)",placeholder="e.g. Monthly sales data for Morocco Q1 2025")
                if st.button("Generate README",key="gen_readme"):
                    with st.spinner("Writing README..."):
                        d=st.session_state.get("doc_dictionary",[])
                        col_s="\n".join([f"- `{e['column']}`: {e.get('business_definition','')}" for e in d]) if d else "\n".join([f"- `{c}`: {df[c].dtype}" for c in df.columns])
                        q=st.session_state.get("quality")
                        readme=call_claude(
                            f"File:{st.session_state.get('filename','dataset')}\nShape:{df.shape[0]:,}×{df.shape[1]}\nContext:{rctx or 'N/A'}\nColumns:\n{col_s}\nQuality score:{q['score'] if q else 'N/A'}\nOps:{len(clog if (clog:=st.session_state.get('clean_log',[])) else [])}\nAuthor: Mohammed Amine Goumri",
                            "Write a professional markdown README for this dataset. Include: # Title, ## Overview, ## Schema (table), ## Data Quality, ## Transformations, ## Usage, ## Author. GitHub-ready.",max_tokens=1200)
                        st.session_state["doc_readme"]=readme
                if st.session_state.get("doc_readme"):
                    st.markdown(f'<div class="ai-box">{st.session_state["doc_readme"]}</div>', unsafe_allow_html=True)
                    st.download_button("⬇ README.md",st.session_state["doc_readme"].encode(),"README.md","text/markdown")
                else: st.markdown('<div class="empty-state"><div class="empty-icon">📖</div><div class="empty-title">Click Generate README</div></div>', unsafe_allow_html=True)

            with dt[3]:
                st.markdown('<div class="sec-hd">Source → transformations → final output</div>', unsafe_allow_html=True)
                if st.button("Generate lineage map",key="gen_lin"):
                    with st.spinner("Building lineage..."):
                        clog2=st.session_state.get("clean_log",[]); orig=st.session_state.get("df_original")
                        lin=call_claude(
                            f"Source:{st.session_state.get('filename','unknown')}\nOriginal:{orig.shape if orig is not None else 'unknown'}\nFinal:{df.shape}\nOrig cols:{list(orig.columns) if orig is not None else []}\nFinal cols:{list(df.columns)}\nOps:\n"+"\n".join(f"{i+1}. {e}" for i,e in enumerate(clog2)),
                            "Write a data lineage document. Sections: ## Source, ## Transformation Steps (numbered, with input→output shape), ## Final Output, ## Column Lineage (what changed per column). Be precise.",max_tokens=1200)
                        st.session_state["doc_lineage"]=lin
                if st.session_state.get("doc_lineage"):
                    orig=st.session_state.get("df_original"); clog3=st.session_state.get("clean_log",[])
                    etl_n=sum(1 for l in clog3 if "[ETL]" in l); clean_n=len(clog3)-etl_n
                    stages=[("Source",st.session_state.get('filename','file'),f"{orig.shape[0]:,}×{orig.shape[1]}" if orig is not None else "—","#1d4ed8"),
                        ("Cleaned",f"{clean_n} ops",f"→ {df.shape[0]:,} rows","#7c3aed"),
                        ("Transformed",f"{etl_n} ETL ops",f"→ {df.shape[1]} cols","#0ea5e9"),
                        ("Output","Final dataset",f"{df.shape[0]:,}×{df.shape[1]}","#16a34a")]
                    sc=st.columns(len(stages))
                    for i,(title,sub,stat,color) in enumerate(stages):
                        with sc[i]:
                            st.markdown(f'<div style="background:#fff;border:1px solid #e2e8f0;border-top:3px solid {color};border-radius:10px;padding:1rem;text-align:center"><div style="font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:{color};font-family:JetBrains Mono,monospace;margin-bottom:.3rem">{title}</div><div style="font-weight:600;font-size:.85rem;color:#0f172a;margin-bottom:.2rem">{sub}</div><div style="font-family:JetBrains Mono,monospace;font-size:.75rem;color:#94a3b8">{stat}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="ai-box" style="margin-top:1rem"><div class="ai-box-hd">◆ Lineage report</div>{st.session_state["doc_lineage"]}</div>', unsafe_allow_html=True)
                else: st.markdown('<div class="empty-state"><div class="empty-icon">🗺</div><div class="empty-title">Click Generate lineage map</div></div>', unsafe_allow_html=True)

            with dt[4]:
                st.markdown('<div class="sec-hd">Complete documentation package</div>', unsafe_allow_html=True)
                hd=st.session_state.get("doc_dictionary") is not None; hr=st.session_state.get("doc_readme") is not None
                hc=st.session_state.get("doc_changelog") is not None; hl=st.session_state.get("doc_lineage") is not None
                c1,c2=st.columns(2)
                with c1:
                    st.markdown(f'<div class="doc-card"><div class="doc-card-title">Data Dictionary</div><div class="doc-card-meta">{"✓ Ready" if hd else "⚪ Not generated"} · {len(st.session_state.get("doc_dictionary") or [])} columns</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="doc-card"><div class="doc-card-title">Changelog</div><div class="doc-card-meta">{"✓ Ready" if hc else "⚪ Not generated"} · {len(st.session_state.get("clean_log",[]))} operations</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="doc-card"><div class="doc-card-title">README.md</div><div class="doc-card-meta">{"✓ Ready" if hr else "⚪ Not generated"} · GitHub-ready</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="doc-card"><div class="doc-card-title">Lineage Map</div><div class="doc-card-meta">{"✓ Ready" if hl else "⚪ Not generated"} · Source to output</div></div>', unsafe_allow_html=True)

                doc_pkg=f"# DataOps AI — Documentation Package\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\nDataset: {st.session_state.get('filename','dataset')}\nAuthor: Mohammed Amine Goumri\n\n"+"="*60+"\n\n"
                if hd:
                    doc_pkg+="# DATA DICTIONARY\n\n"
                    for e in st.session_state["doc_dictionary"]:
                        doc_pkg+=f"## {e.get('column','')}\n- Type: {e.get('data_type','')}\n- Definition: {e.get('business_definition','')}\n- Format: {e.get('format_notes','')}\n- Quality: {e.get('quality_status','')} — {e.get('quality_note','')}\n- Example: {e.get('example','')}\n\n"
                if hc: doc_pkg+="\n"+"="*60+"\n\n# CHANGELOG\n\n"+st.session_state["doc_changelog"]+"\n\n"
                if hr: doc_pkg+="\n"+"="*60+"\n\n# README\n\n"+st.session_state["doc_readme"]+"\n\n"
                if hl: doc_pkg+="\n"+"="*60+"\n\n# LINEAGE\n\n"+st.session_state["doc_lineage"]+"\n\n"
                clog4=st.session_state.get("clean_log",[])
                if clog4: doc_pkg+="\n"+"="*60+"\n\n# RAW LOG\n\n"+"\n".join(f"{i+1:03d}. {e}" for i,e in enumerate(clog4))

                st.markdown("<br>", unsafe_allow_html=True)
                ex1,ex2,ex3=st.columns(3)
                with ex1: st.download_button("⬇ Full package (.txt)",doc_pkg.encode(),"dataops_documentation.txt",use_container_width=True)
                with ex2:
                    if hr: st.download_button("⬇ README.md",st.session_state["doc_readme"].encode(),"README.md","text/markdown",use_container_width=True)
                with ex3:
                    dfc=st.session_state["df"]; buf=io.BytesIO()
                    with pd.ExcelWriter(buf,engine="openpyxl") as w:
                        dfc.to_excel(w,index=False,sheet_name="Data")
                        if hd: pd.DataFrame(st.session_state["doc_dictionary"]).to_excel(w,index=False,sheet_name="Dictionary")
                        if clog4: pd.DataFrame({"operation":clog4}).to_excel(w,index=False,sheet_name="Changelog")
                    st.download_button("⬇ Excel + dictionary",buf.getvalue(),f"dataops_{st.session_state.get('filename','data')}.xlsx",use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="text-align:center;padding:1.5rem;border-top:1px solid #f1f5f9;margin-top:2rem;font-family:JetBrains Mono,monospace;font-size:.68rem;color:#cbd5e1">DataOps AI · All 4 Modules Active · Built by Mohammed Amine Goumri · Powered by Claude</div>', unsafe_allow_html=True)
