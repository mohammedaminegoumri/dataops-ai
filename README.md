# ⚙ DataOps AI

> A BI Automation Agent that replicates what a Power BI Analyst does — data quality checks, cleaning, transformation, reporting and documentation — powered by Claude AI.

Built by **Mohammed Amine Goumri** · Data Scientist & Business Analyst

---

## Modules (build roadmap)

| Module | Status | Description |
|--------|--------|-------------|
| 01 · Data Quality & Cleaning | ✅ Live | Null checks, outlier detection, duplicates, AI diagnosis, interactive cleaning |
| 02 · ETL & Transformation | 🔜 Next | AI-generated SQL, column mapping, data type normalization |
| 03 · Report & Dashboard | 🔜 Soon | Auto KPI cards, trend charts, executive summary |
| 04 · Documentation | 🔜 Soon | Data dictionary, transformation log, Git-ready changelog |

---

## Module 01 — What it does

**Input**: CSV, Excel, or SQL database connection

**Quality analysis**:
- Quality score out of 100
- Null % per column with severity rating
- Duplicate row detection
- Type mismatch detection (numeric stored as text)
- Outlier flagging (3×IQR rule)
- Whitespace / formatting issues
- AI narrative diagnosis (3 paragraphs)

**Cleaning operations** (human approves everything):
- AI-suggested operations (one-click apply)
- Manual operations: fill nulls (mean/median/mode/value), drop nulls, drop duplicates, trim whitespace, convert type, rename column, drop column, cap outliers, uppercase/lowercase, replace value
- Custom formula editor (pandas eval)
- Direct cell editing via interactive table

**Export**:
- Cleaned CSV
- Cleaned Excel
- Cleaning log (.txt) — full audit trail of every operation

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/dataops-ai.git
cd dataops-ai
pip install -r requirements.txt
# Add ANTHROPIC_API_KEY to .streamlit/secrets.toml
streamlit run app.py
```

---

## Tech Stack

- **Frontend**: Streamlit
- **AI**: Claude (Anthropic API) — `claude-haiku-4-5-20251001`
- **Data**: Pandas · NumPy · SQLAlchemy
- **Charts**: Plotly
- **DB support**: PostgreSQL, MySQL, SQLite, and any SQLAlchemy-compatible DB

---

## Author

**Mohammed Amine Goumri**  
MSc Data Science & Business Analytics — Université Internationale de Rabat  
Former Data Scientist @ Bank Al-Maghrib  
mohammedaminegoumri@proton.me
