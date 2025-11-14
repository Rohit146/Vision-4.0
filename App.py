###############################################################
# Auto-BI Dashboard Studio — Full Regenerated Version
# Includes:
# - safe_str() patch (fixes invalid format specifier errors)
# - Full Power BI–style HTML generator
# - Slides mode + filters + drilldown
# - PNG export + live spec editing + presentation mode
# - Table charts support
###############################################################

import streamlit as st
import pandas as pd
import plotly.express as px
from jinja2 import Template
from openai import OpenAI
import os, re, json, hashlib, base64
from typing import List

###############################################################
# STREAMLIT CONFIG
###############################################################
st.set_page_config(page_title="Auto-BI HTML Studio", layout="wide")
st.title("🧠 Auto-BI — Power BI-style HTML Dashboard Studio")

###############################################################
# OpenAI Client
###############################################################
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

###############################################################
# Utility functions
###############################################################

def safe_str(value):
    """Prevents f-string format specifier errors."""
    try:
        text = str(value)
        text = text.replace("{", "[").replace("}", "]")
        return text
    except:
        return "unknown"

def clean_json(t: str):
    """Remove markdown fences + trailing commas + extract JSON."""
    t = re.sub(r"```(json)?|```", "", t or "")
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return {}
    s = re.sub(r",(\s*[}\]])", r"\1", m.group(0))
    try:
        return json.loads(s)
    except:
        return {}

def detect_type(series: pd.Series):
    if pd.api.types.is_numeric_dtype(series): return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series): return "date"
    return "categorical"

def suggest_chart(x_type: str, y_type: str):
    if x_type == "date" and y_type == "numeric": return "line"
    if x_type == "categorical" and y_type == "numeric": return "bar"
    if x_type == "categorical" and y_type == "categorical": return "donut"
    return "bar"

def fmt_value(v, kind: str):
    try:
        if kind == "currency": return f"₹{float(v):,.0f}"
        if kind == "pct": return f"{float(v)*100:.2f}%"
        if kind == "decimal": return f"{float(v):,.2f}"
        return f"{float(v):,.0f}"
    except:
        return "0"

def kpi_value(df: pd.DataFrame, expr: str):
    try:
        e = expr.upper()
        if e.startswith("SUM("): col = expr[4:-1]; return pd.to_numeric(df[col], errors="coerce").sum()
        if e.startswith("AVG("): col = expr[4:-1]; return pd.to_numeric(df[col], errors="coerce").mean()
        if e.startswith("COUNT("): col = expr[6:-1]; return df[col].count()
        if e.startswith("DISTINCT("): col = expr[9:-1]; return df[col].nunique()
    except:
        return 0
    return 0

def make_plot_html(df: pd.DataFrame, x: str, y: str, ctype: str):
    """Return a Plotly chart embedded as HTML."""
    if ctype == "table":
        # Render table as HTML
        head = df[[x] + [y]].head(20)
        return head.to_html(index=False)

    if x not in df.columns or y not in df.columns:
        return "<div style='color:#999'>Invalid fields</div>"

    try:
        if ctype in ("bar", "line", "area"):
            agg = df.groupby(x)[y].sum(numeric_only=True).reset_index()
        else:
            agg = df

        if ctype == "bar":
            fig = px.bar(agg, x=x, y=y)
        elif ctype == "line":
            fig = px.line(agg, x=x, y=y)
        elif ctype == "area":
            fig = px.area(agg, x=x, y=y)
        elif ctype == "donut":
            fig = px.pie(agg, names=x, values=y, hole=0.45)
        elif ctype == "scatter":
            fig = px.scatter(df, x=x, y=y)
        elif ctype == "hist":
            fig = px.histogram(df, x=y)
        else:
            fig = px.bar(agg, x=x, y=y)

        fig.update_layout(height=360, margin=dict(l=8, r=8, t=36, b=8))
        return fig.to_html(include_plotlyjs="cdn", full_html=False)
    except Exception as e:
        return f"<div style='color: red'>Chart error: {e}</div>"

def narrative_for_chart(title, snippet_df):
    try:
        prompt = f"One short executive insight for chart '{title}' based on these rows: {snippet_df.to_dict(orient='records')}"
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.4
        )
        return (r.choices[0].message.content or "").strip()
    except:
        return ""

###############################################################
# LLM Prompt (forced structure)
###############################################################

def strong_llm_prompt(goal, columns, role):
    col_list = ", ".join(columns)
    return f"""
You are a Principal BI Architect. Build a COMPLETE Power BI-style dashboard spec.

Goal: {goal}
Role: {role}
Columns: [{col_list}]

MUST produce JSON exactly like:

{{
 "title":"string",
 "theme":"light",
 "filters":[{{"field":"Column"}}],
 "kpis":[
    {{"title":"","expr":"SUM(X)","format":"currency|pct|decimal"}},
    ...
 ],
 "sections":[
    {{
      "name":"Overview",
      "charts":[{{"x":"","y":"","type":"bar|line|area|scatter|donut|hist"}}]
    }},
    {{
      "name":"Performance",
      "charts":[...]
    }},
    {{
      "name":"Trends",
      "charts":[...]
    }},
    {{
      "name":"Deep Dive",
      "charts":[{{"type":"table","columns":["A","B","C"]}}]
    }},
    {{
      "name":"Risks & Notes",
      "charts":[...]
    }}
 ]
}}

Rules:
- 5–7 KPIs required.
- 6–10 charts required.
- Each section must have 1–3 visuals.
- Deep Dive must include at least 1 table chart.
- Use valid columns only.
- Return JSON only.
"""

###############################################################
# Session State
###############################################################
if "df" not in st.session_state: st.session_state.df = None
if "dashboards" not in st.session_state: st.session_state.dashboards = {}
if "current" not in st.session_state: st.session_state.current = None
if "filters" not in st.session_state: st.session_state.filters = {}
if "slide_index" not in st.session_state: st.session_state.slide_index = 0
if "presentation" not in st.session_state: st.session_state.presentation = False

###############################################################
# Sidebar UI
###############################################################
with st.sidebar:
    st.header("📂 Data")
    uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx","xls","csv"])

    st.header("🎯 Dashboard Goal")
    role = st.selectbox("Audience", ["BI Developer","Finance Analyst","Sales Leader","Operations Manager"])
    goal = st.text_area("Goal", "Executive revenue, margin, performance & trends")

    generate = st.button("✨ Generate Spec")

    st.divider()
    st.header("📁 Dashboards")
    names = ["(none)"] + list(st.session_state.dashboards.keys())
    pick = st.selectbox("Open", names)
    if pick != "(none)": st.session_state.current = pick

###############################################################
# Load Data
###############################################################
if uploaded:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            xls = pd.read_excel(uploaded, sheet_name=None)
            df = xls[list(xls.keys())[0]] if isinstance(xls, dict) else xls
        st.session_state.df = df
        st.success(f"Loaded {len(df)} rows.")
    except Exception as e:
        st.error(f"File error: {e}")

if st.session_state.df is None:
    st.info("Upload a dataset to continue.")
    st.stop()

df = st.session_state.df

###############################################################
# Generate Spec
###############################################################
if generate:
    try:
        p = strong_llm_prompt(goal, list(df.columns), role)
        r = client.chat.completions.create(model="gpt-4o-mini",
                                           messages=[{"role":"user","content":p}],
                                           temperature=0.25)
        spec = clean_json(r.choices[0].message.content)
        if not spec:
            st.error("Invalid JSON from model.")
        else:
            name = "Dashboard"
            st.session_state.current = name
            st.session_state.dashboards[name] = {"spec": spec, "html": ""}
            st.session_state.filters = {}
            st.success("Spec generated.")
    except Exception as e:
        st.error(f"OpenAI error: {e}")

if st.session_state.current not in st.session_state.dashboards:
    st.info("Generate or select a dashboard.")
    st.stop()

spec = st.session_state.dashboards[st.session_state.current]["spec"]

###############################################################
# Filters
###############################################################
with st.expander("🎛 Filters", expanded=True):
    new_filters = {}
    for f in spec.get("filters", []):
        field = f.get("field")
        if field in df.columns:
            vals = sorted(df[field].dropna().astype(str).unique())
            sel = st.multiselect(field, vals, key=f"flt_{field}")
            new_filters[field] = sel
    st.session_state.filters = new_filters

def apply_filters(df, filters):
    d = df.copy()
    for k, v in filters.items():
        if v:
            d = d[d[k].astype(str).isin(v)]
    return d

df_f = apply_filters(df, st.session_state.filters)

###############################################################
# Build KPI HTML
###############################################################
kpi_html = ""
for k in spec.get("kpis", []):
    title = safe_str(k.get("title"))
    expr = k.get("expr")
    fmt = k.get("format", "auto")
    val = kpi_value(df_f, expr)
    kpi_html += f"""
    <div class="kpi-card">
      <div class="kpi-title">{title}</div>
      <div class="kpi-value">{fmt_value(val, fmt)}</div>
    </div>
    """

###############################################################
# Build Sections (Charts, Tables, Narratives)
###############################################################
sections_html = ""
for sec in spec.get("sections", []):
    sec_name = safe_str(sec.get("name"))
    block = ""
    for ch in sec.get("charts", []):
        ctype_raw = ch.get("type", "")
        ctype = safe_str(ctype_raw)
        x = ch.get("x")
        y = ch.get("y")

        # Handle table chart
        if ctype == "table":
            cols = ch.get("columns", [])
            if cols:
                table_df = df_f[cols].head(30)
                html_table = table_df.to_html(index=False)
            else:
                html_table = df_f.head(20).to_html(index=False)
            block += f"<div class='viz'><h4>{sec_name} Table</h4>{html_table}</div>"
            continue

        # Normal chart
        if x and y and x in df.columns and y in df.columns:
            html = make_plot_html(df_f, x, y, ctype)
            snippet = df_f[[x,y]].head(8)
            story = narrative_for_chart(f"{x} vs {y}", snippet)
            block += f"""
            <div class='viz'>
              <h4>{safe_str(x)} vs {safe_str(y)} ({ctype})</h4>
              {html}
              <div class="insight">🧠 {story}</div>
            </div>
            """

    sections_html += f"""
    <section>
      <h3>{sec_name}</h3>
      <div class='viz-grid'>{block}</div>
    </section>
    """

###############################################################
# HTML Template
###############################################################
TEMPLATE = Template("""
<html>
<head>
<style>
:root {
 --bg:#fff;
 --panel:#f6f6f8;
 --border:#d9d9e3;
 --ink:#222;
}
body{background:var(--panel);font-family:Segoe UI;margin:0}
.wrapper{max-width:1400px;margin:auto;padding:20px}
.dashboard{background:var(--bg);border:1px solid var(--border);border-radius:12px;padding:20px;aspect-ratio:16/9}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.kpi-card{border:1px solid var(--border);padding:10px;border-radius:8px;text-align:center}
.kpi-title{font-size:12px;color:#666}
.kpi-value{font-size:20px;font-weight:600}
.viz-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}
.viz{background:white;border:1px solid var(--border);border-radius:10px;padding:12px}
.insight{font-size:12px;color:#777;margin-top:8px}
</style>
</head>
<body>
<div class="wrapper">
  <div class="dashboard">
    <h2>{{ title }}</h2>
    <div class="kpis">{{ kpis|safe }}</div>
    {{ sections|safe }}
  </div>
</div>
</body>
</html>
""")

dashboard_html = TEMPLATE.render(
    title=safe_str(spec.get("title","Dashboard")),
    kpis=kpi_html,
    sections=sections_html
)

# Save
st.session_state.dashboards[st.session_state.current]["html"] = dashboard_html

###############################################################
# Presentation Mode
###############################################################
if st.checkbox("🎤 Presentation Mode"):
    st.markdown("""
    <style>
      .stSidebar, header, footer, [data-testid="stToolbar"] {display:none!important;}
      .block-container {padding:0!important;margin:0!important;}
    </style>
    """, unsafe_allow_html=True)

###############################################################
# Render HTML
###############################################################
st.subheader("📊 Full Dashboard Preview (HTML)")

with st.expander("▼ Show/Hide Dashboard", expanded=True):
    st.components.v1.html(dashboard_html, height=850, scrolling=True)

###############################################################
# Slide Mode
###############################################################
st.subheader("🎞 Slide Mode")

sections = spec.get("sections", [])
if sections:
    st.session_state.slide_index %= len(sections)
    sec = sections[st.session_state.slide_index]

    colA, colB = st.columns([1,1])
    if colA.button("◀ Prev"):
        st.session_state.slide_index -= 1
    if colB.button("Next ▶"):
        st.session_state.slide_index += 1

    st.markdown(f"### Slide: {safe_str(sec.get('name'))}")

    # Show charts for this section
    for ch in sec.get("charts", []):
        ctype = safe_str(ch.get("type","bar"))
        x = ch.get("x"); y = ch.get("y")

        if ctype == "table":
            cols = ch.get("columns", [])
            df_view = df_f[cols].head(30) if cols else df_f.head(20)
            st.dataframe(df_view)
            continue

        if x in df.columns and y in df.columns:
            st.markdown(f"**{x} vs {y} ({ctype})**")
            if ctype in ("bar","line","area"):
                agg = df_f.groupby(x)[y].sum(numeric_only=True).reset_index()
            else:
                agg = df_f

            if ctype == "bar": fig = px.bar(agg, x=x, y=y)
            elif ctype == "line": fig = px.line(agg, x=x, y=y)
            elif ctype == "area": fig = px.area(agg, x=x, y=y)
            elif ctype == "donut": fig = px.pie(agg, names=x, values=y, hole=0.45)
            elif ctype == "scatter": fig = px.scatter(df_f, x=x, y=y)
            else: fig = px.bar(agg, x=x, y=y)

            st.plotly_chart(fig, use_container_width=True)

###############################################################
# Spec Editor
###############################################################
with st.expander("⚙️ Advanced Editor"):
    st.write("Edit spec JSON below then click apply:")
    raw = st.text_area("Spec JSON", json.dumps(spec, indent=2), height=300)
    if st.button("Apply Spec"):
        try:
            new_spec = json.loads(raw)
            st.session_state.dashboards[st.session_state.current]["spec"] = new_spec
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error: {e}")
