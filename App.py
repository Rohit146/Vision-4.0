import streamlit as st
import pandas as pd
import json, re, os
from openai import OpenAI
import plotly.express as px
from jinja2 import Template

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Auto-BI Dashboard Generator", layout="wide")
st.title("🧠 Auto-BI — HTML Dashboard Generator")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in secrets.")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------- HELPERS ----------------
def clean_json(t):
    t = re.sub(r"```(json)?|```", "", t)
    m = re.search(r"\{[\s\S]*\}", t)
    if not m: return {}
    s = re.sub(r",(\s*[}\]])", r"\1", m.group(0))
    try: return json.loads(s)
    except: return {}

def detect_type(series):
    if pd.api.types.is_numeric_dtype(series): return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series): return "date"
    return "categorical"

def suggest_chart(x_type, y_type):
    if x_type=="date" and y_type=="numeric": return "line"
    if x_type=="categorical" and y_type=="numeric": return "bar"
    if x_type=="categorical" and y_type=="categorical": return "pie"
    return "bar"

def build_chart_html(df, x, y, chart_type):
    """Returns a Plotly chart as an HTML <div>"""
    if chart_type == "bar":
        fig = px.bar(df.groupby(x)[y].sum().reset_index(), x=x, y=y)
    elif chart_type == "line":
        fig = px.line(df.groupby(x)[y].sum().reset_index(), x=x, y=y)
    elif chart_type == "pie":
        fig = px.pie(df, names=x, values=y)
    else:
        fig = px.bar(df, x=x, y=y)
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=20))
    return fig.to_html(include_plotlyjs="cdn", full_html=False)

# ---------------- SIDEBAR ----------------
st.sidebar.header("📂 Data & Prompt")
file = st.sidebar.file_uploader("Upload Excel/CSV", type=["xlsx","xls","csv"])
role = st.sidebar.selectbox("Role", ["BI Developer", "Finance Analyst", "Sales Leader", "Operations Manager"])
goal = st.sidebar.text_area("Business Goal", "Show sales by region with YoY growth.")

generate = st.sidebar.button("✨ Generate Dashboard")

# ---------------- PROCESS FILE ----------------
if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
else:
    st.info("Please upload a dataset.")
    st.stop()

# ---------------- GENERATE SPEC ----------------
if generate:
    cols = ", ".join(df.columns)
    prompt = f"""
Act as a Power BI dashboard designer.
Using columns [{cols}] create a dashboard JSON spec for goal "{goal}".
Structure:
{{
 "title": "Dashboard Title",
 "filters": [{{"field":""}}],
 "kpis": [{{"title":"","expr":"SUM(Sales)","format":"currency"}}],
 "charts": [{{"x":"","y":"","type":"bar"}}],
 "theme":"light"
}}
Return JSON only.
"""
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3
    )
    spec = clean_json(r.choices[0].message.content)
    st.session_state.spec = spec
else:
    spec = st.session_state.get("spec", None)

if not spec:
    st.stop()

# ---------------- BUILD HTML DASHBOARD ----------------
charts_html = ""
for c in spec.get("charts", []):
    x, y = c.get("x"), c.get("y")
    if x not in df.columns or y not in df.columns:
        continue
    x_type, y_type = detect_type(df[x]), detect_type(df[y])
    chart_type = suggest_chart(x_type, y_type)
    charts_html += f"""
    <div class="chart-container">
        <h4>{x} vs {y}</h4>
        {build_chart_html(df, x, y, chart_type)}
    </div>
    """

kpi_html = ""
for k in spec.get("kpis", []):
    expr = k.get("expr", "")
    if expr.startswith("SUM("):
        col = expr[4:-1]
        val = df[col].sum() if col in df.columns else 0
    elif expr.startswith("AVG("):
        col = expr[4:-1]
        val = df[col].mean() if col in df.columns else 0
    else:
        val = len(df)
    val_fmt = f"₹{val:,.0f}" if k.get("format")=="currency" else f"{val:,.0f}"
    kpi_html += f"""
    <div class="kpi-card">
        <div class="kpi-title">{k.get('title','')}</div>
        <div class="kpi-value">{val_fmt}</div>
    </div>
    """

# HTML template (16:9 Power BI layout)
template = Template("""
<html>
<head>
<style>
body { font-family: 'Segoe UI', sans-serif; background:#f9fafb; margin:0; padding:0; }
.dashboard { aspect-ratio:16/9; width:100%; max-width:1400px; margin:auto; background:white; border-radius:12px; padding:20px; box-shadow:0 2px 6px rgba(0,0,0,0.1); display:flex; flex-direction:column; gap:16px; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; }
.kpi-card { background:#f1f5f9; border-radius:8px; text-align:center; padding:10px; }
.kpi-title { font-size:13px; opacity:0.7; }
.kpi-value { font-size:20px; font-weight:600; margin-top:4px; }
.charts { display:grid; grid-template-columns:repeat(auto-fit,minmax(400px,1fr)); gap:20px; flex:1; overflow:auto; }
.chart-container h4 { font-size:14px; margin-bottom:6px; }
</style>
</head>
<body>
<div class="dashboard">
    <h2>{{ title }}</h2>
    <div class="kpis">{{ kpis|safe }}</div>
    <div class="charts">{{ charts|safe }}</div>
</div>
</body>
</html>
""")

dashboard_html = template.render(title=spec.get("title","Dashboard"), kpis=kpi_html, charts=charts_html)

# ---------------- RENDER IN STREAMLIT ----------------
st.markdown("### 📊 Generated Dashboard Preview")

with st.expander("Click to expand dashboard view", expanded=True):
    st.components.v1.html(dashboard_html, height=700, scrolling=True)
