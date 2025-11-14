import streamlit as st
import pandas as pd
import plotly.express as px
from jinja2 import Template
from openai import OpenAI
import os, re, json

# =========================
# Streamlit config
# =========================
st.set_page_config(page_title="Auto-BI HTML Studio", layout="wide")
st.title("🧠 Auto-BI — Power BI-style HTML Dashboard Studio")

# =========================
# API key
# =========================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# Utilities
# =========================
def clean_json(t: str):
    """Strip code fences and trailing commas, return dict or {}."""
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
    # sensible, opinionated defaults
    if x_type == "date" and y_type == "numeric": return "line"
    if x_type == "categorical" and y_type == "numeric": return "bar"
    if x_type == "categorical" and y_type == "categorical": return "pie"
    if x_type == "numeric" and y_type == "numeric": return "scatter"
    return "bar"

def fmt_value(v, kind: str):
    try:
        if kind == "currency": return f"₹{float(v):,.0f}"
        if kind == "pct": return f"{float(v)*100:.2f}%"
        if kind == "decimal": return f"{float(v):,.2f}"
        return f"{float(v):,.0f}"
    except:
        return str(v)

def kpi_value(df: pd.DataFrame, expr: str) -> float:
    # Supports SUM(), AVG(), COUNT(), DISTINCT()
    try:
        if expr.startswith("SUM("):   col = expr[4:-1]; return pd.to_numeric(df[col], errors="coerce").sum()
        if expr.startswith("AVG("):   col = expr[4:-1]; return pd.to_numeric(df[col], errors="coerce").mean()
        if expr.startswith("COUNT("): col = expr[6:-1]; return df[expr[6:-1]].count() if expr[6:-1] in df.columns else len(df)
        if expr.startswith("DISTINCT("): col = expr[9:-1]; return df[col].nunique()
    except:
        return 0
    return 0

def make_plot_html(df: pd.DataFrame, x: str, y: str, ctype: str) -> str:
    """Return Plotly chart HTML (div only, plotly.js via CDN)."""
    if x not in df.columns or y not in df.columns:
        return "<div style='color:#999'>Invalid chart fields</div>"
    # aggregate numeric y by x where applicable
    if ctype in ("bar", "line"):
        data = df.groupby(x)[y].sum(numeric_only=True).reset_index()
    else:
        data = df

    if ctype == "bar":
        fig = px.bar(data, x=x, y=y)
    elif ctype == "line":
        fig = px.line(data, x=x, y=y)
    elif ctype == "pie":
        fig = px.pie(data, names=x, values=y)
    elif ctype == "scatter":
        fig = px.scatter(data, x=x, y=y)
    elif ctype == "hist":
        fig = px.histogram(data, x=y)
    else:
        fig = px.bar(data, x=x, y=y)

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_white",
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=False)

def narrative_for_chart(title: str, data_snippet: dict) -> str:
    try:
        prompt = f"Write one crisp executive insight for chart '{title}' given summarized data: {data_snippet}. Focus on the key takeaway in one sentence."
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.4
        )
        return (r.choices[0].message.content or "").strip()
    except:
        return ""

def strong_llm_prompt(goal: str, columns: list[str], role: str) -> str:
    """
    A detailed instruction to guide the LLM to produce high-quality, deployable specs.
    """
    col_list = ", ".join(columns)
    return f"""
You are a senior BI designer building an enterprise-grade Power BI-style dashboard.

User goal: {goal}
Audience Role: {role}

Available columns: [{col_list}]

Output ONLY valid JSON with this exact structure:
{{
  "title": "string - executive title (<= 60 chars)",
  "theme": "light|dark",
  "filters": [{{"field": "string (slicer field, categorical or date)"}}],
  "kpis": [
    {{"title":"string","expr":"SUM(Revenue)|AVG(Margin)|COUNT(OrderID)|DISTINCT(CustomerID)","format":"auto|currency|pct|decimal"}}
  ],
  "sections": [
    {{
      "name": "Overview|Performance|Trends|Risks|Notes",
      "charts": [
        {{
          "x": "string (dimension)",
          "y": "string (measure)",
          "type": "bar|line|pie|scatter|hist"
        }}
      ]
    }}
  ]
}}

Rules:
- Prefer 3–5 KPIs max. Choose meaningful ones (Revenue, Profit, Margin %, Orders, Avg Deal Size).
- Use date fields on X for Trends (type=line), categories for Performance (bar), composition for pie (<=1 pie).
- Avoid trivial or duplicate visuals.
- Ensure each KPI expr references a real column where required (e.g., SUM(Sales)).
- Filters should be business-relevant (Region, Segment, Category, Date).
- Keep chart count to 4–6 max spanning 2–3 sections.
- Theme: start with "light".
"""

# =========================
# Session state
# =========================
if "df" not in st.session_state: st.session_state.df = None
if "dashboards" not in st.session_state: st.session_state.dashboards = {}  # name -> {"spec":..., "html":...}
if "current" not in st.session_state: st.session_state.current = None
if "filters" not in st.session_state: st.session_state.filters = {}  # live slicers for current dashboard

# =========================
# Sidebar — data, prompting, management
# =========================
with st.sidebar:
    st.header("📂 Data & Generation")
    file = st.file_uploader("Upload Excel/CSV", type=["xlsx","xls","csv"])

    role = st.selectbox("Audience Role", ["BI Developer", "Finance Analyst", "Sales Leader", "Operations Manager"])
    goal = st.text_area("Dashboard Goal",
        "Executive sales overview with revenue, margin, and regional performance, plus trends and risks.")

    generate = st.button("✨ Generate New Dashboard")

    st.divider()
    st.header("🗂️ Dashboards")
    existing = ["(none)"] + list(st.session_state.dashboards.keys())
    current_name = st.selectbox("Open dashboard", existing, index=existing.index(st.session_state.current) if st.session_state.current in existing else 0)
    if current_name != "(none)":
        st.session_state.current = current_name

    new_name = st.text_input("Save as name")
    if st.button("💾 Save current") and new_name.strip():
        if st.session_state.current and st.session_state.current in st.session_state.dashboards:
            # rename or duplicate
            st.session_state.dashboards[new_name] = st.session_state.dashboards[st.session_state.current]
            st.session_state.current = new_name
        else:
            # will be saved after generation/render
            st.session_state.current = new_name

    if st.session_state.current and st.session_state.current in st.session_state.dashboards:
        html_to_dl = st.session_state.dashboards[st.session_state.current].get("html","")
        if html_to_dl:
            st.download_button("⬇️ Download HTML", data=html_to_dl, file_name=f"{st.session_state.current}.html", mime="text/html")

# =========================
# Load dataframe
# =========================
if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        # If multiple sheets, default to the first
        xls = pd.read_excel(file, sheet_name=None)
        # if single sheet Excel, returns DataFrame not dict — unify:
        if isinstance(xls, dict):
            first_sheet = list(xls.keys())[0]
            df = xls[first_sheet]
        else:
            df = xls
    st.session_state.df = df

if st.session_state.df is None:
    st.info("Upload a dataset to get started.")
    st.stop()

df = st.session_state.df

# =========================
# Generate spec via LLM
# =========================
if generate:
    cols = list(df.columns)
    prompt = strong_llm_prompt(goal, cols, role)
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content": prompt}],
        temperature=0.25
    )
    spec = clean_json(r.choices[0].message.content)
    if not spec or "sections" not in spec:
        st.error("The model did not return a valid spec. Try again with a clearer goal.")
        st.stop()
    # store spec under a temporary name (or existing slot)
    tmp_name = st.session_state.current or "Untitled"
    if tmp_name not in st.session_state.dashboards:
        st.session_state.dashboards[tmp_name] = {"spec": spec, "html": ""}
        st.session_state.current = tmp_name
    else:
        st.session_state.dashboards[tmp_name]["spec"] = spec
    # reset filters for new spec
    st.session_state.filters = {}

# =========================
# Pick current dashboard
# =========================
if not st.session_state.current or st.session_state.current not in st.session_state.dashboards:
    st.info("Use ‘Generate New Dashboard’ or select an existing one from the sidebar.")
    st.stop()

cur = st.session_state.dashboards[st.session_state.current]
spec = cur.get("spec")
if not spec:
    st.info("No spec found for this dashboard.")
    st.stop()

# =========================
# Slicers / Filters UI (live)
# =========================
with st.expander("🎛 Filters (live slicers)", expanded=True):
    # build slicers based on spec.filters (categorical/date suggested)
    slicers = spec.get("filters", []) or []
    cols = st.columns(max(1, min(4, len(slicers))))
    new_filters = dict(st.session_state.filters)  # copy
    for i, fdef in enumerate(slicers):
        field = fdef.get("field")
        if not field or field not in df.columns:
            continue
        # unique vals preview (limit)
        vals = sorted(df[field].dropna().astype(str).unique().tolist())
        default = new_filters.get(field, [])
        chosen = cols[i % len(cols)].multiselect(field, vals[:500], default=default, key=f"flt_{field}")
        new_filters[field] = chosen
    st.session_state.filters = new_filters

# apply filters
def apply_filters_df(df: pd.DataFrame, filters: dict):
    out = df
    for c, vals in (filters or {}).items():
        if c in out.columns and vals:
            out = out[out[c].astype(str).isin(vals)]
    return out

df_f = apply_filters_df(df, st.session_state.filters)

# =========================
# Build KPI HTML
# =========================
kpi_html = ""
for k in spec.get("kpis", []):
    title = k.get("title", "")
    expr  = k.get("expr", "COUNT(*)")
    fmt   = k.get("format", "auto")
    val   = kpi_value(df_f, expr)
    if fmt == "auto":
        # heuristic: if "margin" or "rate" in title → pct, if "revenue/sales/amount/value" → currency
        t = title.lower()
        if "margin" in t or "rate" in t or "growth" in t or "%" in t: fmt = "pct"
        elif any(w in t for w in ["revenue","sale","amount","value","gmv"]): fmt = "currency"
        else: fmt = "decimal" if abs(val) < 1000 else "auto"
    kpi_html += f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{fmt_value(val, fmt)}</div>
    </div>
    """

# =========================
# Build Charts HTML per section (with narratives)
# =========================
sections_html = ""
palette_css = """
--bg: #ffffff; --panel:#f5f7fb; --ink:#111827; --muted:#6b7280; --border:#e5e7eb;
"""
if (spec.get("theme") or "light") == "dark":
    palette_css = "--bg:#0b0f1a; --panel:#111827; --ink:#f3f4f6; --muted:#9ca3af; --border:#1f2937;"

for sec in spec.get("sections", []):
    sec_block = ""
    sec_name = sec.get("name", "Section")
    for ch in sec.get("charts", []):
        x = ch.get("x"); y = ch.get("y")
        if not x or not y or x not in df.columns or y not in df.columns:
            continue
        x_type = detect_type(df[x]); y_type = detect_type(df[y])
        ctype  = ch.get("type") or suggest_chart(x_type, y_type)

        # chart html
        chart_html = make_plot_html(df_f, x, y, ctype)

        # small data snippet for narrative
        if ctype in ("bar","line"):
            snip_df = df_f.groupby(x)[y].sum(numeric_only=True).reset_index().head(10)
        elif ctype == "pie":
            snip_df = df_f[[x, y]].head(10)
        elif ctype == "hist":
            snip_df = df_f[[y]].head(10)
        else:
            snip_df = df_f[[x, y]].head(10)
        insight = narrative_for_chart(f"{x} vs {y}", snip_df.to_dict(orient="list"))

        sec_block += f"""
        <div class="viz">
            <div class="viz-title">{x} vs {y} <span class="chip">{ctype}</span></div>
            {chart_html}
            <div class="viz-insight">🧠 {insight}</div>
        </div>
        """
    if sec_block:
        sections_html += f"""
        <section>
            <h3>{sec_name}</h3>
            <div class="viz-grid">
                {sec_block}
            </div>
        </section>
        """

# =========================
# HTML Template (Power BI–style, 16:9, responsive)
# =========================
TEMPLATE = Template("""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{{ title }}</title>
<style>
:root { {{ palette }} }
*{box-sizing:border-box}
body{font-family:Segoe UI,system-ui,-apple-system,Arial,sans-serif;background:var(--panel);margin:0;color:var(--ink)}
.wrapper{max-width:1400px;margin:16px auto;padding:0 12px}
.dashboard{width:100%;aspect-ratio:16/9;background:var(--bg);border:1px solid var(--border);border-radius:14px;box-shadow:0 2px 10px rgba(0,0,0,0.05);display:flex;flex-direction:column;gap:14px;padding:16px}
header h2{margin:0;font-size:24px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}
.kpi-card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:10px;text-align:center}
.kpi-title{font-size:12px;color:var(--muted)}
.kpi-value{font-size:22px;font-weight:700;margin-top:4px}
section{padding-top:6px}
section h3{font-size:16px;margin:8px 0 8px}
.viz-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px}
.viz{background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:8px}
.viz-title{font-size:13px;color:var(--muted);margin:2px 0 6px}
.viz-insight{font-size:12px;color:var(--muted);margin-top:4px}
.chip{display:inline-block;background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:0 6px;margin-left:6px;font-size:11px}
footer{font-size:12px;color:var(--muted);text-align:right;margin-top:auto}
</style>
</head>
<body>
  <div class="wrapper">
    <div class="dashboard">
      <header><h2>{{ title }}</h2></header>
      <div class="kpis">{{ kpis|safe }}</div>
      {{ sections|safe }}
      <footer>Powered by Auto-BI • 16:9 canvas • Theme: {{ theme }}</footer>
    </div>
  </div>
</body>
</html>
""")

dashboard_html = TEMPLATE.render(
    title = spec.get("title", "Executive Dashboard"),
    theme = spec.get("theme","light"),
    palette = palette_css,
    kpis = kpi_html,
    sections = sections_html
)

# persist HTML with this spec
st.session_state.dashboards[st.session_state.current]["html"] = dashboard_html

# =========================
# Presentation mode toggle
# =========================
colA, colB = st.columns([1,1])
with colA:
    st.markdown(f"### 📊 Preview — **{st.session_state.current}**")
with colB:
    present = st.toggle("🎤 Presentation Mode", value=False, help="Hide Streamlit chrome for clean viewing")

if present:
    st.markdown("""
    <style>
      .stSidebar, header, footer, [data-testid="stToolbar"] { display:none !important; }
      .block-container { padding-top: 10px !important; }
    </style>""", unsafe_allow_html=True)

# =========================
# Render HTML in dropdown (fit to page)
# =========================
with st.expander("▼ Click to show/hide dashboard", expanded=True):
    # height tuned to show full 16:9 canvas on most laptops; grid is responsive inside
    st.components.v1.html(dashboard_html, height=820, scrolling=True)
