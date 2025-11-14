# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from jinja2 import Template
from openai import OpenAI
import os, re, json, math
from typing import List

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Auto-BI HTML Studio (Advanced)", layout="wide")
st.title("🧠 Auto-BI — Power BI-style HTML Dashboard Studio (Advanced)")

# -----------------------
# OpenAI client
# -----------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------
# Utilities
# -----------------------
def clean_json(t: str):
    t = re.sub(r"```(json)?|```", "", t or "")
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return {}
    s = re.sub(r",(\s*[}\]])", r"\1", m.group(0))
    try:
        return json.loads(s)
    except Exception:
        return {}

def detect_type(series: pd.Series):
    if pd.api.types.is_numeric_dtype(series): return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series): return "date"
    return "categorical"

def suggest_chart(x_type: str, y_type: str):
    if x_type == "date" and y_type == "numeric": return "line"
    if x_type == "categorical" and y_type == "numeric": return "bar"
    if x_type == "categorical" and y_type == "categorical": return "donut"
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
    try:
        e = expr.strip().upper()
        if e.startswith("SUM("): col = expr[4:-1]; return pd.to_numeric(df[col], errors="coerce").sum()
        if e.startswith("AVG("): col = expr[4:-1]; return pd.to_numeric(df[col], errors="coerce").mean()
        if e.startswith("COUNT("): col = expr[6:-1]; return df[col].count() if col in df.columns else len(df)
        if e.startswith("DISTINCT("): col = expr[9:-1]; return df[col].nunique()
    except:
        return 0
    return 0

def make_plot_html(df: pd.DataFrame, x: str, y: str, ctype: str):
    """Return Plotly chart HTML snippet (div)."""
    if x not in df.columns or y not in df.columns:
        return "<div style='color:#999'>Invalid chart fields</div>"
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

        fig.update_layout(height=360, margin=dict(l=8, r=8, t=36, b=8), template="plotly_white")
        return fig.to_html(include_plotlyjs="cdn", full_html=False)
    except Exception as e:
        return f"<div style='color:#c00'>Chart error: {str(e)}</div>"

def narrative_for_chart(title: str, df_snippet: pd.DataFrame):
    try:
        # small prompt; keep short
        prompt = f"One concise executive insight (one sentence) for chart '{title}' with these data rows: {df_snippet.to_dict(orient='records')}"
        r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.4)
        return (r.choices[0].message.content or "").strip()
    except:
        return ""

# -----------------------
# Strong LLM prompt (MANDATORY JSON schema + story rules)
# -----------------------
def strong_llm_prompt(goal: str, columns: List[str], role: str) -> str:
    col_list = ", ".join(columns)
    return f"""
You are a Principal BI Architect. Build a complete enterprise-grade Power BI-style dashboard spec that a BI developer can implement.

USER GOAL:
{goal}

AUDIENCE ROLE: {role}

AVAILABLE COLUMNS:
[{col_list}]

RETURN ONLY VALID JSON EXACTLY in this structure:

{{
 "title":"string (<=60 chars)",
 "theme":"light",
 "filters":[{{"field":"ColumnName"}}],
 "kpis":[
    {{"title":"string","expr":"SUM(Revenue)|SUM(Profit)|AVG(Margin)|COUNT(OrderID)|DISTINCT(CustomerID)","format":"currency|pct|decimal"}},
    ...
 ],
 "sections":[
    {{
      "name":"Overview",
      "charts":[{{"x":"dim","y":"measure","type":"bar|line|donut|scatter|area|hist"}}]
    }},
    {{
      "name":"Performance",
      "charts":[ ... ]
    }},
    {{
      "name":"Trends",
      "charts":[ ... ]
    }},
    {{
      "name":"Deep Dive",
      "charts":[ ... ]  // may include "table" entries in format: {"type":"table","cols":["c1","c2"]}
    }},
    {{
      "name":"Risks & Notes",
      "charts":[ ... ]
    }}
 ]
}}

RULES (ENFORCING QUALITY):
- MUST include 5–7 KPIs (business-relevant and non-duplicative).
- MUST include 6–10 charts total, distributed across sections (each section: 1–3 visuals).
- Use date columns for Trends (line/area). Use categorical columns for Performance (bar). Use donut/pie sparingly (<=1).
- Deep Dive must include at least 1 table with columns specified.
- KPIs should include revenue, profit, margin%, orders or equivalents if columns exist. If no revenue-like column present, pick top numeric measures.
- Avoid trivial or duplicate charts. Each chart must map to available columns.
- Include recommended filters (Region, Category, Date) where applicable.
- Output must be consistent and implementable.

Return valid JSON only — no explanation.
"""

# -----------------------
# Session state
# -----------------------
if "df" not in st.session_state: st.session_state.df = None
if "dashboards" not in st.session_state: st.session_state.dashboards = {}  # name -> {"spec":..., "html":...}
if "current" not in st.session_state: st.session_state.current = None
if "filters" not in st.session_state: st.session_state.filters = {}
if "slide_index" not in st.session_state: st.session_state.slide_index = 0
if "presentation" not in st.session_state: st.session_state.presentation = False
if "drill" not in st.session_state: st.session_state.drill = None  # {"x","y","value","rows_html"}

# -----------------------
# Sidebar: data + controls
# -----------------------
with st.sidebar:
    st.header("Data & Controls")
    uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx","xls","csv"])
    role = st.selectbox("Audience role", ["BI Developer", "Finance Analyst", "Sales Leader", "Operations Manager"])
    goal = st.text_area("Dashboard goal", "Executive sales overview: revenue, margin, regional performance and trends.")
    generate = st.button("✨ Generate Dashboard (LLM)")

    st.divider()
    st.header("Dashboards")
    existing = ["(none)"] + list(st.session_state.dashboards.keys())
    choice = st.selectbox("Open dashboard", existing, index=(existing.index(st.session_state.current) if st.session_state.current in existing else 0))
    if choice != "(none)":
        st.session_state.current = choice

    newname = st.text_input("Save as (name)")
    if st.button("💾 Save current as name") and newname.strip():
        if st.session_state.current and st.session_state.current in st.session_state.dashboards:
            # duplicate under new name
            st.session_state.dashboards[newname] = st.session_state.dashboards[st.session_state.current].copy()
        else:
            # will save after generation/render
            st.session_state.dashboards[newname] = {"spec": None, "html": ""}
        st.success(f"Saved as {newname}")

    st.divider()
    if st.session_state.current and st.session_state.current in st.session_state.dashboards:
        if st.download_button("⬇️ Download HTML", data=st.session_state.dashboards[st.session_state.current].get("html",""), file_name=f"{st.session_state.current}.html", mime="text/html"):
            pass

    st.divider()
    st.checkbox("Presentation Mode (hide chrome)", value=st.session_state.presentation, key="presentation_toggle", help="Toggle full-screen-like presentation mode")

# -----------------------
# Load file into dataframe
# -----------------------
if uploaded:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df_main = pd.read_csv(uploaded)
        else:
            xls = pd.read_excel(uploaded, sheet_name=None)
            # use first sheet by default
            if isinstance(xls, dict):
                first_sheet = list(xls.keys())[0]
                df_main = xls[first_sheet]
            else:
                df_main = xls
        st.session_state.df = df_main
        st.success(f"Loaded data: {uploaded.name} — {len(st.session_state.df)} rows")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

if st.session_state.df is None:
    st.info("Upload data to begin.")
    st.stop()

df = st.session_state.df

# -----------------------
# Generate spec by LLM
# -----------------------
if generate:
    cols = list(df.columns)
    prompt = strong_llm_prompt(goal, cols, role)
    try:
        r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.25)
        spec = clean_json(r.choices[0].message.content)
    except Exception as e:
        st.error("LLM error: " + str(e))
        spec = {}

    if not spec or "sections" not in spec:
        st.error("Model didn't return a valid spec. Try again with a clearer goal or reorganize columns.")
    else:
        name = st.session_state.current or "Untitled"
        st.session_state.dashboards[name] = {"spec": spec, "html": ""}
        st.session_state.current = name
        st.session_state.filters = {}
        st.session_state.slide_index = 0
        st.success(f"Spec generated and stored as '{name}'")

# -----------------------
# Current dashboard checks
# -----------------------
if not st.session_state.current or st.session_state.current not in st.session_state.dashboards:
    st.info("Generate or select a dashboard from the sidebar.")
    st.stop()

cur = st.session_state.dashboards[st.session_state.current]
spec = cur.get("spec")
if not spec:
    st.info("Dashboard spec is empty. Generate one.")
    st.stop()

# -----------------------
# Filters / Slicers UI
# -----------------------
with st.expander("🎛 Slicers / Filters (live)", expanded=True):
    slicers = spec.get("filters", [])
    if not slicers:
        st.info("No filters suggested by spec — you can add filters on the left or edit spec.")
    else:
        cols_ui = st.columns(max(1, min(4, len(slicers))))
        new_filters = dict(st.session_state.filters)
        for i, f in enumerate(slicers):
            field = f.get("field")
            if not field or field not in df.columns:
                continue
            vals = sorted(df[field].dropna().astype(str).unique().tolist())
            default = new_filters.get(field, [])
            chosen = cols_ui[i % len(cols_ui)].multiselect(field, vals[:1000], default=default, key=f"flt_{field}")
            new_filters[field] = chosen
        st.session_state.filters = new_filters

# apply filters
def apply_filters_df(df_local: pd.DataFrame, filters_local: dict):
    out = df_local
    for c, vals in (filters_local or {}).items():
        if c in out.columns and vals:
            out = out[out[c].astype(str).isin(vals)]
    return out

df_f = apply_filters_df(df, st.session_state.filters)

# -----------------------
# Build KPI HTML
# -----------------------
kpi_html = ""
kpis = spec.get("kpis", [])[:7]  # spec should provide 5-7, but be robust
for k in kpis:
    title = k.get("title", "")
    expr = k.get("expr", "COUNT(*)")
    fmt = k.get("format", "auto")
    val = kpi_value(df_f, expr)
    if fmt == "auto":
        t = title.lower()
        if "margin" in t or "rate" in t or "%" in t: fmt = "pct"
        elif any(w in t for w in ["revenue", "sale", "amount", "value", "gmv", "profit"]): fmt = "currency"
        else: fmt = "decimal" if abs(val) < 1000 else "auto"
    kpi_html += f"""<div class="kpi-card"><div class="kpi-title">{title}</div><div class="kpi-value">{fmt_value(val, fmt)}</div></div>"""

# -----------------------
# Build Sections: charts, narratives, drill buttons
# -----------------------
sections_html = ""
for sec in spec.get("sections", []):
    sec_name = sec.get("name", "Section")
    sec_block = ""
    charts = sec.get("charts", []) or []
    for idx, ch in enumerate(charts):
        x = ch.get("x"); y = ch.get("y"); ctype = ch.get("type")
        if not x or not y:
            continue
        if x not in df.columns or y not in df.columns:
            continue
        x_type = detect_type(df[x]); y_type = detect_type(df[y])
        if not ctype:
            ctype = suggest_chart(x_type, y_type)
        chart_html = make_plot_html(df_f, x, y, ctype)

        # snippet for narrative (small table)
        if ctype in ("bar", "line", "area"):
            snip = df_f.groupby(x)[y].sum(numeric_only=True).reset_index().head(8)
        elif ctype == "donut":
            snip = df_f.groupby(x)[y].sum(numeric_only=True).reset_index().head(8)
        else:
            snip = df_f[[x, y]].head(8)
        insight = narrative_for_chart(f"{x} vs {y}", snip)

        # compute top categories for drill suggestions
        try:
            top_series = df_f.groupby(x)[y].sum(numeric_only=True).sort_values(ascending=False).head(3)
            top_keys = list(top_series.index.astype(str))
        except Exception:
            top_keys = []

        # drill UI will be handled in Streamlit area; in HTML we show a small hint and provide a Streamlit drill button later.
        chip = f"<span class='chip'>{ctype}</span>"
        top_hint = ""
        if top_keys:
            top_hint = "<div class='top-hint'>Top: " + ", ".join(top_keys[:3]) + "</div>"

        sec_block += f"""
        <div class="viz">
          <div class="viz-head"><div class="viz-title">{x} vs {y}</div> {chip}</div>
          {chart_html}
          <div class="viz-insight">🧠 {insight}</div>
          {top_hint}
        </div>
        """

    if sec_block:
        sections_html += f"""
        <section>
          <h3>{sec_name}</h3>
          <div class="viz-grid">{sec_block}</div>
        </section>
        """

# -----------------------
# Enhanced HTML Template (PowerBI-like, responsive, 16:9)
# -----------------------
palette_css = "--bg:#ffffff;--panel:#f7fbff;--ink:#0f172a;--muted:#6b7280;--border:#e6eef9;"
if (spec.get("theme") or "light") == "dark":
    palette_css = "--bg:#0b0f1a;--panel:#0f1724;--ink:#e6eef9;--muted:#94a3b8;--border:#12202b;"

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
.wrapper{max-width:1400px;margin:12px auto;padding:0 12px}
.dashboard{width:100%;aspect-ratio:16/9;background:var(--bg);border:1px solid var(--border);border-radius:12px;display:flex;flex-direction:column;gap:12px;padding:14px;box-shadow:0 6px 18px rgba(2,6,23,0.06)}
.header{display:flex;justify-content:space-between;align-items:center}
.header h2{margin:0;font-size:20px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.kpi-card{background:transparent;border:1px solid var(--border);border-radius:10px;padding:10px;text-align:center}
.kpi-title{font-size:12px;color:var(--muted)}
.kpi-value{font-size:22px;font-weight:700;margin-top:6px}
section{padding-top:8px}
section h3{font-size:16px;margin:8px 0 6px}
.viz-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px;align-items:start}
.viz{background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:8px}
.viz-head{display:flex;justify-content:space-between;align-items:center}
.viz-title{font-size:13px;color:var(--muted)}
.viz-insight{font-size:12px;color:var(--muted);margin-top:8px}
.chip{display:inline-block;background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:0 6px;margin-left:6px;font-size:11px;color:var(--muted)}
.top-hint{font-size:12px;color:var(--muted);margin-top:6px}
footer{font-size:12px;color:var(--muted);text-align:right;margin-top:auto}
@media (max-width:900px){
  .viz-grid{grid-template-columns:1fr}
  .kpi-value{font-size:18px}
}
</style>
</head>
<body>
 <div class="wrapper">
   <div class="dashboard">
     <div class="header"><h2>{{ title }}</h2><div class="meta">Auto-BI • 16:9 • Theme: {{ theme }}</div></div>
     <div class="kpis">{{ kpis|safe }}</div>
     {{ sections|safe }}
     <footer>Generated by Auto-BI • Story: Overview → Performance → Trends → Deep Dive → Risks</footer>
   </div>
 </div>
</body>
</html>
""")

dashboard_html = TEMPLATE.render(title=spec.get("title","Executive Dashboard"), palette=palette_css, kpis=kpi_html, sections=sections_html, theme=spec.get("theme","light"))

# store HTML
st.session_state.dashboards[st.session_state.current]["html"] = dashboard_html
st.session_state.dashboards[st.session_state.current]["spec"] = spec

# -----------------------
# Presentation & Slide mode controls
# -----------------------
col_left, col_right = st.columns([3,1])
with col_left:
    st.markdown(f"### 📊 Preview — **{st.session_state.current}**")
with col_right:
    # Slide controls
    if st.button("◀ Prev Slide"):
        st.session_state.slide_index = max(0, st.session_state.slide_index - 1)
    if st.button("Next Slide ▶"):
        st.session_state.slide_index = min(max(0, len(spec.get("sections",[])) - 1), st.session_state.slide_index + 1)
    st.write(f"Slide {st.session_state.slide_index + 1} / {max(1, len(spec.get('sections',[])))}")
    # Toggle presentation mode
    pres = st.checkbox("Presentation Mode (hide chrome)", value=st.session_state.presentation, key="present_toggle2")
    st.session_state.presentation = pres

if st.session_state.presentation:
    st.markdown("""
    <style>
      .stSidebar, header, footer, [data-testid="stToolbar"] { display:none !important; }
      .block-container { padding-top: 12px !important; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------
# Render HTML (dropdown, fit to page)
# -----------------------
with st.expander("▼ Click to show/hide full dashboard (fit-to-page)", expanded=True):
    # For performance, show full HTML. Use a fairly tall height to capture 16:9
    st.components.v1.html(dashboard_html, height=820, scrolling=True)

# -----------------------
# Slide Mode: render one section at a time (fast, presentation-friendly)
# -----------------------
sections = spec.get("sections", []) or []
if sections:
    slide = sections[st.session_state.slide_index % len(sections)]
    st.markdown(f"### 🎞 Slide: {slide.get('name')}")
    # Render slide-specific KPI subset + charts
    # KPIs: we will show all KPIs but highlight top 3
    top_kpis = kpis[:3]
    cols_k = st.columns(min(3, len(top_kpis)))
    for i, k in enumerate(top_kpis):
        v = kpi_value(df_f, k.get("expr", "COUNT(*)"))
        cols_k[i].metric(k.get("title", ""), fmt_value(v, k.get("format", "auto")))

    # Charts in slide
    charts = slide.get("charts", [])
    if charts:
        # responsive columns: two across when possible
        row_chunks = [charts[i:i+2] for i in range(0, len(charts), 2)]
        for row in row_chunks:
            cols = st.columns(len(row))
            for c_el, colw in zip(row, cols):
                x = c_el.get("x"); y = c_el.get("y"); ctype = c_el.get("type")
                if not x or not y or x not in df.columns or y not in df.columns:
                    colw.markdown("**Invalid chart fields**")
                    continue
                if not ctype:
                    ctype = suggest_chart(detect_type(df[x]), detect_type(df[y]))
                # build a fast plotly figure and show
                if ctype in ("bar", "line", "area"):
                    data_agg = df_f.groupby(x)[y].sum(numeric_only=True).reset_index()
                else:
                    data_agg = df_f
                fig = None
                if ctype == "bar":
                    fig = px.bar(data_agg, x=x, y=y)
                elif ctype == "line":
                    fig = px.line(data_agg, x=x, y=y)
                elif ctype == "area":
                    fig = px.area(data_agg, x=x, y=y)
                elif ctype == "donut":
                    fig = px.pie(data_agg, names=x, values=y, hole=0.45)
                elif ctype == "scatter":
                    fig = px.scatter(df_f, x=x, y=y)
                elif ctype == "hist":
                    fig = px.histogram(df_f, x=y)
                else:
                    fig = px.bar(data_agg, x=x, y=y)
                fig.update_layout(height=420, margin=dict(l=8, r=8, t=36, b=8), template="plotly_white")
                colw.plotly_chart(fig, use_container_width=True)
                # Drill controls: show top N categories with drill buttons
                try:
                    top_df = df_f.groupby(x)[y].sum(numeric_only=True).reset_index().sort_values(by=y, ascending=False).head(5)
                    if not top_df.empty:
                        with colw.expander("Drill-down (top categories)"):
                            for _, r in top_df.iterrows():
                                v_label = str(r[x])
                                if st.button(f"Drill: {v_label}", key=f"drill_{x}_{v_label}"):
                                    # build drill rows (filtered df)
                                    drill_rows = df[df[x].astype(str) == v_label]
                                    # store drill info in session for a table view
                                    st.session_state.drill = {"x": x, "y": y, "value": v_label, "rows_html": None}
                                    st.session_state.drill["rows_df"] = drill_rows.head(100).to_dict(orient="records")
                                    st.experimental_rerun()
                except Exception:
                    pass

# -----------------------
# If drill session state exists, show drill table
# -----------------------
if st.session_state.drill:
    d = st.session_state.drill
    st.markdown(f"### 🔎 Drill: {d['x']} = {d['value']}")
    rows = d.get("rows_df", [])
    if rows:
        st.dataframe(pd.DataFrame(rows))
    else:
        st.info("No drill rows found or not loaded.")

# -----------------------
# Save rendered HTML into dashboard store (already done above)
# -----------------------
# done earlier: saved into st.session_state.dashboards[current]['html']

# -----------------------
# OPTIONAL: Allow user to edit spec JSON (power user)
# -----------------------
with st.expander("⚙️ Advanced: View / Edit Spec JSON (power user)", expanded=False):
    st.markdown("**Warning:** editing JSON directly requires valid spec structure. After editing, press 'Apply Spec' to re-render.")
    txt = json.dumps(spec, indent=2)
    edited = st.text_area("Spec JSON", value=txt, height=320, key="spec_editor")
    if st.button("Apply Spec"):
        try:
            new_spec = json.loads(edited)
            st.session_state.dashboards[st.session_state.current]["spec"] = new_spec
            st.experimental_rerun()
        except Exception as e:
            st.error("Invalid JSON: " + str(e))

# -----------------------
# End
# -----------------------
