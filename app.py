
import streamlit as st
from src.utils import safe_str, clean_json, apply_filters, load_dataframe
from src.llm import generate_spec_cached, OpenAIClient
from src.layout import smart_layout_v3, auto_layout_optimize, safe_sections
from src.renderer import build_dashboard_html, render_preview, make_explainability_table

import os, json, time

st.set_page_config(page_title="Auto-BI Modular Studio", layout="wide")
st.title("🧠 Auto-BI — Modular Studio (Smart Layout 3.0 + Explainability)")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.warning("Missing OPENAI_API_KEY in Streamlit secrets. Add it under App settings -> Secrets.")
    st.stop()

client = OpenAIClient(api_key=OPENAI_API_KEY)

# Session state
if 'df' not in st.session_state: st.session_state.df = None
if 'dashboards' not in st.session_state: st.session_state.dashboards = {}
if 'current' not in st.session_state: st.session_state.current = None
if 'filters' not in st.session_state: st.session_state.filters = {}
if 'slide_index' not in st.session_state: st.session_state.slide_index = 0

with st.sidebar:
    st.header('Data')
    uploaded = st.file_uploader('Upload Excel or CSV', type=['xlsx','xls','csv'])
    role = st.selectbox('Audience role', ['BI Developer','Finance Analyst','Sales Leader','Operations Manager'])
    goal = st.text_area('Dashboard goal', 'Executive overview: revenue, margin, regional performance, trends')
    theme_choice = st.selectbox('Theme', ['light','dark'])
    if st.button('✨ Generate Dashboard (LLM)'):
        if st.session_state.df is None:
            st.error('Upload data first.'); st.stop()
        spec = generate_spec_cached(client, goal, list(st.session_state.df.columns), role)
        if spec:
            name = f"Dashboard_{int(time.time())}"
            st.session_state.dashboards[name] = {'spec': spec, 'html': ''}
            st.session_state.current = name
            st.success(f'Generated {name}')

# Load data
if uploaded:
    df = load_dataframe(uploaded)
    st.session_state.df = df
    st.success(f'Loaded {uploaded.name} ({len(df)} rows)')

if st.session_state.df is None:
    st.info('Upload a dataset to continue.')
    st.stop()

df = st.session_state.df

# Open dashboard
if not st.session_state.current or st.session_state.current not in st.session_state.dashboards:
    st.info('Generate or open a dashboard from sidebar.')
    st.stop()

cur = st.session_state.dashboards[st.session_state.current]
spec = cur.get('spec')
if not spec:
    st.info('No spec available. Generate one.'); st.stop()

# Safe sections conversion (fix for string sections)
spec['sections'] = safe_sections(spec.get('sections', []))

# Filters
with st.expander('Filters', expanded=True):
    new_filters = {}
    for f in spec.get('filters', []):
        field = f.get('field') if isinstance(f, dict) else None
        if field and field in df.columns:
            vals = sorted(df[field].dropna().astype(str).unique().tolist())
            sel = st.multiselect(field, vals[:1000], default=st.session_state.filters.get(field, []), key=f"flt_{field}")
            new_filters[field] = sel
    st.session_state.filters = new_filters

df_f = apply_filters(df, st.session_state.filters)

# Smart layout + auto layout
with st.spinner('Optimizing layout...'):
    spec = smart_layout_v3(spec, df_f, client=client, add_anomaly_charts=True)
    spec = auto_layout_optimize(spec)
    st.session_state.dashboards[st.session_state.current]['spec'] = spec

# Build HTML and explainability
with st.spinner('Rendering dashboard...'):
    html = build_dashboard_html(spec, df_f)
    st.session_state.dashboards[st.session_state.current]['html'] = html

st.subheader('Dashboard Preview')
render_preview(html, height=900)

st.subheader('Explainability & Ranking')
explain_table = make_explainability_table(spec, df_f)
st.dataframe(explain_table)
