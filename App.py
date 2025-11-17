
import streamlit as st
from src.utils import safe_str, clean_json, apply_filters, load_dataframe, demo_dataframe
from src.llm import generate_spec_cached, OpenAIClient
from src.layout import smart_layout_v3, auto_layout_optimize, safe_sections
from src.renderer import build_dashboard_html, render_preview, make_explainability_table

import os, json, time

st.set_page_config(page_title="Auto-BI Modular Studio v3", layout="wide")
st.title("🧠 Auto-BI — Modular Studio v3 (Safe Sections + Auto Layout)")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.warning("No OPENAI_API_KEY found. You can still use the demo dataset and fallback spec.")
client = OpenAIClient(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Session state init
if 'df' not in st.session_state: st.session_state.df = None
if 'dashboards' not in st.session_state: st.session_state.dashboards = {}
if 'current' not in st.session_state: st.session_state.current = None
if 'filters' not in st.session_state: st.session_state.filters = {}
if 'slide_index' not in st.session_state: st.session_state.slide_index = 0

with st.sidebar:
    st.header('Data & Generate')
    uploaded = st.file_uploader('Upload Excel or CSV (or use demo)', type=['xlsx','xls','csv','csv'])
    if st.button("Use demo dataset"):
        st.session_state.df = demo_dataframe()
        st.success("Demo dataset loaded")
    role = st.selectbox('Audience role', ['BI Developer','Finance Analyst','Sales Leader','Operations Manager'])
    goal = st.text_area('Dashboard goal', 'Executive overview: revenue, margin, regional performance, trends')
    theme_choice = st.selectbox('Theme', ['light','dark'])
    st.markdown('---')
    if st.button('✨ Generate Dashboard (LLM)'):
        if st.session_state.df is None:
            st.error('Upload data first or use demo dataset.')
        else:
            if client is None:
                st.warning('OpenAI key missing — generating a basic fallback spec.')
                cols = list(st.session_state.df.columns)
                spec = {
                    "title": "Fallback Dashboard",
                    "theme": theme_choice,
                    "filters": [{"field": cols[0]}] if cols else [],
                    "kpis": [{"title":"Row Count","expr":"COUNT(" + (cols[0] if cols else "") + ")", "format":"decimal"}],
                    "sections": [
                        {"name":"Overview","charts":[{"x": cols[0] if cols else "", "y": cols[1] if len(cols)>1 else cols[0] if cols else "", "type":"bar"}]}
                    ]
                }
            else:
                spec = generate_spec_cached(client, goal, list(st.session_state.df.columns), role)
            if spec:
                name = f"Dashboard_{int(time.time())}"
                st.session_state.dashboards[name] = {'spec': spec, 'html': ''}
                st.session_state.current = name
                st.session_state.filters = {}
                st.success(f'Generated {name}')

# Load uploaded data if any
if uploaded:
    try:
        df = load_dataframe(uploaded)
        st.session_state.df = df
        st.success(f'Loaded {uploaded.name} ({len(df)} rows)')
    except Exception as e:
        st.error(f'Failed to load file: {e}')

if st.session_state.df is None:
    st.info('No data loaded. Use the sidebar to upload a file or click "Use demo dataset".')
    st.stop()

df = st.session_state.df

# Require a dashboard selected/generated
if not st.session_state.current or st.session_state.current not in st.session_state.dashboards:
    st.info('Generate a dashboard in the sidebar or open an existing saved dashboard.')
    st.stop()

cur = st.session_state.dashboards[st.session_state.current]
spec = cur.get('spec')
if not spec:
    st.info('No spec available. Generate one.'); st.stop()

# Ensure sections are safe dicts
spec['sections'] = safe_sections(spec.get('sections', []))

# Filters
with st.expander('Filters (live)', expanded=True):
    new_filters = {}
    for f in spec.get('filters', []):
        if not isinstance(f, dict):
            continue
        field = f.get('field')
        if field and field in df.columns:
            vals = sorted(df[field].dropna().astype(str).unique().tolist())
            sel = st.multiselect(field, vals[:1000], default=st.session_state.filters.get(field, []), key=f"flt_{field}")
            new_filters[field] = sel
    st.session_state.filters = new_filters

df_f = apply_filters(df, st.session_state.filters)

# Apply Smart Layout 3.0 + Auto-layout optimizer
with st.spinner('Optimizing layout (Smart Layout 3.0) and packing visuals...'):
    spec = smart_layout_v3(spec, df_f, client=client, add_anomaly_charts=True)
    spec = auto_layout_optimize(spec)
    st.session_state.dashboards[st.session_state.current]['spec'] = spec

# Build dashboard HTML and explainability table
with st.spinner('Rendering dashboard HTML...'):
    html = build_dashboard_html(spec, df_f)
    st.session_state.dashboards[st.session_state.current]['html'] = html
    explain_df = make_explainability_table(spec, df_f)

st.subheader('Dashboard Preview')
render_preview(html, height=900)

st.subheader('Explainability & Ranking')
st.dataframe(explain_df)

# Power-user: view/edit spec
with st.expander('⚙️ Power-user: View / Edit Spec JSON', expanded=False):
    raw = json.dumps(spec, indent=2)
    edited = st.text_area('Spec JSON', value=raw, height=320)
    if st.button('Apply Spec'):
        try:
            new_spec = json.loads(edited)
            st.session_state.dashboards[st.session_state.current]['spec'] = new_spec
            st.experimental_rerun()
        except Exception as e:
            st.error('Invalid JSON: ' + safe_str(e))
