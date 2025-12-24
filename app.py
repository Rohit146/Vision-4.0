import streamlit as st
import json
from llm_engine import generate_dashboard
from validator import normalize_dashboard
from renderer import render_dashboard
from explain_ai import explain_dashboard
from data_binding_ai import generate_data_contract, bind_data_to_dashboard

st.set_page_config(layout="wide")

st.sidebar.title("AI Dashboard Copilot")

prompt = st.sidebar.text_area("Dashboard prompt", height=160)
data_prompt = st.sidebar.text_area("Data description (optional)", height=120)

if "dashboard" not in st.session_state:
    st.session_state.dashboard = None

if st.sidebar.button("Generate Dashboard"):
    raw = generate_dashboard(prompt)
    st.session_state.dashboard = normalize_dashboard(raw)

if st.sidebar.button("Bind Data") and st.session_state.dashboard:
    contract = generate_data_contract(data_prompt)
    st.session_state.dashboard = bind_data_to_dashboard(st.session_state.dashboard, contract)

if st.session_state.dashboard:
    render_dashboard(st.session_state.dashboard)

    st.divider()

    if st.button("Explain this dashboard"):
        st.markdown(explain_dashboard(st.session_state.dashboard))

    with st.expander("Edit JSON"):
        edited = st.text_area("Dashboard JSON", json.dumps(st.session_state.dashboard, indent=2), height=400)
        if st.button("Apply JSON"):
            st.session_state.dashboard = json.loads(edited)

    st.download_button(
        "Download Dashboard JSON",
        json.dumps(st.session_state.dashboard, indent=2),
        file_name="dashboard.json"
    )