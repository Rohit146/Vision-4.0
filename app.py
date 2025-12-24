
import streamlit as st, json
from llm_engine import generate_dashboard
from validator import normalize_dashboard
from renderer import render_component
from explain_ai import explain_dashboard
from data_binding_ai import generate_data_contract, bind_data_to_dashboard
from component_editor import edit_component

st.set_page_config(layout="wide")
st.sidebar.title("AI Dashboard Copilot")

prompt = st.sidebar.text_area("Dashboard prompt", height=150)
data_prompt = st.sidebar.text_area("Data description", height=120)

if "dashboard" not in st.session_state:
    st.session_state.dashboard = None

if st.sidebar.button("Generate Dashboard"):
    try:
        raw = generate_dashboard(prompt)
        st.session_state.dashboard = normalize_dashboard(raw)
    except Exception as e:
        st.error(str(e))

if st.sidebar.button("Bind Data") and st.session_state.dashboard:
    try:
        contract = generate_data_contract(data_prompt)
        st.session_state.dashboard = bind_data_to_dashboard(st.session_state.dashboard, contract)
    except Exception as e:
        st.error(str(e))

if st.session_state.dashboard:
    st.title(st.session_state.dashboard["meta"].get("title","Dashboard"))

    edit_mode = st.toggle("Edit Mode", value=False)

    for i, comp in enumerate(st.session_state.dashboard["components"]):
        with st.container(border=True):
            if edit_mode:
                st.session_state.dashboard["components"][i] = edit_component(
                    comp,
                    st.session_state.dashboard.get("data_contract")
                )
            else:
                render_component(comp)

    if st.button("Explain this dashboard"):
        st.markdown(explain_dashboard(st.session_state.dashboard))

    with st.expander("Edit Raw JSON"):
        edited = st.text_area(
            "Dashboard JSON",
            json.dumps(st.session_state.dashboard, indent=2),
            height=400
        )
        if st.button("Apply JSON"):
            st.session_state.dashboard = json.loads(edited)
