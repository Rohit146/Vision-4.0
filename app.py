
import streamlit as st
import pandas as pd
import json
from llm import get_dashboard_intent_with_repair
from profiler import profile_dataframe
from compiler import compile_dashboard

st.set_page_config(layout="wide")
st.title("AI Dashboard Generator (Streamlit + React Hybrid)")

uploaded = st.file_uploader("Upload CSV or Excel")

if uploaded:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    schema = profile_dataframe(df.head(50))
    intent = get_dashboard_intent_with_repair(schema)
    dashboard = compile_dashboard(intent)

    st.subheader("Generated Dashboard JSON")
    st.json(dashboard)

    st.components.v1.html(
        open("frontend/build/index.html").read(),
        height=800,
        scrolling=True
    )
