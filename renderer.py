import streamlit as st
import plotly.express as px
import pandas as pd

def render_dashboard(dashboard):
    st.title(dashboard.get("meta", {}).get("title", "Dashboard"))
    for c in dashboard.get("components", []):
        st.subheader(c.get("title", ""))
        if c["type"] == "metric_card":
            st.metric(c["title"], c.get("value", "-"), c.get("delta"))
        elif c["type"] == "line_chart":
            df = pd.DataFrame({"x": range(10), "y": range(10)})
            st.plotly_chart(px.line(df, x="x", y="y"), use_container_width=True)
        elif c["type"] == "table":
            st.dataframe(pd.DataFrame(c.get("data", {})))