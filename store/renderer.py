
import streamlit as st
import pandas as pd
import plotly.express as px

def render_component(component):
    if component["type"] == "metric_card":
        st.metric(component["title"], component.get("value","—"), component.get("delta"))
    elif component["type"] == "line_chart":
        df = pd.DataFrame({"x": range(10), "y": range(10)})
        st.plotly_chart(px.line(df,x="x",y="y"), use_container_width=True)
    elif component["type"] == "bar_chart":
        df = pd.DataFrame({"x":["A","B","C"],"y":[4,7,2]})
        st.plotly_chart(px.bar(df,x="x",y="y"), use_container_width=True)
    elif component["type"] == "table":
        st.dataframe(pd.DataFrame(component.get("data",{})))
