
import streamlit as st

def edit_component(component, data_contract=None):
    st.markdown("#### Edit Component")

    component["title"] = st.text_input(
        "Title",
        component.get("title",""),
        key=f"title_{component['id']}"
    )

    component["type"] = st.selectbox(
        "Component Type",
        ["metric_card","line_chart","bar_chart","table"],
        index=["metric_card","line_chart","bar_chart","table"].index(component["type"]),
        key=f"type_{component['id']}"
    )

    col1, col2 = st.columns(2)

    with col1:
        component["position"]["w"] = st.slider("Width",1,12,component["position"]["w"],key=f"w_{component['id']}")
        component["position"]["x"] = st.slider("X",0,11,component["position"]["x"],key=f"x_{component['id']}")

    with col2:
        component["position"]["h"] = st.slider("Height",1,10,component["position"]["h"],key=f"h_{component['id']}")
        component["position"]["y"] = st.slider("Y",0,20,component["position"]["y"],key=f"y_{component['id']}")

    if data_contract:
        component.setdefault("binding",{})
        metrics = [m["name"] for m in data_contract.get("metrics",[])]
        dims = [d["name"] for d in data_contract.get("dimensions",[])]

        if metrics:
            component["binding"]["metric"] = st.selectbox(
                "Metric", metrics,
                index=metrics.index(component["binding"].get("metric",metrics[0])),
                key=f"metric_{component['id']}"
            )
        if dims:
            component["binding"]["dimension"] = st.selectbox(
                "Dimension", dims,
                index=dims.index(component["binding"].get("dimension",dims[0])),
                key=f"dim_{component['id']}"
            )

    return component
