
def compile_dashboard(intent):
    components = []
    y = 0
    for kpi in intent.get("kpis", []):
        components.append({
            "id": f"kpi_{kpi}",
            "type": "kpi",
            "metric": kpi,
            "x":0,"y":y,"w":3,"h":1
        })
    if "time_series" in intent:
        ts = intent["time_series"]
        components.append({
            "id":"ts1",
            "type":"line",
            "x":0,"y":y+1,"w":12,"h":4,
            "xField":ts["x"],
            "yField":ts["y"]
        })
    return {"components":components}
