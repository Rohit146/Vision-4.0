
def normalize_dashboard(dashboard):
    dashboard.setdefault("meta", {})
    dashboard["meta"].setdefault("title", "AI Generated Dashboard")
    dashboard["meta"].setdefault("description", "")

    dashboard.setdefault("layout", {})
    dashboard["layout"].update({
        "canvas_width": 1920,
        "canvas_height": 1080,
        "columns": 12
    })

    dashboard.setdefault("components", [])

    if not dashboard["components"]:
        dashboard["components"].append({
            "id": "fallback_1",
            "type": "metric_card",
            "title": "Placeholder Metric",
            "value": "—",
            "delta": None,
            "editable": True,
            "position": {"x": 0, "y": 0, "w": 3, "h": 2}
        })

    for c in dashboard["components"]:
        c.setdefault("editable", True)
        c.setdefault("position", {"x": 0, "y": 0, "w": 3, "h": 2})
        c.setdefault("title", "Untitled Component")
        c.setdefault("type", "metric_card")

    return dashboard
