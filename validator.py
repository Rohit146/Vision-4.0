
def normalize_dashboard(dashboard):
    dashboard.setdefault("layout", {})
    dashboard["layout"].update({
        "canvas_width": 1920,
        "canvas_height": 1080,
        "columns": 12
    })
    for c in dashboard.get("components", []):
        c.setdefault("editable", True)
        c.setdefault("position", {"x":0,"y":0,"w":3,"h":2})
    return dashboard
