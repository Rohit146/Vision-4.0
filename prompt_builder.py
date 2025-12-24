
def build_dashboard_prompt(user_prompt: str) -> str:
    return f"""
You are an expert BI dashboard designer.

Generate a HIGH-FIDELITY dashboard definition in JSON.

STRICT RULES:
1. Output VALID JSON ONLY
2. No markdown, no explanations
3. JSON MUST include:
   - meta (title, description)
   - layout (12 columns, 1920x1080)
   - components (NON-EMPTY array)

If unsure, FALL BACK to a minimal but valid dashboard with:
- 3 KPI cards
- 1 line chart
- 1 table

Each component MUST include:
- id
- type (metric_card | line_chart | bar_chart | table)
- title
- editable: true
- position {{ x, y, w, h }}

EXAMPLE COMPONENT:
{{
  "id": "kpi_1",
  "type": "metric_card",
  "title": "Total Revenue",
  "value": "—",
  "delta": null,
  "editable": true,
  "position": {{ "x": 0, "y": 0, "w": 3, "h": 2 }}
}}

USER REQUEST:
{user_prompt}

REMEMBER:
- components must NEVER be empty
- JSON ONLY
"""
