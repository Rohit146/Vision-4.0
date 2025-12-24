import json, uuid, os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a senior BI architect.
Generate an enterprise dashboard JSON.
Rules:
- 16:9 (1920x1080)
- 12-column grid
- VALID JSON ONLY
"""

def generate_dashboard(prompt):
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    dashboard = json.loads(response.choices[0].message.content)
    dashboard.setdefault("meta", {})["dashboard_id"] = str(uuid.uuid4())
    return dashboard