
import json, uuid, os
from openai import OpenAI
from utils_llm import sanitize_llm_output, parse_json_with_retry

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = "Return ONLY valid JSON. No markdown."

def generate_dashboard(prompt):
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    raw = sanitize_llm_output(response.choices[0].message.content)
    dashboard = parse_json_with_retry(raw)
    dashboard.setdefault("meta", {})["dashboard_id"] = str(uuid.uuid4())
    return dashboard
