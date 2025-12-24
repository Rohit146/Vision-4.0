
import uuid, os
from openai import OpenAI
from utils_llm import sanitize_llm_output, parse_json_with_retry
from prompt_builder import build_dashboard_prompt

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_dashboard(user_prompt: str) -> dict:
    full_prompt = build_dashboard_prompt(user_prompt)

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.4
    )

    raw = sanitize_llm_output(response.choices[0].message.content)
    dashboard = parse_json_with_retry(raw)

    dashboard.setdefault("meta", {})
    dashboard["meta"].setdefault("dashboard_id", str(uuid.uuid4()))
    return dashboard
