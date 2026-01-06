
from openai import OpenAI
import json, os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_dashboard_intent_with_repair(schema, retries=2):
    prompt = {
        "columns": schema,
        "rules": [
            "Return strict JSON",
            "Include kpis, time_series if applicable"
        ]
    }

    for _ in range(retries):
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            messages=[
                {"role":"system","content":"You are a BI planner. JSON only."},
                {"role":"user","content":json.dumps(prompt)}
            ]
        )
        text = resp.choices[0].message.content
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            prompt["repair"] = f"Fix invalid JSON: {text}"

    raise ValueError("LLM JSON could not be repaired")
