import os
import json

try:
    from openai import OpenAI
except ImportError:
    raise RuntimeError(
        "OpenAI SDK not installed. "
        "Ensure `openai>=1.3.0` is in requirements.txt"
    )

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_dashboard_intent_with_repair(schema, retries=2):
    prompt = {
        "columns": schema,
        "rules": [
            "Return valid JSON only",
            "Suggest kpis and time_series if applicable"
        ]
    }

    for _ in range(retries):
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a BI planner. JSON only."},
                {"role": "user", "content": json.dumps(prompt)}
            ],
        )

        content = response.choices[0].message.content

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            prompt["repair"] = f"Fix invalid JSON:\n{content}"

    return {}
