import os
import json
import requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in Streamlit secrets")


def call_openai(messages, model="gpt-4.1-mini", temperature=0):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }

    response = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenAI API error {response.status_code}: {response.text}"
        )

    return response.json()["choices"][0]["message"]["content"]


def get_dashboard_intent_with_repair(schema, retries=2):
    prompt = {
        "columns": schema,
        "rules": [
            "Return STRICT JSON only",
            "Suggest kpis and time_series if applicable",
            "No markdown, no explanation"
        ]
    }

    messages = [
        {"role": "system", "content": "You are a BI planner. JSON only."},
        {"role": "user", "content": json.dumps(prompt)}
    ]

    for _ in range(retries):
        content = call_openai(messages)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            messages.append({
                "role": "user",
                "content": f"Fix this invalid JSON and return JSON only:\n{content}"
            })

    return {}
