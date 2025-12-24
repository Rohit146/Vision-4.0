import json, os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def explain_dashboard(dashboard):
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "Explain this dashboard for business users."},
            {"role": "user", "content": json.dumps(dashboard)}
        ]
    )
    return response.choices[0].message.content