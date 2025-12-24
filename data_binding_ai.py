import json, os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_data_contract(text):
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "Generate BI data contract JSON only"},
            {"role": "user", "content": text}
        ]
    )
    return json.loads(response.choices[0].message.content)

def bind_data_to_dashboard(dashboard, contract):
    dashboard["data_contract"] = contract
    for c in dashboard.get("components", []):
        if c["type"] in ["line_chart","bar_chart"]:
            c["binding"] = {
                "metric": contract["metrics"][0]["name"],
                "dimension": contract["dimensions"][0]["name"]
            }
    return dashboard