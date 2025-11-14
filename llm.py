
from openai import OpenAI
import hashlib, json, time

class OpenAIClient:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    def chat(self, messages, model='gpt-4o-mini', temperature=0.25):
        return self.client.chat.completions.create(model=model, messages=messages, temperature=temperature)

# caching by dataset hash + prompt
_cache = {}
def generate_spec_cached(client: OpenAIClient, goal, columns, role):
    dataset_hash = hashlib.sha256(('|'.join(columns)).encode('utf-8')).hexdigest()
    key = f"{dataset_hash}:{goal}:{role}"
    if key in _cache:
        return _cache[key]
    prompt = strong_llm_prompt(goal, columns, role)
    try:
        r = client.chat([{'role':'user','content':prompt}])
        content = r.choices[0].message.content
        spec = clean_json(content)
    except Exception:
        spec = {}
    _cache[key] = spec
    return spec

def strong_llm_prompt(goal, columns, role):
    col_list = ', '.join(columns)
    return f"""You are a Principal BI Architect. Build a COMPLETE Power BI-style dashboard spec.
Goal: {goal}
Role: {role}
Columns: [{col_list}]
Return JSON only with keys: title, theme, filters, kpis, sections."""

# small helper to reuse clean_json from utils without circular import
from src.utils import clean_json
