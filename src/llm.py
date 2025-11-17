
from openai import OpenAI
import hashlib, json, time
from src.utils import clean_json

class OpenAIClient:
    def __init__(self, api_key):
        # wrap client creation to be tolerant of missing/old SDKs
        try:
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            self.client = None
    def chat(self, messages, model='gpt-4o-mini', temperature=0.25):
        if not self.client:
            raise RuntimeError('OpenAI client not initialized')
        # compatible wrapper for various openai-python versions
        try:
            return self.client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        except Exception:
            # try old-style
            return self.client.chat.create(model=model, messages=messages, temperature=temperature)

_cache = {}
def generate_spec_cached(client: OpenAIClient, goal, columns, role):
    dataset_hash = hashlib.sha256(('|'.join(columns)).encode('utf-8')).hexdigest()
    key = f"{dataset_hash}:{goal}:{role}"
    if key in _cache:
        return _cache[key]
    prompt = strong_llm_prompt(goal, columns, role)
    try:
        if client is None:
            spec = {}
        else:
            r = client.chat([{'role':'user','content':prompt}])
            content = getattr(r.choices[0].message, 'content', None) or (r.choices[0].message.content if hasattr(r.choices[0].message, 'content') else r.choices[0].text)
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
Return STRICT JSON only with keys: title, theme, filters, kpis, sections.
Rules: sections MUST be a list of objects with keys name and charts. Charts are objects with x, y, type, and optional columns.
Do not include commentary or explanation. JSON only."""
