
import json, re

def sanitize_llm_output(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    return match.group(1) if match else text

def parse_json_with_retry(raw: str, retries: int = 2) -> dict:
    last_error = None
    for _ in range(retries + 1):
        try:
            return json.loads(raw)
        except Exception as e:
            last_error = e
            raw = raw.replace("\n", "").replace("\t", "")
    raise ValueError(f"Unable to parse JSON after retries: {last_error}")
