
import pandas as pd, io, os, json, re
def safe_str(v):
    try:
        return str(v).replace('{','[').replace('}',']')
    except:
        return 'unknown'

def clean_json(text: str):
    text = re.sub(r'```(json)?|```', '', (text or ''))
    m = re.search(r'\{[\s\S]*\}', text)
    if not m:
        return {}
    s = m.group(0)
    try:
        return json.loads(s)
    except:
        try:
            return json.loads(s.replace("'", '"'))
        except:
            return {}

def load_dataframe(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        xl = pd.read_excel(uploaded_file, sheet_name=None)
        return xl[list(xl.keys())[0]] if isinstance(xl, dict) else xl

def apply_filters(df, filters):
    out = df.copy()
    for k,v in (filters or {}).items():
        if k in out.columns and v:
            out = out[out[k].astype(str).isin(v)]
    return out

def demo_dataframe():
    import pandas as pd, numpy as np
    n = 300
    rng = pd.date_range('2024-01-01', periods=n, freq='D')
    df = pd.DataFrame({
        'order_date': rng,
        'region': np.random.choice(['North','South','East','West'], size=n),
        'product': np.random.choice(['Widget A','Widget B','Widget C'], size=n),
        'revenue': (np.random.rand(n) * 10000).round(2),
        'cost': (np.random.rand(n) * 7000).round(2)
    })
    df['profit'] = df['revenue'] - df['cost']
    return df
