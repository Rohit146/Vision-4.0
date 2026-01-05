
import pandas as pd

def infer_type(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    return "categorical"

def profile_dataframe(df):
    return [{
        "name": col,
        "type": infer_type(df[col]),
        "null_pct": float(df[col].isna().mean()),
        "cardinality": int(df[col].nunique(dropna=True))
    } for col in df.columns]
