# Auto-BI Studio — Full app with Smart Layout 3.0 + Explainability Panel
# Save as app.py and run with: streamlit run app.py
# NOTE: keep your OPENAI_API_KEY in Streamlit Secrets or environment variables.

import streamlit as st
import pandas as pd
import plotly.express as px
from jinja2 import Template
from openai import OpenAI
import os, re, json, hashlib, time, math, numpy as np
from collections import defaultdict

st.set_page_config(page_title='Auto-BI Studio (SL3)', layout='wide', initial_sidebar_state='expanded')
st.title('🧠 Auto-BI — Power BI-style Studio (Smart Layout 3.0 + Explainability)')

# -----------------------------
# OpenAI client
# -----------------------------
OPENAI_API_KEY = st.secrets.get('OPENAI_API_KEY', os.getenv('OPENAI_API_KEY'))
if not OPENAI_API_KEY:
    st.warning('Missing OPENAI_API_KEY in Streamlit secrets. Add it under App settings -> Secrets.')
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Utilities
# -----------------------------
def safe_str(v):
    try:
        s = str(v)
        return s.replace('{','[').replace('}',']')
    except:
        return 'unknown'

def clean_json(text: str):
    text = re.sub(r'```(json)?|```', '', (text or ''))
    m = re.search(r'\{[\s\S]*\}', text)
    if not m:
        return {}
    s = re.sub(r',(?=\s*[}\]])', lambda m: m.group(0)[:-1], m.group(0)) if False else m.group(0)
    try:
        return json.loads(s)
    except:
        try:
            return json.loads(s.replace("'", '"'))
        except:
            return {}

def detect_type(series: pd.Series):
    if pd.api.types.is_numeric_dtype(series): return 'numeric'
    if pd.api.types.is_datetime64_any_dtype(series): return 'date'
    return 'categorical'

def fmt_value(v, kind='auto'):
    try:
        if kind == 'currency': return f'₹{float(v):,.0f}'
        if kind == 'pct': return f'{float(v)*100:.2f}%'
        if kind == 'decimal': return f'{float(v):,.2f}'
        return f'{float(v):,.0f}'
    except:
        return str(v)

def kpi_value(df, expr):
    try:
        e = expr.strip().upper()
        if e.startswith('SUM('):
            col = expr[4:-1]; return pd.to_numeric(df[col], errors='coerce').sum()
        if e.startswith('AVG('):
            col = expr[4:-1]; return pd.to_numeric(df[col], errors='coerce').mean()
        if e.startswith('COUNT('):
            col = expr[6:-1]; return df[col].count() if col in df.columns else len(df)
        if e.startswith('DISTINCT('):
            col = expr[9:-1]; return df[col].nunique()
    except:
        return 0
    return 0

# -----------------------------
# Plot generation (handles table)
# -----------------------------
def make_plot_html(df, x, y, ctype, theme='light'):
    try:
        if ctype == 'table':
            if isinstance(y, list):
                cols = [c for c in y if c in df.columns]
                return df[cols].head(40).to_html(index=False)
            else:
                cols = [c for c in [x, y] if c in df.columns]
                return df[cols].head(40).to_html(index=False)
        if x not in df.columns or (isinstance(y, str) and y not in df.columns):
            return "<div style='color:#999'>Invalid fields</div>"

        if ctype in ('bar','line','area'):
            agg = df.groupby(x)[y].sum(numeric_only=True).reset_index()
        else:
            agg = df

        if ctype == 'bar':
            fig = px.bar(agg, x=x, y=y)
        elif ctype == 'line':
            fig = px.line(agg, x=x, y=y)
        elif ctype == 'area':
            fig = px.area(agg, x=x, y=y)
        elif ctype == 'donut':
            fig = px.pie(agg, names=x, values=y, hole=0.45)
        elif ctype == 'scatter':
            fig = px.scatter(df, x=x, y=y)
        elif ctype == 'hist':
            fig = px.histogram(df, x=y)
        else:
            fig = px.bar(agg, x=x, y=y)

        if theme == 'dark':
            fig.update_layout(template='plotly_dark')
        else:
            fig.update_layout(template='plotly_white')

        try:
            fig.update_traces(transition_duration=450)
            fig.update_layout(transition={'duration':350,'easing':'cubic-in-out'})
        except:
            pass
        fig.update_layout(margin=dict(l=8,r=8,t=36,b=8), height=380)
        return fig.to_html(include_plotlyjs='cdn', full_html=False)
    except Exception as e:
        return f"<div style='color:#c00'>Chart error: {safe_str(e)}</div>"

# -----------------------------
# Strong LLM prompt for specs
# -----------------------------
def strong_llm_prompt(goal, columns, role):
    col_list = ', '.join(columns)
    return f"""You are a Principal BI Architect. Build a COMPLETE Power BI-style dashboard spec.
GOAL: {goal}
ROLE: {role}
 COLUMNS: [{col_list}]
Return only valid JSON with keys: title, theme, filters, kpis, sections (Overview, Performance, Trends, Deep Dive, Risks).
Rules: MUST 5-7 KPIs, 6-10 charts, each section 1-3 visuals, Deep Dive includes tables. JSON only."""

# -----------------------------
# Smart Layout 3.0 helpers (condensed)
# -----------------------------
def _pearson_corr(df, x, y):
    try:
        if x not in df.columns or y not in df.columns: return 0.0
        xr = pd.to_numeric(df[x], errors='coerce')
        yr = pd.to_numeric(df[y], errors='coerce')
        mask = xr.notna() & yr.notna()
        if mask.sum() < 3: return 0.0
        return float(np.corrcoef(xr[mask], yr[mask])[0,1])
    except:
        return 0.0

def _entropy_of_series(s):
    try:
        counts = s.value_counts(dropna=True).astype(float)
        p = counts / counts.sum()
        ent = - (p * np.log2(p + 1e-12)).sum()
        n = max(len(counts),1)
        return float(ent / math.log2(n + 1)) if n>1 else 0.0
    except:
        return 0.0

def _compute_insight_metrics(df, x, y):
    try:
        if x not in df.columns: return {'score':0.05}
        richness = min(df[x].nunique()/50.0, 1.0)
        v=0.0; meany=0.0
        if y in df.columns:
            numeric_y = pd.to_numeric(df[y], errors='coerce').dropna()
            v = float(numeric_y.var()) if not numeric_y.empty else 0.0
            meany = float(numeric_y.mean()) if not numeric_y.empty else 0.0
        var_score = min(v/(abs(meany)+1e-6), 2.0)
        var_score = float(np.tanh(var_score))
        slope = 0.0
        try:
            if pd.api.types.is_datetime64_any_dtype(df[x]) and y in df.columns:
                tmp = df[[x,y]].dropna().copy(); tmp[x]=pd.to_datetime(tmp[x],errors='coerce')
                tmp = tmp.set_index(x).resample('M').sum(numeric_only=True).reset_index()
                if not tmp.empty:
                    s = pd.to_numeric(tmp[y].dropna(), errors='coerce')
                    if not s.empty:
                        xarr = np.arange(len(s)); A = np.vstack([xarr, np.ones(len(xarr))]).T
                        m, _ = np.linalg.lstsq(A, s.values, rcond=None)[0]
                        slope = float(m / (abs(s.mean())+1e-9))
        except:
            slope = 0.0
        corr = _pearson_corr(df,x,y) if y in df.columns else 0.0
        grp = df.groupby(x)[y].sum(numeric_only=True) if y in df.columns else pd.Series([])
        conc = float(grp.sort_values(ascending=False).head(3).sum()/(grp.sum()+1e-12)) if (not grp.empty and grp.sum()>0) else 0.0
        ent = _entropy_of_series(df[x])
        score = 0.35*richness + 0.25*var_score + 0.15*abs(slope) + 0.15*min(abs(corr),1.0) + 0.10*conc
        score = float(max(0.01, min(1.0, score)))
        return {'richness':float(richness),'var_score':float(var_score),'slope':float(slope),'corr':float(corr),'conc':float(conc),'entropy':float(ent),'score':score}
    except:
        return {'score':0.05}

def _is_duplicate(candidate, existing_list):
    cx = candidate.get('x'); cy = candidate.get('y'); ctype = str(candidate.get('type','')).lower()
    for ex in existing_list:
        exx = ex.get('x'); exy = ex.get('y'); ex_type = str(ex.get('type','')).lower()
        if cx==exx and cy==exy: return True
        if ctype=='scatter' and ex_type=='scatter' and cx==exy and cy==exx: return True
        if ctype=='table' and ex_type=='table':
            a=set(candidate.get('columns') or []); b=set(ex.get('columns') or [])
            if a and b and (a==b or a.issubset(b) or b.issubset(a)): return True
    return False

def _recommend_visual(df, x, y):
    if x not in df.columns: return 'table'
    x_type = 'numeric' if pd.api.types.is_numeric_dtype(df[x]) else ('date' if pd.api.types.is_datetime64_any_dtype(df[x]) else 'categorical')
    if isinstance(y, list): return 'table'
    y_type = None
    if y in df.columns: y_type = 'numeric' if pd.api.types.is_numeric_dtype(df[y]) else 'categorical'
    metrics = _compute_insight_metrics(df, x, y if isinstance(y,str) else (y[0] if isinstance(y,list) and y else None))
    s = metrics.get('score',0.1)
    if y_type=='numeric' and x_type=='date': return 'line' if abs(metrics.get('slope',0))>0.01 else 'area'
    if x_type=='categorical' and y_type=='numeric':
        if df[x].nunique()<=6 and s<0.25: return 'donut' if s<0.15 else 'bar'
        return 'bar'
    if x_type=='numeric' and y_type=='numeric': return 'scatter'
    return 'bar'

# Smart Layout v3: (see conversation for details)
_section_narrative_cache = {}
def _cache_key_for_section(title, summary_str):
    s = f"{title}|{summary_str}"; return hashlib.sha256(s.encode('utf-8')).hexdigest()

def section_narrative_llm(title, summary, client):
    key = _cache_key_for_section(title, json.dumps(summary, sort_keys=True))
    if key in _section_narrative_cache: return _section_narrative_cache[key]
    prompt = f"Write a concise (1-2 sentence) executive summary for the section '{title}' using this summary: {summary}. Keep it action-oriented."
    try:
        r = client.chat.completions.create(model='gpt-4o-mini', messages=[{'role':'user','content':prompt}], temperature=0.35)
        out = (r.choices[0].message.content or '').strip()
    except:
        out = ''
    _section_narrative_cache[key] = out
    return out

def smart_layout_v3(spec, df, client=None, add_anomaly_charts=True):
    if not spec or 'sections' not in spec: return spec
    collected = []
    for sec in spec.get('sections', []):
        for ch in sec.get('charts', []):
            x = ch.get('x'); y = ch.get('y'); cols = ch.get('columns') if isinstance(ch.get('columns'), list) else None
            metrics = _compute_insight_metrics(df, x, y if isinstance(y,str) else (y[0] if isinstance(y,list) and y else None))
            reco = ch.get('type') or _recommend_visual(df, x, y)
            collected.append({'orig_section':sec.get('name'),'chart':ch,'x':x,'y':y,'columns':cols,'metrics':metrics,'reco_type':reco})
    unique_charts = []
    for item in sorted(collected, key=lambda it: it['metrics'].get('score',0), reverse=True):
        cand = item['chart']
        if not _is_duplicate(cand, [uc['chart'] for uc in unique_charts]):
            if not cand.get('type'): cand['type'] = item['reco_type']
            unique_charts.append(item)
    semantic_map = {'revenue':'Performance','sales':'Performance','gmv':'Performance','profit':'Profitability','margin':'Profitability','cogs':'Profitability','cost':'Profitability','customer':'Customer','segment':'Customer','churn':'Risk','region':'Geography','country':'Geography','state':'Geography','date':'Time','month':'Time','year':'Time','week':'Time','risk':'Risk','variance':'Risk','anomaly':'Risk'}
    def detect(field):
        if not field: return 'Misc'
        f = str(field).lower()
        for k,v in semantic_map.items():
            if k in f: return v
        return 'Misc'
    grouped = defaultdict(list)
    for item in unique_charts:
        ch = item['chart']; x = item['x']; y = item['y']; sem = detect(x) or detect(y)
        if (isinstance(ch.get('type'),str) and ch.get('type').lower()=='table') or ch.get('type')=='table': sem='DeepDive'
        grouped[sem].append(item)
    if add_anomaly_charts:
        time_candidates = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or any(k in col.lower() for k in ['date','month','year']): time_candidates.append(col)
        for tcol in time_candidates:
            for mcol in df.select_dtypes(include=[float,int]).columns:
                try:
                    tmp = df[[tcol,mcol]].dropna().copy(); tmp[tcol] = pd.to_datetime(tmp[tcol], errors='coerce')
                    agg = tmp.set_index(tcol).resample('M').sum(numeric_only=True).fillna(0)
                    if len(agg) < 6: continue
                    vals = agg[mcol].values; z = np.abs((vals - vals.mean())/(vals.std()+1e-9))
                    if np.any(z>2.8):
                        cand = {'x':tcol,'y':mcol,'type':'line','note':'anomaly_suggested'}
                        if not any(_is_duplicate(cand, [it['chart'] for it in unique_charts])):
                            grouped['Risk'].append({'orig_section':'Auto-Risk','chart':cand,'x':tcol,'y':mcol,'columns':None,'metrics':{'score':0.9},'reco_type':'line'})
                except:
                    pass
    priority = ['Performance','Profitability','Geography','Customer','Time','DeepDive','Risk','Misc']
    final_sections = []
    for p in priority:
        items = grouped.get(p) or grouped.get(p if p!='DeepDive' else 'DeepDive')
        if items:
            items_sorted = sorted(items, key=lambda it: (it['metrics'].get('score',0), it['metrics'].get('var_score',0)), reverse=True)
            charts = [it['chart'] for it in items_sorted]
            sect_name = p if p!='DeepDive' else 'Deep Dive'
            final_sections.append({'name':sect_name,'charts':charts})
    deep_idx = None
    for i,s in enumerate(final_sections):
        if s['name'].lower().startswith('deep'): deep_idx = i; break
    if deep_idx is not None and deep_idx != len(final_sections)-1:
        dd = final_sections.pop(deep_idx); final_sections.append(dd)
    if client is not None:
        for s in final_sections:
            sumry = {'charts':[]}
            for ch in s.get('charts',[]):
                x = ch.get('x'); y = ch.get('y'); t = ch.get('type')
                metrics = _compute_insight_metrics(df, x, y if isinstance(y,str) else (y[0] if isinstance(y,list) and y else None))
                sumry['charts'].append({'title':f'{x} vs {y}','type':t,'score':metrics.get('score')})
            narrative = section_narrative_llm(s.get('name','Section'), sumry, client)
            s['narrative'] = narrative
    spec['sections'] = final_sections
    return spec

# -----------------------------
# Auto-layout optimizer (slot packing)
# -----------------------------
def auto_layout_optimize(spec):
    if not spec or 'sections' not in spec: return spec
    order = ['Overview','Performance','Trends','Deep Dive','Risks & Notes']
    sections = spec['sections']
    sections = sorted(sections, key=lambda s: order.index(s.get('name')) if s.get('name') in order else 999)
    WEIGHTS = {'table':3,'bar':2,'line':2,'area':2,'scatter':2,'hist':1,'donut':1,None:2}
    optimized = []
    for sec in sections:
        charts = sec.get('charts', [])
        if not charts: continue
        weighted = []
        for ch in charts:
            ctype = str(ch.get('type','bar')).lower(); weight = WEIGHTS.get(ctype,2)
            weighted.append((ch,weight))
        rows=[]; cur=[]; curw=0
        for ch,w in weighted:
            if w==3:
                if cur: rows.append(cur); cur=[]; curw=0
                rows.append([ch]); continue
            if curw + w <= 3:
                cur.append(ch); curw += w
            else:
                rows.append(cur); cur=[ch]; curw = w
        if cur: rows.append(cur)
        new_order=[]
        for row in rows:
            for ch in row: new_order.append(ch)
        sec['charts'] = new_order
        optimized.append(sec)
    spec['sections'] = optimized
    return spec

# -----------------------------
# Session state
# -----------------------------
if 'df' not in st.session_state: st.session_state.df=None
if 'dashboards' not in st.session_state: st.session_state.dashboards={}
if 'current' not in st.session_state: st.session_state.current=None
if 'filters' not in st.session_state: st.session_state.filters={}
if 'slide_index' not in st.session_state: st.session_state.slide_index=0
if 'presentation' not in st.session_state: st.session_state.presentation=False

# -----------------------------
# Sidebar UI
# -----------------------------
with st.sidebar:
    st.header('Data & Generate')
    uploaded = st.file_uploader('Upload Excel or CSV', type=['xlsx','xls','csv'])
    role = st.selectbox('Audience role', ['BI Developer','Finance Analyst','Sales Leader','Operations Manager'])
    goal = st.text_area('Dashboard goal', 'Executive overview: revenue, margin, regional performance, trends')
    theme_choice = st.selectbox('Theme', ['light','dark'])
    density = st.selectbox('Visual density', ['Normal','Compact','Spacious'], index=0)
    st.markdown('---')
    existing = ['(none)'] + list(st.session_state.dashboards.keys())
    pick = st.selectbox('Open dashboard', existing)
    if pick != '(none)': st.session_state.current = pick
    st.text_input('Save current as', key='save_name')
    if st.button('💾 Save current'):
        name = st.session_state.get('save_name') or f'Dashboard_{int(time.time())}'
        if st.session_state.current and st.session_state.current in st.session_state.dashboards:
            st.session_state.dashboards[name] = st.session_state.dashboards[st.session_state.current].copy()
        else:
            st.session_state.dashboards[name] = {'spec':None,'html':''}
        st.success(f'Saved as {name}')
    if st.button('✨ Generate Dashboard (LLM)'):
        if st.session_state.df is None:
            st.error('Upload data first.'); st.stop()
        dataset_hash = hashlib.sha256(pd.util.hash_pandas_object(st.session_state.df, index=True).values.tobytes()).hexdigest()
        cache_key = f"{dataset_hash}:{goal}:{role}:{theme_choice}"
        if cache_key in st.session_state.get('cache_specs',{}):
            spec = st.session_state.cache_specs[cache_key]
        else:
            prompt = strong_llm_prompt(goal, list(st.session_state.df.columns), role)
            with st.spinner('🔧 Generating spec...'):
                try:
                    resp = client.chat.completions.create(model='gpt-4o-mini', messages=[{'role':'user','content':prompt}], temperature=0.25)
                    spec = clean_json(resp.choices[0].message.content)
                except Exception as e:
                    st.error(f'LLM error: {e}'); spec = {}
            if spec:
                st.session_state.cache_specs = st.session_state.get('cache_specs',{})
                st.session_state.cache_specs[cache_key] = spec
        if spec:
            name = f"Dashboard_{int(time.time())}"
            st.session_state.dashboards[name] = {'spec':spec,'html':''}
            st.session_state.current = name
            st.session_state.filters = {}
            st.success(f'Generated {name}')

# -----------------------------
# Load data
# -----------------------------
if uploaded:
    try:
        if uploaded.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded)
        else:
            xl = pd.read_excel(uploaded, sheet_name=None)
            df = xl[list(xl.keys())[0]] if isinstance(xl, dict) else xl
        st.session_state.df = df; st.success(f'Loaded {uploaded.name} ({len(df)} rows)')
    except Exception as e:
        st.error(f'Failed to read file: {e}')

if st.session_state.df is None:
    st.info('Upload a dataset to continue.'); st.stop()
df = st.session_state.df
# -----------------------------
# Validate dashboard spec
# -----------------------------
if not st.session_state.current or st.session_state.current not in st.session_state.dashboards:
    st.info('Generate or open a dashboard.'); st.stop()
cur = st.session_state.dashboards[st.session_state.current]
spec = cur.get('spec')
if not spec:
    st.info('No spec. Generate with LLM.'); st.stop()
# -----------------------------
# Filters
# -----------------------------
with st.expander('🎛 Filters (live)', expanded=True):
    new_filters = {}
    for f in spec.get('filters',[]):
        field = f.get('field')
        if field and field in df.columns:
            vals = sorted(df[field].dropna().astype(str).unique().tolist())
            sel = st.multiselect(field, vals[:1000], default=st.session_state.filters.get(field,[]), key=f'flt_{field}')
            new_filters[field] = sel
    st.session_state.filters = new_filters

def apply_filters(df_local, filters_local):
    out = df_local
    for c, vals in (filters_local or {}).items():
        if c in out.columns and vals:
            out = out[out[c].astype(str).isin(vals)]
    return out

df_f = apply_filters(df, st.session_state.filters)
# -----------------------------
# Apply Smart Layout 3.0 + Auto Layout
# -----------------------------
with st.spinner('🔁 Optimizing layout (Smart Layout 3.0)...'):
    spec = smart_layout_v3(spec, df_f, client=client, add_anomaly_charts=True)
    spec = auto_layout_optimize(spec)
    st.session_state.dashboards[st.session_state.current]['spec'] = spec
# -----------------------------
# Build visual HTML with progress and explainability data
# -----------------------------
sections = spec.get('sections',[])
kpis = spec.get('kpis',[])
total_items = max(1, len(kpis) + sum(len(s.get('charts',[])) for s in sections))
progress = st.progress(0)
step = 0
kpi_html = ''
explainability = []
for k in kpis:
    step += 1; progress.progress(step/total_items)
    title = safe_str(k.get('title')); expr = k.get('expr','COUNT(*)'); fmt = k.get('format','auto')
    val = kpi_value(df_f, expr)
    if fmt == 'auto':
        t = title.lower()
        if any(w in t for w in ('margin','rate','percent','%')): fmt = 'pct'
        elif any(w in t for w in ('revenue','sale','amount','profit','gmv')): fmt = 'currency'
        else: fmt = 'decimal'
    kpi_html += f"<div class='kpi-card'><div class='kpi-title'>{title}</div><div class='kpi-value'>{fmt_value(val,fmt)}</div></div>"
sections_html = ''
for sec in sections:
    sec_name = safe_str(sec.get('name','Section'))
    viz_block = ''
    for ch in sec.get('charts',[]):
        step += 1; progress.progress(min(1.0, step/total_items))
        ctype = safe_str(ch.get('type','bar')).lower(); x = ch.get('x'); y = ch.get('y')
        metrics = _compute_insight_metrics(df_f, x, y if isinstance(y,str) else (y[0] if isinstance(y,list) and y else None))
        recommended = _recommend_visual(df_f, x, y)
        explainability.append({'section':sec_name,'x':x,'y':y,'type':ctype,'recommendation':recommended,'metrics':metrics})
        if ctype == 'table':
            cols = ch.get('columns') or [c for c in ([x] + ([y] if isinstance(y,str) else (y or []))) if c in df.columns]
            html_piece = df_f[cols].head(40).to_html(index=False)
        else:
            html_piece = make_plot_html(df_f, x or '', y or '', ctype, theme=spec.get('theme','light'))
        narrative = sec.get('narrative','') or ''
        viz_block += f"<div class='viz'><div class='viz-head'><div class='viz-title'>{safe_str(x)} vs {safe_str(y)}</div><div class='viz-type'>{safe_str(ctype)}</div></div>{html_piece}<div class='insight'>🧠 {safe_str(narrative)}</div></div>"
    if viz_block: sections_html += f"<section><h3>{sec_name}</h3><div class='viz-grid'>{viz_block}</div></section>"
progress.empty()
# -----------------------------
# HTML Template (Power BI style)
# -----------------------------
TEMPLATE = Template('''<!doctype html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><title>{{ title }}</title><style>:root{--pb-bg:#fff;--pb-panel:#f7f8fa;--pb-border:#d0d3d9;--pb-ink:#242424;--pb-muted:#606060;--pb-radius:10px;--pb-shadow:0 2px 10px rgba(2,6,23,0.06);}body{background:var(--pb-panel);font-family:Segoe UI,Arial,sans-serif;margin:0;color:var(--pb-ink)}.wrapper{max-width:1480px;margin:auto;padding:18px}.dashboard{background:var(--pb-bg);border:1px solid var(--pb-border);border-radius:var(--pb-radius);box-shadow:var(--pb-shadow);padding:18px;aspect-ratio:16/9;display:flex;flex-direction:column;gap:18px;overflow:hidden}.header{display:flex;justify-content:space-between;align-items:center}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}.kpi-card{background:white;border-radius:8px;border:1px solid var(--pb-border);padding:12px;min-height:64px;display:flex;flex-direction:column;justify-content:center;box-shadow:0 1px 4px rgba(0,0,0,0.03)}.kpi-title{font-size:12px;color:var(--pb-muted)}.kpi-value{font-size:22px;font-weight:700;margin-top:6px}.viz-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px;align-items:start}.viz{background:white;padding:12px;border-radius:8px;border:1px solid var(--pb-border);min-height:420px;display:flex;flex-direction:column;box-shadow:0 2px 8px rgba(0,0,0,0.04);overflow:hidden}.viz-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}.viz-title{font-size:14px;color:var(--pb-ink)}.viz-type{font-size:12px;color:var(--pb-muted);background:var(--pb-panel);padding:4px 8px;border-radius:6px;border:1px solid var(--pb-border)}.insight{margin-top:12px;font-size:13px;color:var(--pb-muted)}.footer{text-align:right;font-size:12px;color:var(--pb-muted);margin-top:auto}</style></head><body><div class="wrapper"><div class="dashboard"><div class="header"><h2>{{ title }}</h2><div class="meta">Auto-BI • 16:9 • Theme: {{ theme }}</div></div><div class="kpis">{{ kpis|safe }}</div>{{ sections|safe }}<div class="footer">Generated by Auto-BI • Story: Overview → Performance → Trends → Deep Dive → Risks</div></div></div></body></html>''')

dashboard_html = TEMPLATE.render(title=safe_str(spec.get('title','Executive Dashboard')), theme=spec.get('theme','light'), kpis=kpi_html, sections=sections_html)
st.session_state.dashboards[st.session_state.current]['html'] = dashboard_html
# -----------------------------
# Render preview & explainability panel
# -----------------------------
st.subheader(f"📊 Preview — {safe_str(spec.get('title','Dashboard'))}")
with st.expander('▼ Show/Hide full dashboard (fit to page)', expanded=True):
    st.components.v1.html(dashboard_html, height=900, scrolling=True)
st.subheader('🧾 Explainability & Ranking')
st.markdown('This panel shows the computed insight score and why a chart was prioritized.')
for ex in explainability:
    st.markdown(f"**Section:** {safe_str(ex['section'])} — **Chart:** {safe_str(ex['x'])} vs {safe_str(ex['y'])}")
    st.write('Recommendation:', ex.get('recommendation'))
    st.write('Metrics score breakdown:')
    st.json(ex.get('metrics', {}))
    st.markdown('---')
# -----------------------------
# Slide mode (abbreviated)
# -----------------------------
st.subheader('🎞 Slide Mode')
sections_list = spec.get('sections', [])
if sections_list:
    st.session_state.slide_index = st.session_state.slide_index % len(sections_list)
    prev_col, next_col = st.columns([1,1])
    if prev_col.button('◀ Prev'): st.session_state.slide_index = max(0, st.session_state.slide_index-1)
    if next_col.button('Next ▶'): st.session_state.slide_index = min(len(sections_list)-1, st.session_state.slide_index+1)
    sec = sections_list[st.session_state.slide_index]
    st.markdown(f"### {safe_str(sec.get('name'))}")
    for ch in sec.get('charts', []):
        ctype = safe_str(ch.get('type','bar'))
        if ctype == 'table':
            cols = ch.get('columns') or df.columns[:6].tolist()
            st.dataframe(df_f[cols].head(100))
        else:
            x = ch.get('x'); y = ch.get('y')
            if x in df.columns and (isinstance(y,str) and y in df.columns):
                if ctype in ('bar','line','area'):
                    agg = df_f.groupby(x)[y].sum(numeric_only=True).reset_index()
                else:
                    agg = df_f
                if ctype == 'bar': fig=px.bar(agg, x=x, y=y)
                elif ctype == 'line': fig=px.line(agg, x=x, y=y)
                elif ctype == 'area': fig=px.area(agg, x=x, y=y)
                elif ctype == 'donut': fig=px.pie(agg, names=x, values=y, hole=0.45)
                elif ctype == 'scatter': fig=px.scatter(df_f, x=x, y=y)
                else: fig=px.bar(agg, x=x, y=y)
                st.plotly_chart(fig, use_container_width=True)
# -----------------------------
# Power-user spec editor
# -----------------------------
with st.expander('⚙️ Power-user: View / Edit Spec JSON', expanded=False):
    raw = json.dumps(spec, indent=2)
    edited = st.text_area('Spec JSON', value=raw, height=320)
    if st.button('Apply Spec'):
        try:
            new_spec = json.loads(edited)
            st.session_state.dashboards[st.session_state.current]['spec'] = new_spec
            st.experimental_rerun()
        except Exception as e:
            st.error('Invalid JSON: ' + safe_str(e))

st.caption('Tip: Use the Explainability panel to understand why charts are prioritized.')
