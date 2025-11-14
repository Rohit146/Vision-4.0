
import pandas as pd, numpy as np, math, hashlib, json
from collections import defaultdict
from utils import safe_str

# Reuse insight metrics and functions (condensed)
def _pearson_corr(df, x, y):
    try:
        if x not in df.columns or y not in df.columns: return 0.0
        xr = pd.to_numeric(df[x], errors='coerce'); yr = pd.to_numeric(df[y], errors='coerce')
        mask = xr.notna() & yr.notna()
        if mask.sum() < 3: return 0.0
        return float(np.corrcoef(xr[mask], yr[mask])[0,1])
    except: return 0.0

def _entropy_of_series(s):
    try:
        counts = s.value_counts(dropna=True).astype(float); p = counts / counts.sum()
        ent = - (p * np.log2(p + 1e-12)).sum(); n = max(len(counts),1)
        return float(ent / math.log2(n + 1)) if n>1 else 0.0
    except: return 0.0

def _compute_insight_metrics(df, x, y):
    try:
        if x not in df.columns: return {'score':0.05}
        richness = min(df[x].nunique()/50.0, 1.0)
        v=0.0; meany=0.0
        if y in df.columns:
            numeric_y = pd.to_numeric(df[y], errors='coerce').dropna()
            v = float(numeric_y.var()) if not numeric_y.empty else 0.0
            meany = float(numeric_y.mean()) if not numeric_y.empty else 0.0
        var_score = min(v/(abs(meany)+1e-6), 2.0); var_score = float(np.tanh(var_score))
        slope = 0.0
        try:
            if pd.api.types.is_datetime64_any_dtype(df[x]) and y in df.columns:
                tmp = df[[x,y]].dropna().copy(); tmp[x]=pd.to_datetime(tmp[x], errors='coerce')
                tmp = tmp.set_index(x).resample('M').sum(numeric_only=True).reset_index()
                s = pd.to_numeric(tmp[y].dropna(), errors='coerce')
                if not s.empty:
                    xarr = np.arange(len(s)); A = np.vstack([xarr, np.ones(len(xarr))]).T
                    m,_ = np.linalg.lstsq(A, s.values, rcond=None)[0]; slope = float(m/(abs(s.mean())+1e-9))
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
        ch=item['chart']; x=item['x']; y=item['y']; sem=detect(x) or detect(y)
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
    priority=['Performance','Profitability','Geography','Customer','Time','DeepDive','Risk','Misc']
    final_sections=[]
    for p in priority:
        items = grouped.get(p) or grouped.get(p if p!='DeepDive' else 'DeepDive')
        if items:
            items_sorted = sorted(items, key=lambda it: (it['metrics'].get('score',0), it['metrics'].get('var_score',0)), reverse=True)
            charts = [it['chart'] for it in items_sorted]
            sect_name = p if p!='DeepDive' else 'Deep Dive'
            final_sections.append({'name':sect_name,'charts':charts})
    deep_idx=None
    for i,s in enumerate(final_sections):
        if s['name'].lower().startswith('deep'): deep_idx=i; break
    if deep_idx is not None and deep_idx != len(final_sections)-1:
        dd = final_sections.pop(deep_idx); final_sections.append(dd)
    if client is not None:
        for s in final_sections:
            sumry={'charts':[]}
            for ch in s.get('charts',[]): 
                x=ch.get('x'); y=ch.get('y'); t=ch.get('type')
                metrics=_compute_insight_metrics(df, x, y if isinstance(y,str) else (y[0] if isinstance(y,list) and y else None))
                sumry['charts'].append({'title':f'{x} vs {y}','type':t,'score':metrics.get('score')})
            # lightweight call - may fail silently
            try:
                prompt = f"Write a concise executive summary for {s.get('name')} using this summary: {sumry}"
                r = client.chat(completion_messages=[{'role':'user','content':prompt}])
                s['narrative'] = r.choices[0].message.content if hasattr(r,'choices') else ''
            except:
                s['narrative'] = ''
    spec['sections'] = final_sections
    return spec
