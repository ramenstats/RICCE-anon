"""v2-final: recompute the flat-vs-event comparison under the FINAL v2
evaluator (dual matching, exclude_root=True, hard gate). Re-parses the raw
candidate/reference text in artifacts_cvpr_final/baselines/inputs/{diag_*,e04_*}.jsonl
with the same frozen HybridParser used everywhere else, rescoring ONLY the
'ricce' column with event_v2.ricce_scores_v2; every other metric's
column (kiritchenko/lca_f/hcm/taxaware) is copied unchanged from the
existing per-sample CSV, since those baselines do not depend on v1 vs v2.
Writes results_v2/baselines_v2/{diag_*.csv, e04_*.csv} plus rebuilt
diagnostics_summary_v2.csv and perturbation_summary_v2.csv (same aggregation
logic as scripts/analyze_baselines.py)."""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
from ricce.metrics.event_v2 import ricce_scores_v2
import yaml

cfg = yaml.safe_load(open(ROOT / 'configs/project.yaml'))
taxes = {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}
W = cfg['metric']['weights']
parser = HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'],
                       use_embeddings=True, embedding_model=cfg['parser'].get('embedding_model'),
                       similarity_threshold=float(cfg['parser'].get('similarity_threshold', 0.58)))
_cache = {}
def parse(t):
    if t not in _cache:
        _cache[t] = parser.parse(t, allow_multi_event=True)
    return _cache[t]

FINAL_V2 = dict(gate_mode='hard', exclude_root=True, dual_matching=True)

IN = ROOT / 'artifacts_cvpr_final' / 'baselines' / 'inputs'
OLD = ROOT / 'results_cvpr_final' / 'baselines' / 'per_sample'
OUT = ROOT / 'results_v2' / 'baselines_v2'; OUT.mkdir(parents=True, exist_ok=True)

def rescore_file(stem):
    inp = IN / f'{stem}.jsonl'
    old_csv = OLD / f'{stem}.csv'
    if not inp.exists() or not old_csv.exists():
        print('skip (missing)', stem); return None
    recs = [json.loads(l) for l in open(inp)]
    old = pd.read_csv(old_csv).set_index('id')
    rows = []
    for r in recs:
        ce, ge = parse(r['candidate']), parse(r['reference'])
        s = ricce_scores_v2(ce, ge, taxes, W, **FINAL_V2)
        p = 0.0 if s['not_scorable'] else s['ce_precision']
        rc = 0.0 if s['not_scorable'] else s['ce_recall']
        f = 0.0 if s['not_scorable'] else s['ce_f1']
        row = old.loc[r['id']].to_dict() if r['id'] in old.index else {}
        row.update(dict(id=r['id'], dataset=r['dataset'], group=r['group'],
                         ricce_P=p, ricce_R=rc, ricce_F=f))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f'{stem}.csv', index=False)
    print(stem, len(df), 'rows rescored', flush=True)
    return df

for stem in ['diag_bind_swap', 'diag_st_swap', 'diag_identity', 'diag_clause_order',
             'e04_LEVIR_CC', 'e04_SECOND_CC', 'e04_DUBAI_CC']:
    rescore_file(stem)

# ---------------- rebuild diagnostics_summary_v2.csv (same logic as analyze_baselines.diagnostics()) ----------------
METRICS = ['ricce', 'kiritchenko_micro', 'kiritchenko_macro', 'lca_f', 'hcm', 'taxaware_single', 'taxaware_axisflat']
NB = 2000
rng = np.random.default_rng(0)
def agg_fn(m, df):
    if m == 'kiritchenko_micro':
        I, NP, NR = (df[f'{m}_{k}'].to_numpy(float) for k in ('inter', 'np', 'nr'))
        def f(ix):
            i, p, r = I[ix].sum(), NP[ix].sum(), NR[ix].sum()
            P = i / p if p else 0.0; Rr = i / r if r else 0.0
            return 2 * P * Rr / (P + Rr) if P + Rr else 0.0
        return f
    v = df[f'{m}_F'].to_numpy(float)
    return lambda ix: v[ix].mean()

rows = []
for g in ['identity', 'clause_order', 'bind_swap', 'st_swap']:
    p = OUT / f'diag_{g}.csv'
    if not p.exists(): continue
    d = pd.read_csv(p)
    for m in METRICS:
        f = agg_fn(m, d); ix = np.arange(len(d))
        v = f(ix)
        bs = np.array([f(rng.integers(0, len(d), len(d))) for _ in range(NB)])
        rows.append(dict(group=g, metric=m, n=len(d), mean_F=v, ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                         frac_F_eq_1=float((d[f'{m}_F'] > 1 - 1e-9).mean())))
pd.DataFrame(rows).to_csv(OUT / 'diagnostics_summary_v2.csv', index=False)
print('wrote diagnostics_summary_v2.csv', flush=True)

# ---------------- rebuild perturbation_summary_v2.csv (drop vs P0_original, action reversal / change-state / omission / hallucination / distant ancestor) ----------------
def boot_ci(fa, fb, n):
    ds = np.empty(NB)
    for b in range(NB):
        ix = rng.integers(0, n, n)
        ds[b] = fa(ix) - fb(ix)
    return np.percentile(ds, [2.5, 97.5])

e = pd.concat([pd.read_csv(OUT / f'e04_{ds}.csv') for ds in ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC'] if (OUT / f'e04_{ds}.csv').exists()], ignore_index=True)
e['image_id'] = e['id'].str.split('|').str[0]
out = []
for ds in ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC', 'ALL']:
    d = e if ds == 'ALL' else e[e.dataset == ds]
    d = d.assign(key=d.dataset + '/' + d.image_id)
    base = d[d.group == 'P0_original'].drop_duplicates('key').set_index('key')
    for t in sorted(d.group.unique()):
        sub = d[d.group == t].drop_duplicates('key')
        sub = sub[sub.key.isin(base.index)].set_index('key')
        b = base.loc[sub.index].reset_index(); s = sub.reset_index()
        n = len(s)
        if n == 0:
            continue
        for m in METRICS:
            fs, fb = agg_fn(m, s), agg_fn(m, b)
            ix = np.arange(n)
            mean_t, mean_0 = fs(ix), fb(ix)
            lo, hi = boot_ci(fb, fs, n)
            out.append(dict(dataset=ds, type=t, metric=m, n=n, mean_type=mean_t, mean_P0=mean_0,
                            drop=mean_0 - mean_t, drop_ci_lo=lo, drop_ci_hi=hi))
pd.DataFrame(out).to_csv(OUT / 'perturbation_summary_v2.csv', index=False)
print('wrote perturbation_summary_v2.csv', flush=True)
print('DONE')
