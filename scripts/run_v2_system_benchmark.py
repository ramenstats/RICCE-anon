"""v2-final: system benchmark under the FINAL v2 evaluator.
Mirrors scripts/run_canonical_system_scores.py exactly (same manifests, same
predictions, same multi-event parse mode, same mean-over-references
protocol, same paired-bootstrap/Holm statistics) but scores with
ricce_scores_v2(dual_matching=True, exclude_root=True, gate_mode='hard')
instead of the frozen ricce_scores. Outputs results_v2/
system_benchmark_v2/ (summary.csv with ce_p/ce_r/ce_f, pairwise_*.csv)."""
import argparse, json, os, sys, itertools
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
from ricce.metrics.event_v2 import ricce_scores_v2

ap = argparse.ArgumentParser()
ap.add_argument('--systems', default='qwen,sft_seed42,chg2cap,swmcc,rsiccformer,promptcc,rscama')
ap.add_argument('--datasets', default='LEVIR_CC,SECOND_CC,DUBAI_CC')
ap.add_argument('--n-boot', type=int, default=10000)
ap.add_argument('--out', default=str(ROOT / 'results_v2' / 'system_benchmark_v2'))
a = ap.parse_args()
OUT = Path(a.out); (OUT / 'per_image').mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(open(ROOT / 'configs' / 'project.yaml'))
taxes = {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}
W = cfg['metric']['weights']; p = cfg['parser']
parser = HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'],
                      use_embeddings=p.get('use_embeddings', True), embedding_model=p.get('embedding_model'),
                      similarity_threshold=float(p.get('similarity_threshold', 0.58)))
FINAL_V2 = dict(gate_mode='hard', exclude_root=True, dual_matching=True)
cache = {}
def parse(text):
    if text not in cache:
        cache[text] = parser.parse(text, allow_multi_event=True)
    return cache[text]

def v2f(pe, ge):
    r = ricce_scores_v2(pe, ge, taxes, W, **FINAL_V2)
    if r['not_scorable']:
        return np.nan, np.nan, np.nan
    return r['ce_precision'], r['ce_recall'], r['ce_f1']

def boot_ci(x, n, rng):
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

rng = np.random.default_rng(0)
rows = []
for ds in a.datasets.split(','):
    man = {r['image_id']: r for r in read_jsonl(ROOT / 'artifacts' / 'manifests' / f'{ds}.jsonl')}
    per = {}
    for sy in a.systems.split(','):
        pf = ROOT / 'results' / 'baselines' / sy / f'{ds}_test.jsonl'
        if not pf.exists():
            print('missing predictions', ds, sy); continue
        preds = [r for r in read_jsonl(pf) if r['image_id'] in man]
        recs = []
        for r in preds:
            pe = parse(r['prediction'])
            vals = [v2f(pe, parse(g)) for g in man[r['image_id']]['references']]
            ps_ = [v[0] for v in vals if not np.isnan(v[0])]
            rs_ = [v[1] for v in vals if not np.isnan(v[1])]
            fs_ = [v[2] for v in vals if not np.isnan(v[2])]
            recs.append((r['image_id'], np.mean(ps_) if ps_ else np.nan, np.mean(rs_) if rs_ else np.nan, np.mean(fs_) if fs_ else np.nan))
        df = pd.DataFrame(recs, columns=['image_id', 'ce_p', 'ce_r', 'ce_f']).dropna()
        df.to_csv(OUT / 'per_image' / f'{ds}__{sy}.csv', index=False)
        per[sy] = df.set_index('image_id')
        mj = ROOT / 'results' / 'E08' / ds / sy / 'metrics.json'
        conv = json.load(open(mj)) if mj.exists() else {}
        lo, hi = boot_ci(df.ce_f.values, a.n_boot, rng)
        rows.append(dict(dataset=ds, system=sy, n=len(df), ce_p=df.ce_p.mean(), ce_r=df.ce_r.mean(),
                         ce_f=df.ce_f.mean(), ce_f_lo=lo, ce_f_hi=hi,
                         bleu4=conv.get('bleu4'), cider_d=conv.get('cider_d_corpus')))
        print(ds, sy, 'CE-F %.4f' % df.ce_f.mean(), flush=True)
    pairs = []
    for s1, s2 in itertools.combinations(sorted(per), 2):
        j = per[s1].index.intersection(per[s2].index)
        d = (per[s1].loc[j, 'ce_f'] - per[s2].loc[j, 'ce_f']).values
        idx = rng.integers(0, len(d), (a.n_boot, len(d))); bm = d[idx].mean(1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        pv = max(2 * min((bm <= 0).mean(), (bm >= 0).mean()), 1.0 / (a.n_boot + 1))
        pairs.append(dict(dataset=ds, a=s1, b=s2, n=len(d), mean_diff=float(d.mean()), ci_lo=float(lo), ci_hi=float(hi), p_raw=float(pv)))
    order = np.argsort([q['p_raw'] for q in pairs]); m = len(pairs); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pairs[i]['p_raw'])); pairs[i]['p_holm'] = run
    if pairs:
        pd.DataFrame(pairs).to_csv(OUT / f'pairwise_{ds}.csv', index=False)
pd.DataFrame(rows).to_csv(OUT / 'summary.csv', index=False)
print('done', OUT)
