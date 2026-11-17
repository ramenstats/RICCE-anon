"""Canonical system benchmark under the FROZEN final evaluator.

Every system's frozen test predictions are re-scored with the final
event.py (whole-event contradiction gates) under two parser modes:
  multi  : HybridParser.parse(..., allow_multi_event=True)   [headline config]
  single : HybridParser.parse(...)                           [sensitivity]
Mean over references (the reported multi-reference protocol). Conventional
metrics (BLEU-4/METEOR/ROUGE-L/CIDEr-D corpus) are read from the existing
E08 metrics.json (they do not depend on the RICCE parser).
Outputs: results_cvpr_final/system_benchmark/
  per_image/<DS>__<system>__<mode>.csv  (image_id, ce_p, ce_r, ce_f)
  summary.csv  (dataset, system, mode, n, ce_p, ce_r, ce_f, CI95, conventional)
  pairwise_<mode>.csv (paired bootstrap, Holm across all pairs per dataset)
Usage: python scripts/run_canonical_system_scores.py [--systems a,b,c] [--datasets ...]
"""
import argparse, json, os, sys, itertools
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
from ricce.metrics.event import ricce_scores

ap = argparse.ArgumentParser()
ap.add_argument('--systems', default='qwen,sft_seed42,chg2cap,swmcc')
ap.add_argument('--datasets', default='LEVIR_CC,SECOND_CC,DUBAI_CC')
ap.add_argument('--modes', default='multi,single')
ap.add_argument('--n-boot', type=int, default=10000)
ap.add_argument('--out', default=str(ROOT / 'results_cvpr_final' / 'system_benchmark'))
a = ap.parse_args()
OUT = Path(a.out); (OUT / 'per_image').mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(open(ROOT / 'configs' / 'project.yaml'))
taxes = {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}
W = cfg['metric']['weights']; p = cfg['parser']
parser = HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'],
                      use_embeddings=p.get('use_embeddings', True), embedding_model=p.get('embedding_model'),
                      similarity_threshold=float(p.get('similarity_threshold', 0.58)))
cache = {}
def parse(text, mode):
    k = (text, mode)
    if k not in cache:
        cache[k] = parser.parse(text, allow_multi_event=True) if mode == 'multi' else parser.parse(text)
    return cache[k]

def boot_ci(x, n, rng):
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

rng = np.random.default_rng(0)
rows = []
for ds in a.datasets.split(','):
    man = {r['image_id']: r for r in read_jsonl(ROOT / 'artifacts' / 'manifests' / f'{ds}.jsonl')}
    for mode in a.modes.split(','):
        per = {}
        for sy in a.systems.split(','):
            pf = ROOT / 'results' / 'baselines' / sy / f'{ds}_test.jsonl'
            if not pf.exists():
                print('missing predictions', ds, sy); continue
            preds = [r for r in read_jsonl(pf) if r['image_id'] in man]
            recs = []
            for r in preds:
                pe = parse(r['prediction'], mode)
                ces = [ricce_scores(pe, parse(g, mode), taxes, W) for g in man[r['image_id']]['references']]
                recs.append((r['image_id'], np.mean([c['ce_precision'] for c in ces]),
                             np.mean([c['ce_recall'] for c in ces]), np.mean([c['ce_f1'] for c in ces])))
            df = pd.DataFrame(recs, columns=['image_id', 'ce_p', 'ce_r', 'ce_f'])
            df.to_csv(OUT / 'per_image' / f'{ds}__{sy}__{mode}.csv', index=False)
            per[sy] = df.set_index('image_id')
            mj = ROOT / 'results' / 'E08' / ds / sy / 'metrics.json'
            conv = json.load(open(mj)) if mj.exists() else {}
            lo, hi = boot_ci(df.ce_f.values, a.n_boot, rng)
            rows.append(dict(dataset=ds, system=sy, mode=mode, n=len(df), ce_p=df.ce_p.mean(), ce_r=df.ce_r.mean(),
                             ce_f=df.ce_f.mean(), ce_f_lo=lo, ce_f_hi=hi,
                             bleu4=conv.get('bleu4'), meteor=conv.get('meteor'), rougeL=conv.get('rougeL'),
                             cider_d=conv.get('cider_d_corpus')))
            print(ds, mode, sy, 'CE-F %.4f' % df.ce_f.mean(), flush=True)
        # paired bootstrap on ce_f differences, Holm over all pairs
        pairs = []
        for s1, s2 in itertools.combinations(sorted(per), 2):
            j = per[s1].index.intersection(per[s2].index)
            d = (per[s1].loc[j, 'ce_f'] - per[s2].loc[j, 'ce_f']).values
            idx = rng.integers(0, len(d), (a.n_boot, len(d))); bm = d[idx].mean(1)
            lo, hi = np.percentile(bm, [2.5, 97.5])
            pv = max(2 * min((bm <= 0).mean(), (bm >= 0).mean()), 1.0 / (a.n_boot + 1))
            pairs.append(dict(dataset=ds, mode=mode, a=s1, b=s2, n=len(d), mean_diff=float(d.mean()),
                              ci_lo=float(lo), ci_hi=float(hi), p_raw=float(pv)))
        order = np.argsort([q['p_raw'] for q in pairs]); m = len(pairs); run = 0.0
        for rank, i in enumerate(order):
            run = max(run, min(1.0, (m - rank) * pairs[i]['p_raw'])); pairs[i]['p_holm'] = run
        if pairs:
            pd.DataFrame(pairs).to_csv(OUT / f'pairwise_{ds}_{mode}.csv', index=False)
pd.DataFrame(rows).to_csv(OUT / 'summary.csv', index=False)
json.dump(dict(command=' '.join(sys.argv), systems=a.systems, datasets=a.datasets, modes=a.modes,
               n_boot=a.n_boot, note='p_raw floor is 1/(n_boot+1); Holm over all system pairs within dataset+mode'),
          open(OUT / 'config.json', 'w'), indent=1)
print('done', OUT)
os._exit(0)
