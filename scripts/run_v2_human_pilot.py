"""v2-final: human-pilot correlation recomputed under the
FINAL v2 evaluator. Identical to scripts/run_human_pilot_canonical.py in
every respect (same 300 items, same real human ratings in
results/E07_human_pred_validation/per_item.json, same statistics) except
CE-F is computed with ricce_scores_v2 (dual matching, exclude_root=True,
hard gate) instead of the frozen ricce_scores."""
import json, sys, os
from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
from ricce.metrics.event_v2 import ricce_scores_v2
cfg = yaml.safe_load(open(ROOT / 'configs/project.yaml'))
taxes = {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}; W = cfg['metric']['weights']; p = cfg['parser']
P = HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'], use_embeddings=True,
                 embedding_model=p.get('embedding_model'), similarity_threshold=float(p.get('similarity_threshold', 0.58)))
FINAL_V2 = dict(gate_mode='hard', exclude_root=True, dual_matching=True)
items = json.load(open(ROOT / 'results/E07_human_pred_validation/per_item.json'))
man = {ds: {r['image_id']: r for r in read_jsonl(ROOT / f'artifacts/manifests/{ds}.jsonl')} for ds in ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC']}
strip = {ds: {k.lstrip('0') or '0': k for k in m} for ds, m in man.items()}
preds = {}
def pred_text(ds, sy, iid):
    k = (ds, sy)
    if k not in preds:
        f = ROOT / 'results/baselines' / sy / f'{ds}_test.jsonl'
        preds[k] = {r['image_id']: r['prediction'] for r in read_jsonl(f)} if f.exists() else {}
    return preds[k].get(iid)
rows = []
for it in items:
    ds, sy = it['dataset'], it['system']; iid = it['image_id']; iid = iid if iid in man[ds] else strip[ds].get(iid, iid)
    text = pred_text(ds, sy, iid)
    if text is None or iid not in man[ds]: continue
    pe = P.parse(text, allow_multi_event=True)
    ces = []
    for g in man[ds][iid]['references']:
        r_ = ricce_scores_v2(pe, P.parse(g, allow_multi_event=True), taxes, W, **FINAL_V2)
        if not r_['not_scorable']:
            ces.append(r_['ce_f1'])
    rows.append(dict(dataset=ds, system=sy, image_id=iid, human=it['human_overall'], ce_f=float(np.mean(ces)) if ces else np.nan,
                     bleu4=it['bleu4'], rouge_l=it['rouge_l'], meteor=it['meteor'], bertscore=it['bertscore']))
d = pd.DataFrame(rows).dropna(subset=['ce_f'])
out = ROOT / 'results_v2/human_pilot_v2'; out.mkdir(parents=True, exist_ok=True); d.to_csv(out / 'per_item.csv', index=False)
rng = np.random.default_rng(0); res = []
for m in ['ce_f', 'bleu4', 'rouge_l', 'meteor', 'bertscore']:
    x = d[[m, 'human']].dropna(); rho, pv = stats.spearmanr(x[m], x.human); tau, pt = stats.kendalltau(x[m], x.human)
    idx = rng.integers(0, len(x), (5000, len(x))); bs = [stats.spearmanr(x[m].values[i], x.human.values[i])[0] for i in idx[:2000]]
    res.append(dict(metric=m, n=len(x), spearman=rho, p_raw=pv, ci_lo=np.nanpercentile(bs, 2.5), ci_hi=np.nanpercentile(bs, 97.5), kendall=tau, p_kendall=pt))
r = pd.DataFrame(res); o = np.argsort(r.p_raw.values); m = len(r); run = 0
holm = np.zeros(m)
for rank, i in enumerate(o): run = max(run, min(1, (m - rank) * r.p_raw.values[i])); holm[i] = run
r['p_holm'] = holm
bh = np.zeros(m); prev = 1
for rank, i in reversed(list(enumerate(o))): prev = min(prev, r.p_raw.values[i] * m / (rank + 1)); bh[i] = prev
r['p_bh'] = bh; r.to_csv(out / 'correlations.csv', index=False)
pd.set_option('display.width', 200); print(len(d), 'items'); print(r.round(4).to_string(index=False))
print('DONE')
