"""v2-final: cross-dataset parse-clean decomposition
recomputed from RAW predictions and references under the FINAL v2 evaluator
(multi-event parse, dual matching, exclude_root=True, hard gate), replacing
the previous version's reliance on results/E08/*/per_sample_metrics.csv
(an older experiment stage that predates the frozen multi-event evaluator
convention used everywhere else in this paper). OOV/style/vocabulary
statistics are parser-based, not scoring-based, and are unchanged between
v1 and v2; they are recomputed here anyway from the manifests directly, for
a self-contained, fully current artifact."""
import sys, json, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from ricce.utils import read_jsonl
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
from ricce.metrics.event_v2 import ricce_scores_v2
import yaml

OUT = ROOT / 'results_v2' / 'cross_dataset_v2'; OUT.mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load(open(ROOT / 'configs/project.yaml'))
taxes = {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}
W = cfg['metric']['weights']
parser = HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'],
                       use_embeddings=True, embedding_model=cfg['parser'].get('embedding_model'),
                       similarity_threshold=float(cfg['parser'].get('similarity_threshold', 0.58)))
FINAL_V2 = dict(gate_mode='hard', exclude_root=True, dual_matching=True)
DATASETS = ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC']
SYS = ['sft_seed42', 'chg2cap', 'swmcc', 'qwen']
STOP = set('a an the of in on at to and is are was were be been by with from for as its it this that there near some'.split())
norm = HybridParser.normalize
rng = np.random.default_rng(0)

pcache = {}
def parse_multi(t):
    if t not in pcache:
        pcache[t] = parser.parse(t, allow_multi_event=True)
    return pcache[t]

def stratum(ev, mode):
    if ev.change_state == 'no_change':
        return 'no_change'
    core = int(ev.object is not None) + int(ev.action is not None)
    if mode == 'core':
        return {2: 'full', 1: 'partial', 0: 'failed'}[core]
    n = core + int(ev.location is not None) + int(ev.quantity is not None)
    return 'full' if n == 4 else ('failed' if n == 0 else 'partial')

def event_stratum(events, mode):
    """A (possibly multi-event) parse is 'full' only if every event in it is full."""
    if not events:
        return 'failed'
    if all(e.change_state == 'no_change' for e in events):
        return 'no_change'
    strata = [stratum(e, mode) for e in events]
    if all(s in ('full', 'no_change') for s in strata):
        return 'full'
    if all(s == 'failed' for s in strata):
        return 'failed'
    return 'partial'

def v2f(pe, ge):
    r = ricce_scores_v2(pe, ge, taxes, W, **FINAL_V2)
    return np.nan if r['not_scorable'] else r['ce_f1']

manifests = {ds: [r for r in read_jsonl(ROOT / f'artifacts/manifests/{ds}.jsonl')] for ds in DATASETS}
tokz = lambda t: norm(t).split()
levir_train_vocab = collections.Counter(w for r in manifests['LEVIR_CC'] if r['split'] == 'train' for c in r['references'] for w in tokz(c))
alias_vocab = set(w for t in taxes.values() for _, ph in t.labels_and_aliases() for w in tokz(ph))

res = {'per_dataset': {}, 'score_shift': {}}
ref_parse_multi = {}
clean_ref = {}
ref_nc = {}
for ds in DATASETS:
    test = [r for r in manifests[ds] if r['split'] == 'test']
    refs = [(r['image_id'], c) for r in test for c in r['references']]
    evs_multi = {}
    for iid, c in refs:
        evs_multi.setdefault(iid, []).append(parse_multi(c))
    ref_parse_multi[ds] = evs_multi
    clean_ref[ds] = {iid: all(event_stratum(es_, 'core') in ('full', 'no_change') for es_ in es) for iid, es in evs_multi.items()}
    ref_nc[ds] = {iid: float(np.mean([all(e.change_state == 'no_change' for e in es_) for es_ in es])) for iid, es in evs_multi.items()}
    allev = [e for es in evs_multi.values() for es_ in es for e in es_]
    toks = [w for _, c in refs for w in tokz(c)]; types = set(toks); content = [w for w in toks if w not in STOP]
    d = {'n_test_images': len(test), 'n_reference_captions': len(refs),
         'vocab_oov_vs_LEVIR_train': {'content_token_rate': float(np.mean([w not in levir_train_vocab for w in content])), 'n_types': len(types)},
         'style': {'mean_caption_tokens': float(np.mean([len(tokz(c)) for _, c in refs])),
                   'no_change_ref_fraction': float(np.mean([e.change_state == 'no_change' for e in allev])),
                   'frac_images_all_refs_core_parsed': float(np.mean(list(clean_ref[ds].values())))}}
    res['per_dataset'][ds] = d
    print(ds, 'refs parsed (v2 recompute)', len(refs), flush=True)

rows = []
for ds in DATASETS:
    man = {r['image_id']: r for r in manifests[ds] if r['split'] == 'test'}
    for s in SYS:
        pf = ROOT / 'results/baselines' / s / f'{ds}_test.jsonl'
        if not pf.exists():
            continue
        preds = [r for r in read_jsonl(pf) if r['image_id'] in man]
        for r in preds:
            pe = parse_multi(r['prediction'])
            ge_list = ref_parse_multi[ds].get(r['image_id'], [])
            vals = [v2f(pe, ge) for ge in ge_list]
            vals = [v for v in vals if not np.isnan(v)]
            cef = float(np.mean(vals)) if vals else np.nan
            rows.append(dict(dataset=ds, system=s, image_id=r['image_id'], ce_f1=cef,
                             pred_core=event_stratum(pe, 'core'), ref_clean=clean_ref[ds].get(r['image_id'], False),
                             ref_nc_frac=ref_nc[ds].get(r['image_id'], np.nan)))
        print(ds, s, 'scored (v2)', flush=True)
df = pd.DataFrame(rows).dropna(subset=['ce_f1'])
df.to_csv(OUT / 'per_image_v2.csv', index=False)

def boot(x, B=1000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if not len(x):
        return None
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

for s in SYS:
    lv = df[(df.system == s) & (df.dataset == 'LEVIR_CC')]
    if not len(lv):
        continue
    lvc = lv[lv.ref_clean & (lv.pred_core.isin(['full', 'no_change']))]
    res['score_shift'][s] = {'LEVIR_CC': {'ce_f1': float(lv.ce_f1.mean()), 'ce_f1_CI95': boot(lv.ce_f1), 'n': len(lv),
                                          'ce_f1_parse_clean': float(lvc.ce_f1.mean()) if len(lvc) else None, 'n_parse_clean': len(lvc)}}
    for ds in ['SECOND_CC', 'DUBAI_CC']:
        g = df[(df.system == s) & (df.dataset == ds)]
        gc = g[g.ref_clean & (g.pred_core.isin(['full', 'no_change']))]
        res['score_shift'][s][ds] = {'ce_f1': float(g.ce_f1.mean()) if len(g) else None, 'ce_f1_CI95': boot(g.ce_f1), 'n': len(g),
                                      'ce_f1_parse_clean': float(gc.ce_f1.mean()) if len(gc) else None, 'n_parse_clean': len(gc),
                                      'pred_no_change_share': float((g.pred_core == 'no_change').mean()) if len(g) else None,
                                      'mean_ref_no_change_frac': float(g.ref_nc_frac.mean()) if len(g) else None}
    print(s, res['score_shift'][s], flush=True)

json.dump(res, open(OUT / 'summary_v2.json', 'w'), indent=1)
print('DONE')
