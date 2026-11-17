"""v3: re-derive every RICCE-Bench ordering number, the ordering
heatmap's RICCE column, the Fig. 1 teaser scores and the root-only
example under the FINAL evaluator (event_v2: dual assignment, root-excluded
ancestors, not-scorable handling, hard gate).

Uses the same pure RICCE-Bench items, the same pre-parsed candidate/source
events and the same constraint builder / cluster bootstrap as the canonical
pipeline (scripts/run_ordering_eval.py, scripts/run_semantic_validity_ordering.py),
so only the scoring formulation differs. As a basis check it also recomputes
the historical evaluator on the same events and asserts it reproduces the
canonical per-item CE-F file exactly.

Outputs -> results_v3/ordering/ and results_v3/teaser/.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.schema import Event
from ricce.benchmarks.env import load_taxes, load_weights
from ricce.evaluation.ordering_report import load_items
from ricce.evaluation.ordering import build_constraints, pairwise_ordering_accuracy
from ricce.metrics.event import ricce_scores
from ricce.metrics.event_v2 import ricce_scores_v2
from ricce.metrics.final import FINAL_CONFIG

OUT = ROOT / 'results_v3' / 'ordering'; OUT.mkdir(parents=True, exist_ok=True)
TOUT = ROOT / 'results_v3' / 'teaser'; TOUT.mkdir(parents=True, exist_ok=True)
ITEMS = ROOT / 'artifacts_cvpr_final' / 'riccebench' / 'items.jsonl'
TAXES, W = load_taxes(), load_weights()
FINAL = FINAL_CONFIG
E = lambda L: [Event(**{k: v for k, v in d.items() if k in Event.__dataclass_fields__}) for d in L]

# ------------------------------------------------------------------ per-item scores
items = load_items(ITEMS)
rows = []
for r in items.itertuples():
    c, g = E(r.candidate_event), E(r.source_event)
    v1 = ricce_scores(c, g, TAXES, W)
    v2 = ricce_scores_v2(c, g, TAXES, W, **FINAL)
    ns = v2['not_scorable']
    rows.append(dict(sample_id=r.sample_id, v1_F=v1['ce_f1'],
                     P=np.nan if ns else v2['ce_precision'], R=np.nan if ns else v2['ce_recall'],
                     F=np.nan if ns else v2['ce_f1']))
S = pd.DataFrame(rows).set_index('sample_id')
S.to_csv(OUT / 'per_item_final.csv')

# basis check: historical evaluator on these events == canonical per-item file
canon = pd.read_csv(ROOT / 'results_cvpr_final/ordering/scores/ricce.csv')
canon = canon[canon.metric == 'CE-F'].set_index('sample_id')['score']
common = S.index.intersection(canon.index)
maxdiff = float(np.abs(S.loc[common, 'v1_F'] - canon.loc[common]).max())
assert len(common) == len(S) and maxdiff < 1e-9, (len(common), len(S), maxdiff)
print('basis check ok: historical evaluator reproduces canonical per-item CE-F, max|diff|=', maxdiff, flush=True)

# ------------------------------------------------------------------ ordering suites
cons = build_constraints(items)
res = {'n_items': int(len(S)), 'n_not_scorable': int(S.F.isna().sum()), 'basis_check_maxdiff': maxdiff}
suites = {
    'core': cons[cons.suite == 'core'],
    'ancestor_chain': cons[cons.group.isin(['P0_paraphrase>P1_near_ancestor', 'P1_near_ancestor>P2_distant_ancestor'])],
    'near_gt_sibling': cons[cons.suite == 'extended'],
    'invariance': cons[cons.suite == 'invariance'],
}
for col in ('F', 'P', 'R'):
    for name, c in suites.items():
        o = pairwise_ordering_accuracy(S[col], c, n_boot=2000, seed=0)
        res[f'{col}|{name}'] = dict(n=o['n_constraints'], dropped_nan=o['n_dropped_nan'], acc=o['accuracy_strict'],
                                    lo=o['accuracy_strict_ci95'][0], hi=o['accuracy_strict_ci95'][1], tie_rate=o['tie_rate'])

# semantic-validity suite, identical construction to run_semantic_validity_ordering.py
GOOD = ['P0_paraphrase', 'P1_near_ancestor']
BAD = ['P4_action_reversal', 'P9_change_state_reversal', 'negation_flip', 'source_target_swap']
it = items.set_index('sample_id')
for col in ('F', 'P', 'R'):
    d = pd.DataFrame({'source_id': it.loc[S.index, 'source_id'].values, 'ptype': it.loc[S.index, 'perturbation_type'].values, 'score': S[col].values})
    piv = d.pivot_table(index='source_id', columns='ptype', values='score')
    parts = []
    for gd in GOOD:
        for bd in BAD:
            x = piv[[gd, bd]].dropna()
            parts.append(pd.DataFrame({'source_id': x.index.values, 'bad': bd, 'win': (x[gd] > x[bd] + 1e-9).astype(float)}))
    r = pd.concat(parts, ignore_index=True)
    ps = r.groupby('source_id')['win'].agg(['sum', 'count'])
    rng = np.random.default_rng(0); idx = rng.integers(0, len(ps), (2000, len(ps)))
    boots = ps['sum'].values[idx].sum(1) / ps['count'].values[idx].sum(1)
    res[f'{col}|validity'] = dict(n=int(len(r)), acc=float(r.win.mean()), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)),
                                  **{f'vs_{bd}': float(r[r.bad == bd].win.mean()) for bd in BAD})

# per-family (heatmap rows)
fam = []
for g, c in cons.groupby('group'):
    o = pairwise_ordering_accuracy(S['F'], c, n_boot=200, seed=0)
    fam.append(dict(group=g, n=o['n_constraints'], acc=o['accuracy_strict']))
pd.DataFrame(fam).to_csv(OUT / 'per_family_final_cef.csv', index=False)
res['mean_cef_all_items'] = float(S.F.mean())
res['mean_cef_all_items_historical'] = float(S.v1_F.mean())
json.dump(res, open(OUT / 'summary.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k.startswith('F|') or not isinstance(v, dict)}, indent=1), flush=True)

# ------------------------------------------------------------------ teaser + root-only
from ricce.baselines.flatten import FlatContext
ctx = FlatContext()
tz = json.load(open(ROOT / 'results_cvpr_final/teaser/teaser_scores.json'))
ref = ctx.parse(tz['reference'])
trows = []
for x in tz['items']:
    ev = ctx.parse(x['caption'])
    s = ricce_scores_v2(ev, ref, TAXES, W, **FINAL)
    trows.append(dict(kind=x['kind'], caption=x['caption'], cep=s['ce_precision'], cer=s['ce_recall'], cef=s['ce_f1'],
                      cef_historical=x['cef'], bleu4=x['bleu4'], meteor=x['meteor'], bertscore=x['bertscore'],
                      events=[{k: getattr(e, k) for k in ('object', 'action', 'source', 'target', 'location', 'quantity', 'change_state')} for e in ev]))
gt = [Event(object='residential_building', action='constructed', location='upper_left', quantity='several', change_state='change')]
root = [Event(object='entity', action='temporal_state', location='image', quantity='quantity', change_state='change')]
root_nostate = [Event(object='entity', action='temporal_state', location='image', quantity='quantity')]
ro = dict(reference=str(gt[0]), root_only=str(root[0]),
          historical_cef=ricce_scores(root, gt, TAXES, W)['ce_f1'],
          final_cef=ricce_scores_v2(root, gt, TAXES, W, **FINAL)['ce_f1'],
          final_cef_without_change_state=ricce_scores_v2(root_nostate, gt, TAXES, W, **FINAL)['ce_f1'],
          historical_cef_without_change_state=ricce_scores(root_nostate, gt, TAXES, W)['ce_f1'])
json.dump(dict(reference=tz['reference'], items=trows, root_only=ro, evaluator='event_v2 FINAL', command='python scripts/v3_bench_ordering_teaser.py'),
          open(TOUT / 'teaser_final.json', 'w'), indent=1)
for t in trows:
    print(t['kind'], round(t['cep'], 3), round(t['cer'], 3), round(t['cef'], 3), '| hist', round(t['cef_historical'], 3))
print('root-only', ro)
print('DONE')
