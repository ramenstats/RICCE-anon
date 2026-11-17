"""v3: validate the contradiction gate by measuring the gate itself.

The previous version counted a "false contradiction" whenever a same-image
reference pair scored CE-F < 0.15, whether or not the gate acted; that proxy
also moved with every unrelated scoring change and made the soft-gate gamma
sweep an artifact of the 0.15 cut-off. Here a gate activation is measured
directly (ricce/metrics/final.py:gate_activations): an event pair aligned
by the ungated optimal assignments whose events contradict (change/no-change,
opposed action families, conversion with swapped source/target).

  1. hand-written stress sets (20 + 227 items, expected gated / not gated):
     detection recall, false activations, residual CE-F per gate strength;
  2. real same-image LEVIR-CC reference pairs (first two references of every
     image with >=2; all-pairs as a robustness check): activation rate by
     contradiction type, CE-F cost of the gate; the old CE-F<0.15 proxy is
     recomputed only to show how little of it involved the gate;
  3. RICCE-Bench: semantic-validity accuracy and residual credit of
     contradictory captions per gate strength;
  4. 8-system LEVIR-CC ranking per gate strength.
Outputs -> results_v3/gate/."""
import itertools, json, random, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import kendalltau
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.schema import Event
from ricce.utils import read_jsonl
from ricce.benchmarks.env import load_taxes, load_weights, make_parser
from ricce.evaluation.parse_store import ParseStore
from ricce.metrics.final import FINAL_CONFIG, gate_activations
from ricce.metrics.event_v2 import ricce_scores_v2

OUT = ROOT / 'results_v3' / 'gate'; OUT.mkdir(parents=True, exist_ok=True)
T, W = load_taxes(), load_weights()
GATES = [('hard', None)] + [('soft', g) for g in (0.05, 0.1, 0.2, 0.3, 0.5)] + [('none', None)]
gname = lambda m, g: m if g is None else f'soft{g}'


def F(p, g, mode, gamma):
    s = ricce_scores_v2(p, g, T, W, **{**FINAL_CONFIG, 'gate_mode': mode, 'soft_gamma': gamma or 0.1})
    return np.nan if s['not_scorable'] else s['ce_f1']


res = {}
# ------------------------------------------------------------------ 1 stress sets
parser = make_parser()
for name, fn in [('original20', 'original_20_stress.jsonl'), ('enlarged227', 'action_reversal_stress.jsonl')]:
    rows = []
    for it in read_jsonl(ROOT / 'artifacts_r1' / fn):
        p = parser.parse(it['pred'], allow_multi_event=True); g = parser.parse(it['gt'], allow_multi_event=True)
        act = gate_activations(p, g, T, W)
        r = dict(pred=it['pred'], gt=it['gt'], expected=it['expected'], family=it.get('family'), activated=bool(act),
                 types=';'.join(sorted({a[2] for a in act})))
        for m, gm in GATES:
            r[gname(m, gm)] = F(p, g, m, gm)
        rows.append(r)
    d = pd.DataFrame(rows); d.to_csv(OUT / f'stress_{name}.csv', index=False)
    pos, neg = d[d.expected == 'gated'], d[d.expected != 'gated']
    res[f'stress_{name}'] = dict(n=len(d), n_expected_gated=len(pos), n_expected_not=len(neg),
                                 detection_recall=float(pos.activated.mean()), false_activation_rate=float(neg.activated.mean()),
                                 missed=pos[~pos.activated][['pred', 'gt', 'family']].to_dict('records'),
                                 residual_cef_expected_gated={gname(m, gm): float(pos[gname(m, gm)].mean()) for m, gm in GATES},
                                 cef_expected_not_gated={gname(m, gm): float(neg[gname(m, gm)].mean()) for m, gm in GATES},
                                 recall_by_family=pos.groupby('family').activated.mean().to_dict())
    print(name, {k: v for k, v in res[f'stress_{name}'].items() if k != 'missed'}, flush=True)

# ------------------------------------------------------------------ 2 real same-image reference pairs
store = ParseStore(ROOT / 'results_v3' / 'parse_cache.json')
_c = {}
def ev(t):
    if t not in _c:
        _c[t] = store.events(t, 'multi')
    return _c[t]
man = list(read_jsonl(ROOT / 'artifacts' / 'manifests' / 'LEVIR_CC.jsonl'))
rows = []
for r in man:
    refs = [x for x in r.get('references', []) if x and x.strip()]
    if len(refs) < 2:
        continue
    for k, (a, b) in enumerate(itertools.combinations(range(len(refs)), 2)):
        pa, pb = ev(refs[a]), ev(refs[b])
        act = gate_activations(pa, pb, T, W)
        rows.append(dict(image_id=r['image_id'], split=r.get('split'), first_pair=(a, b) == (0, 1), ref_a=refs[a], ref_b=refs[b], activated=bool(act),
                         types=';'.join(sorted({x[2] for x in act})), cef_hard=F(pa, pb, 'hard', None), cef_none=F(pa, pb, 'none', None),
                         cef_previous=(lambda s: np.nan if s['not_scorable'] else s['ce_f1'])(ricce_scores_v2(pa, pb, T, W, gate_mode='hard', dual_matching=True, exclude_root=True))))
R = pd.DataFrame(rows); R.to_csv(OUT / 'reference_pairs.csv.gz', index=False)
real = {}
for label, d in (('first_pair', R[R.first_pair]), ('all_pairs', R)):
    act = d[d.activated]
    types = act.types.str.split(';').explode().value_counts().to_dict()
    proxy_prev = d.cef_previous < 0.15
    real[label] = dict(n_pairs=len(d), n_images=int(d.image_id.nunique()), activation_rate=float(d.activated.mean()), n_activated=int(d.activated.sum()),
                       activations_by_type=types, mean_cef_hard=float(d.cef_hard.mean()), mean_cef_none=float(d.cef_none.mean()),
                       mean_gate_cost_all=float((d.cef_none - d.cef_hard).mean()), mean_gate_cost_activated=float((act.cef_none - act.cef_hard).mean()),
                       previous_proxy_rate=float(proxy_prev.mean()), previous_proxy_n=int(proxy_prev.sum()),
                       previous_proxy_share_with_activation=float(d[proxy_prev].activated.mean()))
res['real_reference_pairs'] = real
print(json.dumps(real, indent=1), flush=True)
ex = R[R.first_pair & R.activated].sample(n=min(12, int((R.first_pair & R.activated).sum())), random_state=0)
ex[['image_id', 'ref_a', 'ref_b', 'types', 'cef_hard', 'cef_none']].to_csv(OUT / 'activated_reference_pair_examples.csv', index=False)

# ------------------------------------------------------------------ 3 RICCE-Bench per gate strength
items = pd.DataFrame([json.loads(l) for l in open(ROOT / 'artifacts_cvpr_final' / 'riccebench' / 'items.jsonl')])
E = lambda L: [Event(**{k: v for k, v in d.items() if k in Event.__dataclass_fields__}) for d in L]
GOOD = ['P0_paraphrase', 'P1_near_ancestor']
BAD = ['P4_action_reversal', 'P9_change_state_reversal', 'negation_flip', 'source_target_swap']
sub = items[items.perturbation_type.isin(GOOD + BAD)]
cand = [E(x) for x in sub.candidate_event]; src = [E(x) for x in sub.source_event]
bench = {}
for m, gm in GATES:
    f = np.array([F(c, s, m, gm) for c, s in zip(cand, src)])
    d = pd.DataFrame({'source_id': sub.source_id.values, 'ptype': sub.perturbation_type.values, 'F': f})
    piv = d.pivot_table(index='source_id', columns='ptype', values='F')
    w = n = 0; marg = []
    for gd in GOOD:
        for bd in BAD:
            x = piv[[gd, bd]].dropna(); w += int((x[gd] > x[bd] + 1e-9).sum()); n += len(x); marg += list((x[gd] - x[bd]).values)
    fam = d.groupby('ptype').F.mean()
    bench[gname(m, gm)] = dict(validity_acc=w / n, n=n, mean_margin=float(np.mean(marg)),
                               residual_cef_contradictory=float(d[d.ptype.isin(BAD)].F.mean()),
                               residual_by_family={b: float(fam[b]) for b in BAD}, mean_cef_correct=float(d[d.ptype.isin(GOOD)].F.mean()))
res['bench'] = bench
print(pd.DataFrame(bench).T[['validity_acc', 'mean_margin', 'residual_cef_contradictory', 'mean_cef_correct']].round(4).to_string(), flush=True)

# ------------------------------------------------------------------ 4 system ranking per gate strength
MAIN = ['qwen', 'sft_seed42', 'swmcc', 'chg2cap', 'rsiccformer', 'promptcc', 'rscama']
mtest = {r['image_id']: r for r in man if r.get('split') == 'test'}
sysm = {}
for sy in MAIN:
    preds = list(read_jsonl(ROOT / 'results' / 'baselines' / sy / 'LEVIR_CC_test.jsonl'))
    out = {}
    for m, gm in GATES:
        per = []
        for r in preds:
            pe = ev(str(r['prediction']))
            v = [F(pe, ev(g), m, gm) for g in mtest[r['image_id']]['references']]
            v = [x for x in v if not np.isnan(x)]
            if v:
                per.append(np.mean(v))
        out[gname(m, gm)] = float(np.mean(per))
    sysm[sy] = out
S = pd.DataFrame(sysm).T
res['system_cef_by_gate'] = S.to_dict()
res['system_tau_vs_hard'] = {c: float(kendalltau(S['hard'], S[c])[0]) for c in S.columns}
res['system_order_by_gate'] = {c: list(S[c].sort_values(ascending=False).index) for c in S.columns}
print(S.round(3).to_string(), '\n', res['system_tau_vs_hard'], flush=True)
json.dump(res, open(OUT / 'summary.json', 'w'), indent=1, default=float)
print('DONE')
