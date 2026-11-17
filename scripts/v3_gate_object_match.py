"""v3: should a change-state / action-family contradiction require the
two events to be about compatible objects?

Motivation (documented limitation of the previous version): the action-family
test compares only actions, so when an assignment aligns "trees are removed"
with "villas are built" the hard gate fires although the two events describe
different objects, not opposite claims about one object.

Decision rule, fixed before results were computed. Adopt gate_object_match
only if ALL hold:
  C1 detection recall on every expected-contradiction stress item stays 1.0
     (20-item and 227-item sets);
  C2 no activation on stress items without any reversal (and, for the
     compound items with one reversed event, activation on that event is kept);
  C3 activation rate on real same-image LEVIR-CC reference pairs does not
     increase;
  C4 RICCE-Bench semantic-validity accuracy stays 1.0;
and it lowers C3 (otherwise there is nothing to gain).
Outputs -> results_v3/gate_object_match/summary.json."""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.schema import Event
from ricce.utils import read_jsonl
from ricce.benchmarks.env import load_taxes, load_weights, make_parser
from ricce.evaluation.parse_store import ParseStore
from ricce.metrics.final import gate_activations
from ricce.metrics.event_v2 import ricce_scores_v2

OUT = ROOT / 'results_v3' / 'gate_object_match'; OUT.mkdir(parents=True, exist_ok=True)
T, W = load_taxes(), load_weights()
CFG = dict(gate_mode='hard', dual_matching=True, root_mode='unspecified')
parser = make_parser()
res = {}
for om in (False, True):
    r = {}
    for name, fn in [('original20', 'original_20_stress.jsonl'), ('enlarged227', 'action_reversal_stress.jsonl')]:
        rows = []
        for it in read_jsonl(ROOT / 'artifacts_r1' / fn):
            p = parser.parse(it['pred'], allow_multi_event=True); g = parser.parse(it['gt'], allow_multi_event=True)
            act = gate_activations(p, g, T, W, object_match=om)
            s = ricce_scores_v2(p, g, T, W, **CFG, gate_object_match=om)
            rows.append(dict(expected=it['expected'], family=it.get('family'), activated=bool(act), n_act=len(act), cef=s['ce_f1']))
        d = pd.DataFrame(rows)
        pos = d[d.expected == 'gated']; comp = d[(d.expected != 'gated') & (d.family == 'compound')]; clean = d[(d.expected != 'gated') & (d.family != 'compound')]
        r[name] = dict(recall=float(pos.activated.mean()), n_pos=len(pos), clean_activation=float(clean.activated.mean()) if len(clean) else None,
                       n_clean=len(clean), compound_activation=float(comp.activated.mean()) if len(comp) else None, n_compound=len(comp),
                       mean_cef_pos=float(pos.cef.mean()), mean_cef_compound=float(comp.cef.mean()) if len(comp) else None)
    store = ParseStore(ROOT / 'results_v3' / 'parse_cache.json')
    n = a = 0
    for m in read_jsonl(ROOT / 'artifacts' / 'manifests' / 'LEVIR_CC.jsonl'):
        refs = [x for x in m.get('references', []) if x and x.strip()]
        if len(refs) >= 2:
            n += 1; a += bool(gate_activations(store.events(refs[0], 'multi'), store.events(refs[1], 'multi'), T, W, object_match=om))
    r['real_refpair'] = dict(n=n, activated=a, rate=a / n)
    items = [json.loads(l) for l in open(ROOT / 'artifacts_cvpr_final' / 'riccebench' / 'items.jsonl')]
    E = lambda L: [Event(**{k: v for k, v in x.items() if k in Event.__dataclass_fields__}) for x in L]
    GOOD = ['P0_paraphrase', 'P1_near_ancestor']; BAD = ['P4_action_reversal', 'P9_change_state_reversal', 'negation_flip', 'source_target_swap']
    rows = [dict(source_id=x['source_id'], ptype=x['perturbation_type'],
                 F=(lambda s: np.nan if s['not_scorable'] else s['ce_f1'])(ricce_scores_v2(E(x['candidate_event']), E(x['source_event']), T, W, **CFG, gate_object_match=om)))
            for x in items if x['perturbation_type'] in GOOD + BAD]
    piv = pd.DataFrame(rows).pivot_table(index='source_id', columns='ptype', values='F'); w = k = 0
    for gd in GOOD:
        for bd in BAD:
            x = piv[[gd, bd]].dropna(); w += int((x[gd] > x[bd] + 1e-9).sum()); k += len(x)
    r['bench_validity'] = dict(acc=w / k, n=k)
    res[f'object_match={om}'] = r
    print(om, json.dumps(r, indent=0), flush=True)
b, t = res['object_match=False'], res['object_match=True']
crit = dict(C1=all(t[s]['recall'] == 1.0 for s in ('original20', 'enlarged227')),
            C2=all((t[s]['clean_activation'] or 0) == 0 for s in ('original20', 'enlarged227')) and (t['enlarged227']['compound_activation'] == b['enlarged227']['compound_activation']),
            C3=t['real_refpair']['rate'] <= b['real_refpair']['rate'], C4=t['bench_validity']['acc'] == 1.0,
            lowers_C3=t['real_refpair']['rate'] < b['real_refpair']['rate'])
res['criteria'] = crit; res['adopt'] = all(crit.values())
json.dump(res, open(OUT / 'summary.json', 'w'), indent=1)
print(crit, 'ADOPT' if res['adopt'] else 'KEEP CURRENT GATE')
