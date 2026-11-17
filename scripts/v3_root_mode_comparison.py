"""v3: how should a label equal to its taxonomy root be scored?

Four treatments, everything else fixed at the final configuration:
  include     -- root counted in ancestor sets (historical behaviour)
  legacy      -- the previous final evaluator: root excluded on object/action/
                 source/target/location, quantity unchanged (inconsistent)
  exclude     -- root excluded on every axis
  unspecified -- a root-level label is an absent field on that side; other
                 nodes compared with the root excluded (Kiritchenko-style)

Selection is by a-priori properties, stated before results were computed, not
by benchmark score:
  R1 a root-only prediction against a specific reference scores CE-F < 0.5
     (the threshold already used in tests/test_metric_properties.py);
  R2 a root-level PREDICTED field costs recall only, never precision (the
     method's own rule: a field the reference has but the prediction does not
     specify costs recall only);
  R3 a root-level REFERENCE field costs precision only, never recall
     (symmetric rule);
  R4 a correct-but-coarse caption outranks a wrong-sibling caption
     (RICCE-Bench near>sibling suite, strict).
Reported alongside, not used for selection: core / validity / ancestor-chain
ordering, mean CE-F on all items, distant-ancestor drop, frequency of
root-level labels in real parses.
Outputs results_v3/root_mode/."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.schema import Event
from ricce.benchmarks.env import load_taxes, load_weights
from ricce.evaluation.ordering_report import load_items
from ricce.evaluation.ordering import build_constraints, pairwise_ordering_accuracy
from ricce.metrics.event_v2 import ricce_scores_v2
from ricce.metrics.final import FINAL_CONFIG

OUT = ROOT / 'results_v3' / 'root_mode'; OUT.mkdir(parents=True, exist_ok=True)
T, W = load_taxes(), load_weights()
MODES = ['include', 'legacy', 'exclude', 'unspecified']
E = lambda L: [Event(**{k: v for k, v in d.items() if k in Event.__dataclass_fields__}) for d in L]


def sc(p, g, mode):
    return ricce_scores_v2(p, g, T, W, **{**FINAL_CONFIG, 'root_mode': mode})


res = {m: {} for m in MODES}
# ---------------- R1 root-only (reference from results_cvpr_final/properties/FAILURES.md, F2)
ref = [Event(object='residential_building', action='constructed', target='residential_building', location='upper_left',
             quantity='several', change_state='change')]
root = [Event(object='entity', action='temporal_state', location='image', quantity='quantity', change_state='change')]
for m in MODES:
    s = sc(root, ref, m)
    res[m]['R1_root_only'] = dict(cep=s['ce_precision'], cer=s['ce_recall'], cef=s['ce_f1'], pass_=bool(s['ce_f1'] < 0.5))

# ---------------- R2 / R3 on every axis, against the same specific event
base = Event(object='residential_building', action='constructed', target='residential_building', location='upper_left',
             quantity='several', change_state='change')
roots = {'object': 'entity', 'location': 'image', 'quantity': 'quantity', 'action': 'temporal_state'}
for m in MODES:
    r2, r3 = {}, {}
    for ax, rt in roots.items():
        vague = Event(**{**base.to_dict(), ax: rt}) if hasattr(base, 'to_dict') else None
        d = {k: getattr(base, k) for k in ('object', 'action', 'source', 'target', 'location', 'quantity', 'change_state')}
        vague = Event(**{**d, ax: rt}); absent = Event(**{**d, ax: None})
        # R2: vague prediction vs specific reference -> precision must equal the absent-field prediction's precision
        a, b = sc([vague], [base], m), sc([absent], [base], m)
        r2[ax] = dict(cep_vague=a['ce_precision'], cep_absent=b['ce_precision'], cer_vague=a['ce_recall'], cer_absent=b['ce_recall'],
                      pass_=bool(abs(a['ce_precision'] - b['ce_precision']) < 1e-9))
        # R3: specific prediction vs vague reference -> recall must equal recall against the absent-field reference
        a, b = sc([base], [vague], m), sc([base], [absent], m)
        r3[ax] = dict(cer_vs_vague_ref=a['ce_recall'], cer_vs_absent_ref=b['ce_recall'], cep_vs_vague_ref=a['ce_precision'],
                      cep_vs_absent_ref=b['ce_precision'], pass_=bool(abs(a['ce_recall'] - b['ce_recall']) < 1e-9))
    res[m]['R2_vague_prediction'] = r2
    res[m]['R3_vague_reference'] = r3
    res[m]['R2_pass'] = all(v['pass_'] for v in r2.values())
    res[m]['R3_pass'] = all(v['pass_'] for v in r3.values())

# ---------------- RICCE-Bench suites (pre-parsed events, as in the canonical pipeline)
items = load_items(ROOT / 'artifacts_cvpr_final' / 'riccebench' / 'items.jsonl')
cons = build_constraints(items)
suites = {'core': cons[cons.suite == 'core'], 'near_gt_sibling': cons[cons.suite == 'extended'],
          'ancestor_chain': cons[cons.group.isin(['P0_paraphrase>P1_near_ancestor', 'P1_near_ancestor>P2_distant_ancestor'])]}
GOOD = ['P0_paraphrase', 'P1_near_ancestor']
BAD = ['P4_action_reversal', 'P9_change_state_reversal', 'negation_flip', 'source_target_swap']
cand = [E(x) for x in items['candidate_event']]; srce = [E(x) for x in items['source_event']]
scores = {}
for m in MODES:
    F = []
    for c, g in zip(cand, srce):
        s = sc(c, g, m)
        F.append(np.nan if s['not_scorable'] else s['ce_f1'])
    F = pd.Series(F, index=items['sample_id'].values)
    scores[m] = F
    for name, c in suites.items():
        o = pairwise_ordering_accuracy(F, c, n_boot=1000, seed=0)
        res[m][f'bench_{name}'] = dict(n=o['n_constraints'], acc=o['accuracy_strict'], lo=o['accuracy_strict_ci95'][0],
                                       hi=o['accuracy_strict_ci95'][1], tie_rate=o['tie_rate'])
    d = pd.DataFrame({'source_id': items['source_id'].values, 'ptype': items['perturbation_type'].values, 'F': F.values})
    piv = d.pivot_table(index='source_id', columns='ptype', values='F')
    wins = n = 0
    for gd in GOOD:
        for bd in BAD:
            x = piv[[gd, bd]].dropna(); wins += int((x[gd] > x[bd] + 1e-9).sum()); n += len(x)
    res[m]['bench_validity'] = dict(n=n, acc=wins / n)
    fam = d.groupby('ptype').F.mean()
    res[m]['mean_cef_all'] = float(F.mean())
    res[m]['n_not_scorable'] = int(F.isna().sum())
    res[m]['drop_distant_ancestor'] = float(fam['P0_identity'] - fam['P2_distant_ancestor'])
    res[m]['drop_near_ancestor'] = float(fam['P0_identity'] - fam['P1_near_ancestor'])
    res[m]['R4_pass'] = bool(res[m]['bench_near_gt_sibling']['acc'] >= 0.99)
    print(m, {k: (round(v['acc'], 4) if isinstance(v, dict) and 'acc' in v else v) for k, v in res[m].items() if k.startswith('bench') or k.startswith('R') and k.endswith('pass')}, flush=True)

# ---------------- frequency of root-level labels in real parses (benchmark source captions)
cnt = {k: 0 for k in ('object', 'location', 'quantity', 'action', 'source', 'target')}
nev = 0
for evs in srce:
    for e in evs:
        nev += 1
        for k in cnt:
            v = getattr(e, k)
            tx = T['object'] if k in ('source', 'target') else T[k]
            if v is not None and v == tx.root:
                cnt[k] += 1
res['root_label_frequency_in_source_events'] = dict(n_events=nev, **cnt)
pd.DataFrame(scores).to_csv(OUT / 'bench_cef_by_root_mode.csv')
json.dump(res, open(OUT / 'summary_bench_properties.json', 'w'), indent=1, default=float)
print(json.dumps({m: {k: res[m][k] for k in ('R1_root_only', 'R2_pass', 'R3_pass', 'R4_pass', 'mean_cef_all', 'drop_distant_ancestor')} for m in MODES}, indent=1, default=float))
print('freq', res['root_label_frequency_in_source_events'])
print('DONE')
