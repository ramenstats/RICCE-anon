"""v2-final: natural multi-event evaluation recomputed under the
FINAL v2 evaluator, mirroring scripts/run_natural_multievent_evaluation.py
exactly (same 759 real (prediction, naturally-multi-event-reference)
LEVIR-CC pairs, same 4 systems) but scoring RICCE with
ricce_scores_v2 instead of the frozen ricce_scores. The three flat
baselines (Kiritchenko, HCM, TA-axisflat) are unchanged by v1-vs-v2 (they do
not use event.py's matching at all) and are not recomputed."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl
from ricce.baselines.flatten import FlatContext
from ricce.metrics.event_v2 import ricce_scores_v2

SYSTEMS = ['sft_seed42', 'chg2cap', 'swmcc', 'qwen']
FINAL_V2 = dict(gate_mode='hard', exclude_root=True, dual_matching=True)

ctx = FlatContext()
man = list(read_jsonl(ROOT / 'artifacts/manifests/LEVIR_CC.jsonl'))

n_multi_ref = n_total_ref = 0
multievent_refs_by_image = {}
for r in man:
    mr = []
    for g in r.get('references', []):
        n_total_ref += 1
        events = ctx.parse(g)
        if len(events) >= 2:
            n_multi_ref += 1
            mr.append(g)
    if mr:
        multievent_refs_by_image[r['image_id']] = mr
print(f'{n_multi_ref} / {n_total_ref} multi-event refs, {len(multievent_refs_by_image)} images', flush=True)

rows = []
for sy in SYSTEMS:
    pf = ROOT / 'results/baselines' / sy / 'LEVIR_CC_test.jsonl'
    if not pf.exists():
        continue
    preds = {p['image_id']: p['prediction'] for p in read_jsonl(pf)}
    for iid, mrefs in multievent_refs_by_image.items():
        if iid not in preds:
            continue
        pred = preds[iid]
        pe = ctx.parse(pred)
        for g in mrefs:
            ge = ctx.parse(g)
            r_ = ricce_scores_v2(pe, ge, ctx.taxes, ctx.weights, **FINAL_V2)
            f = None if r_['not_scorable'] else r_['ce_f1']
            rows.append({'system': sy, 'image_id': iid, 'ricce_v2': f})
    print(sy, 'scored', sum(1 for r in rows if r['system'] == sy), flush=True)

df = pd.DataFrame(rows)
OUT = ROOT / 'results_v2'; OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / 'natural_multievent_v2_scores.csv', index=False)
summary = {'n_multi_event_refs': n_multi_ref, 'n_total_refs': n_total_ref, 'per_system': {}}
for sy, g in df.groupby('system'):
    summary['per_system'][sy] = {'mean': float(g.ricce_v2.dropna().mean()), 'n': int(g.ricce_v2.notna().sum())}
print(json.dumps(summary, indent=1), flush=True)
json.dump(summary, open(OUT / 'natural_multievent_v2_summary.json', 'w'), indent=1)
print('DONE')
