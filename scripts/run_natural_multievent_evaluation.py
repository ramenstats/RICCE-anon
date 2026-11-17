"""Addition (README sec. 8): evaluate RICCE
against the flat prior-work baselines specifically on NATURALLY occurring
multi-event reference captions (not RICCE-Bench's constructed
multi-event items), using real system predictions. A reference is
'multi-event' by the same definition used for the full dataset scan
: every clause independently resolves its own
directional action under the frozen parser. This is real data at real
scale (thousands of naturally occurring multi-event LEVIR-CC test
references), not a manually-verified sample -- we report exactly that
scope, not more."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl
from ricce.baselines.registry import BaselineContext, score_pair

SYSTEMS = ['sft_seed42', 'chg2cap', 'swmcc', 'qwen']
METRICS = ['ricce', 'kiritchenko_micro', 'hcm', 'taxaware_axisflat']

ctx = BaselineContext()
man = list(read_jsonl(ROOT / 'artifacts/manifests/LEVIR_CC.jsonl'))
man_by_id = {r['image_id']: r for r in man}

# identify naturally multi-event references (same definition as the full-scan dataset table)
n_multi_ref = 0
n_total_ref = 0
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
print(f'{n_multi_ref} / {n_total_ref} LEVIR-CC references are naturally multi-event '
      f'({len(multievent_refs_by_image)} images have >=1)', flush=True)

rows = []
for sy in SYSTEMS:
    pf = ROOT / 'results/baselines' / sy / 'LEVIR_CC_test.jsonl'
    if not pf.exists():
        print('missing', sy); continue
    preds = {p['image_id']: p['prediction'] for p in read_jsonl(pf)}
    for iid, mrefs in multievent_refs_by_image.items():
        if iid not in preds:
            continue
        pred = preds[iid]
        for g in mrefs:
            rec = {'system': sy, 'image_id': iid}
            for m in METRICS:
                try:
                    rec[m] = score_pair(m, pred, g, ctx)['F']
                except Exception as e:
                    rec[m] = None
            rows.append(rec)
    print(sy, 'scored', sum(1 for r in rows if r['system'] == sy), '(pred, multi-event ref) pairs', flush=True)

df = pd.DataFrame(rows)
OUT = ROOT / 'results_metric_v2'; OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / 'natural_multievent_scores.csv', index=False)

summary = {'n_multi_event_refs': n_multi_ref, 'n_total_refs': n_total_ref,
           'n_images_with_multi_event_ref': len(multievent_refs_by_image),
           'per_system': {}}
for sy, g in df.groupby('system'):
    summary['per_system'][sy] = {m: {'mean': float(g[m].dropna().mean()), 'n': int(g[m].notna().sum())} for m in METRICS}
print(json.dumps(summary, indent=1), flush=True)
json.dump(summary, open(OUT / 'natural_multievent_summary.json', 'w'), indent=1)
print('DONE')
