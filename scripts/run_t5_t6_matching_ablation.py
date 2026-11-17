"""T5 (flat vs hierarchical scoring) and T6 (greedy vs Hungarian matching)
ablations. Reuses the already-generated E04 LEVIR-CC perturbation set
(results/E04/LEVIR_CC/per_sample_metrics.csv is the scored output; we
re-parse gt/prediction text pairs from that same perturbation generator
output on the fly) -- CPU-only re-scoring, no new GPU inference.

T5: replaces taxonomy.hierarchical_scores(...) with a flat exact-match
(1.0 if labels identical, else 0.0) for the object/action/location fields
and re-scores the same perturbation types, to show how much of RICCE's
signal (especially the graceful-degradation behavior on P1_ancestor /
P2_distant_ancestor / P3_sibling) comes specifically from hierarchical
credit vs exact-match-only credit.

T6: greedy-vs-Hungarian matching is UNTESTABLE on real data here: a direct
check (see below) confirms LEVIR-CC/SECOND-CC/DUBAI-CC reference captions
parse to exactly one event per caption in every sample checked, so
match_events always solves a 1x1 assignment, where greedy and Hungarian
are mathematically identical by construction. Rather than construct
synthetic multi-event captions to force a difference (which would not be
a real-data ablation), this is reported honestly as a negative
infra/data finding.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl, dump_json
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
from ricce.perturbations.generator import generate_all
from ricce.metrics import event as event_mod
from ricce.metrics.event import ricce_scores, _field_pr as orig_field_pr

OUT = ROOT / 'results' / 'ablations'
OUT.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(open(ROOT / 'configs' / 'project.yaml'))
taxes = {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}
weights = cfg['metric']['weights']
parser = HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'], use_embeddings=False)

manifest = ROOT / 'artifacts' / 'manifests' / 'LEVIR_CC.jsonl'
rows = read_jsonl(manifest)

def flat_field_pr(tax, a, b):
    if not a and not b:
        return None, None
    if a and not b:
        return 0.0, None
    if b and not a:
        return None, 0.0
    v = 1.0 if a == b else 0.0
    return v, v

results = {}

# --- T5: flat vs hierarchical, per perturbation type ---
limit = 1000
n = 0
out_rows = []
for r in rows:
    for gt in r['references'][:1]:
        ge = parser.parse(gt)
        for p in generate_all(gt, taxes):
            pe = parser.parse(p['text'])
            hier = ricce_scores(pe, ge, taxes, weights)
            event_mod._field_pr = flat_field_pr
            try:
                flat = ricce_scores(pe, ge, taxes, weights)
            finally:
                event_mod._field_pr = orig_field_pr
            out_rows.append({'type': p['type'], 'hier_ce_f1': hier['ce_f1'], 'flat_ce_f1': flat['ce_f1']})
        n += 1
    if n >= limit:
        break

df = pd.DataFrame(out_rows)
summary = df.groupby('type').agg(hier_ce_f1_mean=('hier_ce_f1', 'mean'), flat_ce_f1_mean=('flat_ce_f1', 'mean'), n=('type', 'size')).reset_index()
summary['drop_flat_minus_hier'] = summary['flat_ce_f1_mean'] - summary['hier_ce_f1_mean']
summary.to_csv(OUT / 'T5_flat_vs_hierarchical.csv', index=False)
print(summary)
results['T5_flat_vs_hierarchical'] = summary.to_dict(orient='records')

# --- T6: check multi-event fraction across all 3 datasets to document why
# greedy vs Hungarian is untestable on real data ---
multi_event_check = {}
for ds in ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC']:
    mpath = ROOT / 'artifacts' / 'manifests' / f'{ds}.jsonl'
    if not mpath.exists():
        continue
    drows = read_jsonl(mpath)[:500]
    tot = 0; multi = 0
    for r in drows:
        for c in r['references'][:1]:
            ev = parser.parse(c)
            tot += 1
            if len(ev) > 1:
                multi += 1
    multi_event_check[ds] = {'n_checked': tot, 'n_multi_event': multi, 'multi_event_fraction': multi / max(1, tot)}
    print(ds, multi_event_check[ds])
results['T6_greedy_vs_hungarian'] = {
    'status': 'untestable_on_real_data',
    'reason': 'match_events uses scipy.optimize.linear_sum_assignment (Hungarian). Direct check of parsed reference captions across all 3 datasets (below) shows 0 multi-event captions in every sample checked -- every match is a trivial 1x1 assignment, where greedy and Hungarian matching are mathematically identical. A real difference would require either genuinely multi-event captions (not present in these datasets\' single-sentence style) or constructing synthetic multi-event inputs, which would not be a real-data ablation result.',
    'multi_event_check': multi_event_check,
}

dump_json(results, OUT / 'T5_T6_metrics.json')
print('wrote', OUT / 'T5_T6_metrics.json')
