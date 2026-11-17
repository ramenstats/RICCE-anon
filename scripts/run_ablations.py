"""Ontology-component, contradiction-gate, weight-sensitivity, and
reference-aggregation ablations (internal design review, Sec. 7.6
G2/G4/G5/G6). Reuses the already-frozen Qwen2.5-VL zero-shot predictions on
LEVIR-CC test (results/baselines/qwen/LEVIR_CC_test.jsonl) -- these are
pure re-scoring passes over text already on disk, no new VLM inference and
no GPU required, so they can run alongside GPU-bound training jobs.

Every number here comes from actually re-running ricce_scores with a
modified config against the same frozen predictions; nothing is estimated.
"""
import copy, itertools, json, random
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl, dump_json
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
from ricce.metrics import event as event_mod
from ricce.metrics.event import ricce_scores

OUT = ROOT / 'results' / 'ablations'
OUT.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(open(ROOT / 'configs' / 'project.yaml'))
taxes = {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}
base_weights = cfg['metric']['weights']
parser = HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'], use_embeddings=False)

manifest = {r['image_id']: r for r in read_jsonl(ROOT / 'artifacts' / 'manifests' / 'LEVIR_CC.jsonl')}
preds = [r for r in read_jsonl(ROOT / 'results' / 'baselines' / 'qwen' / 'LEVIR_CC_test.jsonl') if r['image_id'] in manifest]
print(f'{len(preds)} frozen LEVIR-CC test predictions loaded for ablation re-scoring')

# Pre-parse once (parsing is the expensive step, scoring against many
# weight configs afterward is cheap) and cache per-reference-set parses.
parsed_preds = {}
parsed_refs = {}
for r in preds:
    iid = r['image_id']
    parsed_preds[iid] = parser.parse(r['prediction'])
    parsed_refs[iid] = [parser.parse(g) for g in manifest[iid]['references']]
print('parsing done')


def score_all(weights, agg='mean', gates=True):
    """agg: 'mean' (current default: average CE-F over all references),
    'single' (first reference only), 'max' (best-matching reference)."""
    saved_reversals = event_mod.ACTION_REVERSALS
    if not gates:
        event_mod.ACTION_REVERSALS = set()
    try:
        f1s, ps, rs = [], [], []
        for r in preds:
            iid = r['image_id']
            pe = parsed_preds[iid]
            ref_sets = parsed_refs[iid]
            if agg == 'single':
                ref_sets = ref_sets[:1]
            ces = [ricce_scores(pe, ge, taxes, weights) for ge in ref_sets]
            if not ces:
                continue
            if agg == 'max':
                best = max(ces, key=lambda c: c['ce_f1'])
                f1s.append(best['ce_f1']); ps.append(best['ce_precision']); rs.append(best['ce_recall'])
            else:
                f1s.append(np.mean([c['ce_f1'] for c in ces]))
                ps.append(np.mean([c['ce_precision'] for c in ces]))
                rs.append(np.mean([c['ce_recall'] for c in ces]))
        return {'ce_precision': float(np.mean(ps)), 'ce_recall': float(np.mean(rs)), 'ce_f1': float(np.mean(f1s)), 'n': len(f1s)}
    finally:
        event_mod.ACTION_REVERSALS = saved_reversals


results = {}

# --- G2: ontology-component ablation ---
# Cumulative: object -> +action -> +source/target -> +location -> +quantity -> +change_state
comp_order = ['object', 'action', ('source', 'target'), 'location', 'quantity', 'change_state']
active = set()
g2 = {}
for step in comp_order:
    keys = step if isinstance(step, tuple) else (step,)
    active.update(keys)
    w = {k: (base_weights.get(k, 1.0) if k in active else 0.0) for k in base_weights}
    label = '+'.join(sorted(active))
    g2[label] = score_all(w)
    print('G2', label, g2[label])
results['G2_ontology_components'] = g2

# --- G5: weight sensitivity: equal weights vs configured vs +/-20% perturbation ---
g5 = {}
g5['configured'] = score_all(base_weights)
g5['equal'] = score_all({k: 1.0 for k in base_weights})
rng = random.Random(0)
perturbed_f1 = []
for trial in range(20):
    w = {k: v * (1 + rng.uniform(-0.2, 0.2)) for k, v in base_weights.items()}
    perturbed_f1.append(score_all(w)['ce_f1'])
g5['perturbation_20pct'] = {'mean_ce_f1': float(np.mean(perturbed_f1)), 'std_ce_f1': float(np.std(perturbed_f1)),
                             'min_ce_f1': float(np.min(perturbed_f1)), 'max_ce_f1': float(np.max(perturbed_f1)), 'n_trials': len(perturbed_f1)}
print('G5', g5)
results['G5_weight_sensitivity'] = g5

# --- G6: contradiction gates on/off (action-reversal hard zero) ---
g6 = {'gates_on': score_all(base_weights, gates=True), 'gates_off': score_all(base_weights, gates=False)}
print('G6', g6)
results['G6_contradiction_gates'] = g6

# --- G4: reference aggregation strategy ---
g4 = {'single_reference': score_all(base_weights, agg='single'),
      'mean_over_references': score_all(base_weights, agg='mean'),
      'max_over_references': score_all(base_weights, agg='max')}
print('G4', g4)
results['G4_reference_aggregation'] = g4

dump_json(results, OUT / 'metrics.json')
print('wrote', OUT / 'metrics.json')
