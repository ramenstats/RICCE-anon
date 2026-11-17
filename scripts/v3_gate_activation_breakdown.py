"""v3: what kind of event pairs does the gate act on in real same-image
LEVIR-CC reference pairs (first two references per image)? For each activation,
record whether both objects are specified (and equal vs ancestor/descendant) or
whether at least one side leaves the object unspecified, and whether either
reference is multi-event. Output: results_v3/gate/activation_breakdown.json."""
import json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl
from ricce.benchmarks.env import load_taxes, load_weights
from ricce.evaluation.parse_store import ParseStore
from ricce.metrics.final import gate_activations
T, W = load_taxes(), load_weights(); ot = T['object']
store = ParseStore(ROOT / 'results_v3' / 'parse_cache.json')
c = Counter(); n = 0
for m in read_jsonl(ROOT / 'artifacts' / 'manifests' / 'LEVIR_CC.jsonl'):
    refs = [x for x in m.get('references', []) if x and x.strip()]
    if len(refs) < 2:
        continue
    a, b = store.events(refs[0], 'multi'), store.events(refs[1], 'multi')
    act = gate_activations(a, b, T, W)
    if not act:
        continue
    n += 1
    multi = len(a) >= 2 or len(b) >= 2
    kinds = set()
    for i, j, k in act:
        oa = None if a[i].object in (None, ot.root) else a[i].object
        ob = None if b[j].object in (None, ot.root) else b[j].object
        kinds.add('object_unspecified' if (oa is None or ob is None) else ('same_object' if oa == ob else 'ancestor_descendant'))
    key = 'object_unspecified' if 'object_unspecified' in kinds else ('same_object' if 'same_object' in kinds else 'ancestor_descendant')
    c[key] += 1; c['multi_event' if multi else 'single_event'] += 1
    c[f'{key}|{"multi" if multi else "single"}'] += 1
out = dict(n_activated_pairs=n, **c)
json.dump(out, open(ROOT / 'results_v3' / 'gate' / 'activation_breakdown.json', 'w'), indent=1)
print(out)
