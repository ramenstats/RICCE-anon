"""v2-final: gate-stress recall and real reference-pair FCR under the TRUE
final adopted v2 configuration (dual matching + exclude_root=True + hard
gate) -- distinct from the earlier G1/G2/G3 ablation, which deliberately
isolated gate strength alone (dual_matching=False, exclude_root=False) so
that only the gate variable changed. This script reports the number that
actually corresponds to the paper's adopted evaluator."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
from ricce.metrics.event_v2 import ricce_scores_v2
from ricce.utils import read_jsonl
import yaml
cfg = yaml.safe_load(open(ROOT / 'configs/project.yaml'))
taxes = {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}
W = cfg['metric']['weights']
parser = HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'],
                       use_embeddings=True, embedding_model=cfg['parser'].get('embedding_model'),
                       similarity_threshold=float(cfg['parser'].get('similarity_threshold', 0.58)))
FINAL_V2 = dict(gate_mode='hard', exclude_root=True, dual_matching=True)

def v2f(cand, ref):
    r = ricce_scores_v2(cand, ref, taxes, W, **FINAL_V2)
    return 0.0 if r['not_scorable'] else r['ce_f1']

out = {}
for name, fn in [('original20', 'original_20_stress.jsonl'), ('enlarged227', 'action_reversal_stress.jsonl')]:
    items = [json.loads(l) for l in open(ROOT / 'artifacts_r1' / fn)]
    tp = fp = fn_ = tn = 0
    for it in items:
        pe = parser.parse(it['pred'], allow_multi_event=True)
        ge = parser.parse(it['gt'], allow_multi_event=True)
        f1 = v2f(pe, ge)
        obs = 'gated' if f1 < 0.15 else 'not_gated'
        if it['expected'] == 'gated':
            tp += obs == 'gated'; fn_ += obs != 'gated'
        else:
            fp += obs == 'gated'; tn += obs != 'gated'
    n = tp + fp + fn_ + tn
    out[name] = dict(n=n, gated_recall=tp / max(1, tp + fn_), false_contradiction_rate=fp / max(1, fp + tn), tp=tp, fp=fp, fn=fn_, tn=tn)
    print(name, out[name], flush=True)

man = list(read_jsonl(ROOT / 'artifacts/manifests/LEVIR_CC.jsonl'))
pairs = []
for r in man:
    refs = [x for x in r.get('references', []) if x and x.strip()]
    if len(refs) >= 2:
        pairs.append((parser.parse(refs[0], allow_multi_event=True), parser.parse(refs[1], allow_multi_event=True)))
fired = sum(1 for a, b in pairs if v2f(a, b) < 0.15)
out['real_refpair'] = dict(n=len(pairs), false_contradiction_rate=fired / max(1, len(pairs)), n_fired=fired)
print('real_refpair', out['real_refpair'], flush=True)

OUT = ROOT / 'results_v2'; OUT.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT / 'final_v2_gate_stress.json', 'w'), indent=1)
print('DONE')
