"""v2-final: sweep the soft-gate penalty gamma in {0.05,0.10,0.20,0.30,0.50}
(the single gamma=0.1 point was already covered in results_metric_v2/). Same
three tests as the G1/G2/G3 ablation: RICCE-Bench ordering (scoring
formula only, frozen matching/root handling held fixed so only gate
strength varies), the 227-item reversal stress set, and the real
same-image reference-pair false-contradiction test (10,077 pairs, cached
from the earlier run to avoid re-parsing)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from ricce.schema import Event
from ricce.utils import read_jsonl
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
from ricce.metrics.event_v2 import ricce_scores_v2
import yaml

cfg = yaml.safe_load(open(ROOT / 'configs/project.yaml'))
taxes = {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}
W = cfg['metric']['weights']
GAMMAS = [0.05, 0.10, 0.20, 0.30, 0.50]
COMMON = dict(exclude_root=False, dual_matching=False)  # isolate gate strength only, as in the G1/G2/G3 ablation


def v2f(cand, ref, gamma):
    r = ricce_scores_v2(cand, ref, taxes, W, gate_mode='soft', soft_gamma=gamma, **COMMON)
    return None if r['not_scorable'] else r['ce_f1']


def mk(dicts):
    return [Event(**{k: v for k, v in d.items() if k in Event.__dataclass_fields__}) for d in dicts]


# ---- RICCE-Bench ordering ----
items = []
with open(ROOT / 'artifacts_cvpr_final/riccebench/items.jsonl') as f:
    for l in f:
        d = json.loads(l)
        if d['purity_checks']['pure']:
            items.append((d['source_id'], d['perturbation_type'], mk(d['candidate_event']), mk(d['source_event'])))

import pandas as pd
def ordering_valid_acc(scores):
    df = pd.DataFrame({'source_id': [x[0] for x in items], 'ptype': [x[1] for x in items], 'score': scores})
    piv = df.pivot_table(index='source_id', columns='ptype', values='score')
    GOOD = ['P0_paraphrase', 'P1_near_ancestor']; BAD = ['P4_action_reversal', 'P9_change_state_reversal', 'negation_flip', 'source_target_swap']
    n = win = 0
    for gd in GOOD:
        for bd in BAD:
            if gd in piv and bd in piv:
                x = piv[[gd, bd]].dropna()
                win += int((x[gd] > x[bd] + 1e-9).sum()); n += len(x)
    return win / max(1, n), n


# ---- stress set ----
parser = HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'],
                       use_embeddings=True, embedding_model=cfg['parser'].get('embedding_model'),
                       similarity_threshold=float(cfg['parser'].get('similarity_threshold', 0.58)))
stress = [json.loads(l) for l in open(ROOT / 'artifacts_r1/action_reversal_stress.jsonl')]
stress_parsed = [(it, parser.parse(it['pred'], allow_multi_event=True), parser.parse(it['gt'], allow_multi_event=True)) for it in stress]

# ---- real reference pairs ----
man = list(read_jsonl(ROOT / 'artifacts/manifests/LEVIR_CC.jsonl'))
pairs = []
for r in man:
    refs = [x for x in r.get('references', []) if x and x.strip()]
    if len(refs) >= 2:
        pairs.append((parser.parse(refs[0], allow_multi_event=True), parser.parse(refs[1], allow_multi_event=True)))
print(f'{len(pairs)} real reference pairs, {len(stress)} stress items, {len(items)} bench items', flush=True)

out = {}
for gamma in GAMMAS:
    scores = [v2f(c, r, gamma) for _, _, c, r in items]
    scores = [np.nan if s is None else s for s in scores]
    valid_acc, n_valid = ordering_valid_acc(scores)

    tp = fp = fn = tn = 0
    for it, pe, ge in stress_parsed:
        f1 = v2f(pe, ge, gamma); f1 = 0.0 if f1 is None else f1
        obs = 'gated' if f1 < 0.15 else 'not_gated'
        if it['expected'] == 'gated':
            tp += obs == 'gated'; fn += obs != 'gated'
        else:
            fp += obs == 'gated'; tn += obs != 'gated'
    recall = tp / max(1, tp + fn); fcr_stress = fp / max(1, fp + tn)

    fired = 0
    for ea, eb in pairs:
        f1 = v2f(ea, eb, gamma); f1 = 0.0 if f1 is None else f1
        if f1 < 0.15:
            fired += 1
    fcr_real = fired / max(1, len(pairs))

    out[f'gamma_{gamma}'] = dict(gamma=gamma, bench_valid_acc=valid_acc, n_valid=n_valid,
                                  stress_recall=recall, stress_fcr=fcr_stress,
                                  real_refpair_fcr=fcr_real, n_real_pairs=len(pairs))
    print(out[f'gamma_{gamma}'], flush=True)

OUT = ROOT / 'artifacts_v2'; OUT.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT / 'gate_gamma_sweep.json', 'w'), indent=1)
print('DONE')
