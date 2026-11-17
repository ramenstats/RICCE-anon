"""Re-run the contradiction-gate stress tests (original 20 items; enlarged 227 items) under the FROZEN evaluator.
An item is 'gated' when CE-F < 0.15 (same rule as the repair study). Reports accuracy, gated-class precision/recall,
false-contradiction rate on non-gated items, per family; both parse modes."""
import json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, yaml
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
from ricce.metrics.event import ricce_scores
cfg = yaml.safe_load(open(ROOT / 'configs/project.yaml')); taxes = {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}; W = cfg['metric']['weights']
P = HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'], use_embeddings=True, embedding_model=cfg['parser'].get('embedding_model'), similarity_threshold=float(cfg['parser'].get('similarity_threshold', 0.58)))
out = {}
for name, f in [('original20', 'original_20_stress.jsonl'), ('enlarged227', 'action_reversal_stress.jsonl')]:
    items = [json.loads(l) for l in open(ROOT / 'artifacts_r1' / f)]
    for mode in ['multi', 'single']:
        tp = fp = fn = tn = 0; fam = defaultdict(lambda: [0, 0])
        for it in items:
            kw = dict(allow_multi_event=True) if mode == 'multi' else {}
            f1 = ricce_scores(P.parse(it['pred'], **kw), P.parse(it['gt'], **kw), taxes, W)['ce_f1']
            obs = 'gated' if f1 < 0.15 else 'not_gated'; ok = obs == it['expected']
            if it['expected'] == 'gated': tp += obs == 'gated'; fn += obs != 'gated'
            else: fp += obs == 'gated'; tn += obs != 'gated'
            fam[it.get('family', 'all')][0] += ok; fam[it.get('family', 'all')][1] += 1
        n = tp + fp + fn + tn
        out[f'{name}_{mode}'] = dict(n=n, accuracy=(tp + tn) / n, gated_recall=tp / max(1, tp + fn), gated_precision=tp / max(1, tp + fp), false_contradiction_rate=fp / max(1, fp + tn),
                                     tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), per_family={k: v[0] / v[1] for k, v in fam.items()})
        print(name, mode, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[f'{name}_{mode}'].items() if k != 'per_family'}, flush=True)
d = ROOT / 'results_cvpr_final/gate_stress'; d.mkdir(parents=True, exist_ok=True); json.dump(out, open(d / 'summary.json', 'w'), indent=1); os._exit(0)
