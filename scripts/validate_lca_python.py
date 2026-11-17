"""Cross-check the experimental Python LCA-F port against the HEMKit reference on REAL parsed label sets
(sample of E04 rows + all bind/st diagnostics). Writes results_cvpr_final/baselines/lca_validation/."""
import json, random, sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from ricce.baselines.registry import BaselineContext
from ricce.baselines.hierarchical.lca_f import lca_python
OUT = ROOT / 'results_cvpr_final' / 'baselines' / 'lca_validation'; OUT.mkdir(parents=True, exist_ok=True)
ctx = BaselineContext(); H = ctx.lca.H
random.seed(0)
items = []
for f in sorted((ROOT / 'artifacts_cvpr_final' / 'baselines' / 'inputs').glob('e04_*.jsonl')):
    rows = [json.loads(l) for l in open(f)]; items += random.sample(rows, min(1500, len(rows)))
for g in ['bind_swap', 'st_swap']:
    items += [json.loads(l) for l in open(ROOT / 'artifacts_cvpr_final' / 'baselines' / 'inputs' / f'diag_{g}.jsonl')]
ok = tot = 0; dP = []; dF = []; single_ok = single_tot = 0
for it in items:
    p, r = ctx.labels(it['candidate']), ctx.labels(it['reference'])
    a, b = ctx.lca(p, r), lca_python(p, r, H); tot += 1
    same = abs(a['P'] - b['P']) < 1e-5 and abs(a['R'] - b['R']) < 1e-5
    ok += same; dP.append(abs(a['P'] - b['P'])); dF.append(abs(a['F'] - b['F']))
summ = dict(n_pairs=tot, exact_agreement=ok / tot, mean_abs_diff_P=float(np.mean(dP)), max_abs_diff_P=float(np.max(dP)),
            mean_abs_diff_F=float(np.mean(dF)), max_abs_diff_F=float(np.max(dF)))
json.dump(summ, open(OUT / 'summary.json', 'w'), indent=1); print(summ)
