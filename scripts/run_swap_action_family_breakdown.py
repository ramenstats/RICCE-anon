"""Addition (README sec. 13): break the pooled
T1/T2 swap directional-inversion accuracy down by action family
(construction/demolition vs expansion/contraction vs conversion), pooling
the existing per-sample_swap.jsonl files across all three seeds for each of
zero-shot Qwen, plain SFT, the step-matched control (sft2x) and the
swap-augmented model (swapsft). Reads only already-generated artifacts;
does not re-run inference or re-parse anything."""
import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAMILY = {'constructed': 'construction/demolition', 'demolished': 'construction/demolition',
          'appearance': 'construction/demolition', 'disappearance': 'construction/demolition',
          'expansion': 'expansion/contraction', 'contraction': 'expansion/contraction',
          'conversion': 'conversion'}


def load(paths):
    rows = []
    for p in paths:
        if not p.exists():
            print('missing', p); continue
        rows.extend(json.loads(l) for l in open(p))
    return rows


def breakdown(rows, boot=10000, seed=0):
    d = defaultdict(list)
    for r in rows:
        if r['category'] != 'directional':
            continue
        fam = FAMILY.get(r['normal_action'], 'other')
        d[fam].append(bool(r['consistent']))
    rng = np.random.default_rng(seed)
    out = {}
    for fam, vals in d.items():
        x = np.array(vals, dtype=float)
        n = len(x)
        acc = float(x.mean())
        if n > 1:
            idx = rng.integers(0, n, (boot, n))
            bm = x[idx].mean(1)
            lo, hi = float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))
        else:
            lo, hi = None, None
        out[fam] = {'n': n, 'inversion_acc': acc, 'ci95': [lo, hi]}
    out['ALL'] = {'n': sum(len(v) for v in d.values()), 'inversion_acc': float(np.mean([c for v in d.values() for c in v])) if d else None}
    return out


groups = {
    'qwen_zeroshot': [ROOT / 'results/E10/LEVIR_CC/per_sample_swap.jsonl'],
    'sft': [ROOT / f'results_cvpr_final/temporal_equivariance/runs/sft_seed{s}/e10/per_sample_swap.jsonl' for s in (1, 42, 123)],
    'sft2x_control': [ROOT / f'results_cvpr_final/temporal_equivariance/runs/sft2x_seed{s}/e10/per_sample_swap.jsonl' for s in (1, 42, 123)],
    'swapsft_augmented': [ROOT / f'results_cvpr_final/temporal_equivariance/runs/swapsft_seed{s}/e10/per_sample_swap.jsonl' for s in (1, 42, 123)],
}

result = {}
for name, paths in groups.items():
    rows = load(paths)
    result[name] = breakdown(rows)
    print(name, json.dumps(result[name], indent=1), flush=True)

OUT = ROOT / 'results_metric_v2'
OUT.mkdir(parents=True, exist_ok=True)
json.dump(result, open(OUT / 'swap_action_family_breakdown.json', 'w'), indent=1)
print('wrote', OUT / 'swap_action_family_breakdown.json')
