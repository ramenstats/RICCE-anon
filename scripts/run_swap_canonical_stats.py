"""Paired bootstrap (10,000 resamples, per image, seeds averaged) of frozen-evaluator CE-F: swapsft vs sft and vs sft2x."""
import json, numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1] / 'results_cvpr_final'
def per(g): return pd.concat([pd.read_csv(R / f'system_benchmark_swap/per_image/LEVIR_CC__{g}_seed{s}__multi.csv').set_index('image_id').ce_f for s in [1, 42, 123]], axis=1).mean(axis=1)
rng = np.random.default_rng(0); out = {}
for a, b in [('swapsft', 'sft'), ('swapsft', 'sft2x')]:
    d = (per(a) - per(b)).values; m = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    out[f'{a}_minus_{b}'] = dict(diff=float(d.mean()), ci95=[float(x) for x in np.percentile(m, [2.5, 97.5])], n_images=len(d), resamples=10000)
json.dump(out, open(R / 'temporal_equivariance/canonical_cef_paired.json', 'w'), indent=1); print(out)
