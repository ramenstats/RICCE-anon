"""Robustness to simulated acquisition interference, step 3b: two contrasts Real-CD
draws for change detection (Fang et al., TIP 2026, Sec. III-B/C), tested on the
captions scored by scripts/cvpr27_25_robustness_score.py.

  1. Same vs. different interference on the two images. Real-CD reports that
     different types on the two images shift the bi-temporal distributions most.
     For captions we test, per (system, dataset), with images as the unit:
       hallucinated change on unchanged scenes, mean over the 20 mixed-type
       scenarios minus mean over the 5 same-type scenarios;
       missed change on changed scenes, same-type minus mixed-type;
       CE-F on unchanged scenes (same - mixed) and on changed scenes (mixed - same).
  2. T1 vs. T2 sensitivity. Real-CD reports that the same interference applied to
     one or the other image degrades performance differently. Per type k:
     mean CE-F over the 5 scenarios with k on T1 minus the 5 with k on T2 (the k-k
     scenario is in both and cancels).
95% CIs: paired bootstrap over images (10,000 resamples, seed 0); a contrast is
called consistent when its CI excludes 0. Severity 3 scenarios only.
Outputs -> results_cvpr27_robustness/same_vs_mixed.csv, t1_vs_t2.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results_cvpr27_robustness'
B = 10000
TYPES = 'FGIMS'
SAME = [f's3_{k}-{k}' for k in TYPES]
MIXED = [f's3_{a}-{b}' for a in TYPES for b in TYPES if a != b]
D = pd.read_csv(R / 'per_image_scores.csv.gz')
D = D[D.condition.str.startswith('s3_')]
D['hall'] = D.pred_stratum.isin(['full', 'partial']).astype(float)
D['miss'] = (D.pred_stratum == 'no_change').astype(float)
rng = np.random.default_rng(0)


def boot(d):
    d = np.asarray(d, float)
    bm = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return dict(n=len(d), mean=float(d.mean()), lo=float(np.percentile(bm, 2.5)), hi=float(np.percentile(bm, 97.5)))


rows, rows2 = [], []
for (ds, sy), g in D.groupby(['dataset', 'system']):
    assert set(g.condition) == set(SAME + MIXED), (ds, sy)
    for name, val, sub, sign in (('hallucinated_change_unchanged', 'hall', ~g.ref_changed, +1),
                                 ('missed_change_changed', 'miss', g.ref_changed, -1),
                                 ('ce_f_unchanged', 'ce_f', ~g.ref_changed, -1),
                                 ('ce_f_changed', 'ce_f', g.ref_changed, +1)):
        p = g[sub].pivot(index='image_id', columns='condition', values=val).dropna()
        same, mixed = p[SAME].mean(axis=1), p[MIXED].mean(axis=1)
        r = boot(sign * (mixed - same))   # oriented so that > 0 is the expected direction
        rows.append(dict(dataset=ds, system=sy, contrast=name, direction='mixed - same' if sign > 0 else 'same - mixed',
                         same=float(same.mean()), mixed=float(mixed.mean()), **r, consistent=bool(r['lo'] > 0)))
    p = g.pivot(index='image_id', columns='condition', values='ce_f').dropna()
    for k in TYPES:
        t1 = p[[f's3_{k}-{x}' for x in TYPES]].mean(axis=1); t2 = p[[f's3_{x}-{k}' for x in TYPES]].mean(axis=1)
        r = boot(t1 - t2)
        rows2.append(dict(dataset=ds, system=sy, type=k, ce_f_on_t1=float(t1.mean()), ce_f_on_t2=float(t2.mean()), **r,
                          ci_excludes_0=bool(r['lo'] > 0 or r['hi'] < 0)))
A = pd.DataFrame(rows); A.to_csv(R / 'same_vs_mixed.csv', index=False)
T = pd.DataFrame(rows2); T.to_csv(R / 't1_vs_t2.csv', index=False)
pd.set_option('display.width', 250)
print(A.round(3).to_string())
print(A.groupby('contrast').consistent.agg(['sum', 'count']))
print(T.round(3).to_string())
print('T1-vs-T2 CI excludes 0:', int(T.ci_excludes_0.sum()), 'of', len(T))
