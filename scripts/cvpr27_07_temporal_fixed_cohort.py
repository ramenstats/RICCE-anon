"""CVPR'27, T1: temporal-swap results on a FIXED cohort of image pairs.

The paper's inversion accuracy is conditional on the model's own output being
directional, so its denominator differs between models and seeds (104-138 of 300).
Here the cohort is fixed by the REFERENCES, identically for every model/seed:
  * gt-directional pairs: the majority action of the image's references (same parser and
    rule as scripts/eval_temporal_swap.py: single-event parse, no embeddings) is
    directional (in RHO_ACTION and not no_change);
  * gt-no-change pairs: the majority action is no_change.
Endpoints per run (3 groups x 3 seeds, same 300 LEVIR-CC test pairs):
  E0  conditional inversion accuracy (paper's metric; denominator = model-directional outputs)
  E1  gt-directional cohort: the normal-order output is directional AND the swapped-order output
      reverses it (per_sample_swap 'consistent'); any other output counts as a failure
  E2  gt-directional cohort: the swapped-order output reverses the reference action
  E3  gt-no-change cohort: both orders yield no_change
  E4  all 300 pairs: temporally consistent pair (directional and inverted, or no-change kept)
Breakdown of E1 by reference action family. Paired differences swapsft - sft2x and
swapsft - sft: per-image means over seeds, image bootstrap (B = 10,000, seed 0).
Output: submission_cvpr27_new_version/10_TEMPORAL/"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl, load_yaml
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
from ricce.metrics.event import is_action_reversal
from ricce.training.temporal_swap import RHO_ACTION

R = ROOT / 'results_cvpr_final' / 'temporal_equivariance' / 'runs'
OUT = ROOT / 'submission_cvpr27_new_version' / '10_TEMPORAL'; OUT.mkdir(parents=True, exist_ok=True)
cfg = load_yaml(ROOT / 'configs' / 'project.yaml')
T = {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}
parser = HybridParser(T['object'], T['action'], T['location'], T['quantity'], use_embeddings=False)
gt = {r['image_id']: r['references'] for r in read_jsonl(ROOT / 'artifacts' / 'manifests' / 'LEVIR_CC.jsonl') if r['split'] == 'test'}
RUNS = sorted(p.name for p in R.iterdir() if (p / 'e10' / 'per_sample_swap.jsonl').exists())
ids = sorted(json.loads(l)['image_id'] for l in open(R / RUNS[0] / 'e10' / 'per_sample_swap.jsonl'))
cohort = {}
for i in ids:
    votes, evs = Counter(), {}
    for ref in gt[i]:
        e = parser.parse(ref)[0]; votes[e.action] += 1; evs.setdefault(e.action, e)
    ga = votes.most_common(1)[0][0]
    kind = 'directional' if ga in RHO_ACTION and ga != 'no_change' else 'no_change' if ga == 'no_change' else 'other'
    cohort[i] = (kind, ga, evs[ga])
pd.DataFrame([dict(image_id=i, cohort=k, reference_majority_action=a) for i, (k, a, _) in cohort.items()]).to_csv(OUT / 'frozen_300_image_pairs.csv', index=False)

rows, per_img = [], {}
for run in RUNS:
    g, s = run.rsplit('_seed', 1)
    ps = {json.loads(l)['image_id']: json.loads(l) for l in open(R / run / 'e10' / 'per_sample_swap.jsonl')}
    raw = {json.loads(l)['image_id']: json.loads(l) for l in open(R / run / 'e10' / 'raw_predictions.jsonl')}
    assert sorted(ps) == ids, run
    rec = []
    for i in ids:
        k, ga, ge = cohort[i]; p = ps[i]
        es = parser.parse(raw[i]['swapped_order'])[0]
        rec.append(dict(image_id=i, cohort=k, family=ga, model_directional=p['category'] == 'directional', consistent=p['consistent'] is True,
                        model_category=p['category'],
                        E1=float(p['category'] == 'directional' and p['consistent'] is True) if k == 'directional' else np.nan,
                        E2=float(bool(is_action_reversal(es, ge, T))) if k == 'directional' else np.nan,
                        E3=float(p['category'] == 'no_change' and p['consistent'] is True) if k == 'no_change' else np.nan,
                        E4=float(p['consistent'] is True and p['category'] in ('directional', 'no_change'))))
    d = pd.DataFrame(rec); per_img[(g, int(s))] = d.set_index('image_id')
    dirm = d[d.model_directional]
    rows.append(dict(group=g, seed=int(s), E0_conditional_inversion=dirm.consistent.mean(), E0_n=len(dirm),
                     E1_fixed_gt_directional=d.E1.mean(), E1_n=int(d.E1.notna().sum()), E2_swapped_reverses_reference=d.E2.mean(),
                     E3_fixed_gt_nochange=d.E3.mean(), E3_n=int(d.E3.notna().sum()), E4_all_pairs=d.E4.mean(), E4_n=len(d),
                     model_directional_in_gt_directional=d[d.cohort == 'directional'].model_directional.mean()))
S = pd.DataFrame(rows).sort_values(['group', 'seed']); S.to_csv(OUT / 'fixed_denominator_results.csv', index=False)
agg = S.groupby('group').agg(['mean', 'std']).drop(columns='seed', level=0)
agg.to_csv(OUT / 'fixed_denominator_by_group.csv')

fam = []
for (g, s), d in per_img.items():
    x = d[d.cohort == 'directional']
    for f, h in x.groupby('family'):
        fam.append(dict(group=g, seed=s, family=f, n=len(h), E1=h.E1.mean()))
F = pd.DataFrame(fam); F.groupby(['family', 'group']).agg(n=('n', 'first'), E1_mean=('E1', 'mean'), E1_std=('E1', 'std')).reset_index().to_csv(
    OUT / 'fixed_cohort_by_action_family.csv', index=False)

rng = np.random.default_rng(0); B = 10000
boot = []
for a, b in (('swapsft', 'sft2x'), ('swapsft', 'sft')):
    for e in ('E1', 'E2', 'E3', 'E4'):
        ma = np.nanmean(np.stack([per_img[(a, s)][e].to_numpy(float) for s in (1, 42, 123)]), 0)
        mb = np.nanmean(np.stack([per_img[(b, s)][e].to_numpy(float) for s in (1, 42, 123)]), 0)
        ok = ~np.isnan(ma) & ~np.isnan(mb); dlt = (ma - mb)[ok]
        bs = np.array([dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(B)])
        boot.append(dict(comparison=f'{a} - {b}', endpoint=e, n_images=int(ok.sum()), diff=dlt.mean(),
                         ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
pd.DataFrame(boot).to_csv(OUT / 'fixed_cohort_paired_differences.csv', index=False)
print(pd.Series({k: v for k, v in Counter(c[0] for c in cohort.values()).items()}).to_string())
print(agg.round(3).to_string()); print(pd.DataFrame(boot).round(3).to_string())
print(F.groupby(['family', 'group']).agg(n=('n', 'first'), E1=('E1', 'mean')).round(3).to_string())
