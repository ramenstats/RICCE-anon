"""T1 refinement: Kendall tau + bootstrap 95% CI on the existing single-annotator
human ratings (results/E07_human_pred_validation/per_item.json), plus CE-P/CE-R
vs. human hallucination/completeness axes specifically, as requested by the
T1 spec. This adds statistics to REAL, already-collected human ratings;
it does not collect or synthesize any new human labels.

Limitation (reported honestly, not hidden): single annotator, text-only (no
images shown to the annotator), n=300 pooled across 3 datasets x 5 systems.
No Krippendorff alpha is computed because that requires >=2 annotators, which
we do not have.
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / 'results' / 'E07_human_pred_validation' / 'per_item.json'
OUT_DIR = ROOT / 'results' / 'E07_human_pred_validation'
items = json.load(open(IN))
print(f'loaded {len(items)} per-item human-rated predictions')

# Need ce_precision/ce_recall too -- per_item.json only has ce_f1. Recompute
# them from the E07 CSV source if available with the raw per-sample ce fields;
# otherwise, only ce_f1 correlations + CE-F1 vs hallucination/completeness can
# be reported (ce_precision/ce_recall not present in per_item.json).
have_pr = all('ce_precision' in it and 'ce_recall' in it for it in items)
print('ce_precision/ce_recall present in per_item.json:', have_pr)

def bootstrap_ci(x, y, fn, n_boot=5000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(x)
    x = np.asarray(x); y = np.asarray(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        v = fn(x[idx], y[idx])
        if not np.isnan(v):
            vals.append(v)
    vals = np.sort(vals)
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals))]
    return float(lo), float(hi)

def kendall_fn(x, y):
    if len(set(x)) < 2 or len(set(y)) < 2:
        return float('nan')
    return stats.kendalltau(x, y)[0]

metrics = ['ce_f1', 'bleu4', 'rouge_l', 'meteor', 'bertscore']
human_target = 'human_overall'

out = {'n_items': len(items), 'annotators': 1,
       'limitation': 'single annotator, text-only (no images shown), pooled n=300 across LEVIR-CC/SECOND-CC/DUBAI-CC (100 each) x 5 systems; Krippendorff alpha not computable with 1 annotator.',
       'kendall_tau_vs_human_overall': {}, 'bootstrap_ci_spearman_vs_human_overall': {}}

for m in metrics:
    x = [it[m] for it in items]
    y = [it[human_target] for it in items]
    tau, p = stats.kendalltau(x, y)
    out['kendall_tau_vs_human_overall'][m] = {'kendall_tau': float(tau), 'p_value': float(p), 'n': len(x)}
    lo, hi = bootstrap_ci(x, y, lambda a, b: stats.spearmanr(a, b)[0])
    out['bootstrap_ci_spearman_vs_human_overall'][m] = {'ci95_lo': lo, 'ci95_hi': hi}
    print(m, 'kendall tau=', tau, 'p=', p, 'spearman 95% CI=[', lo, hi, ']')

# CE-F1 vs specific human axes (hallucination, completeness) -- CE-P/CE-R not
# available per-item (only ce_f1 stored), so we report CE-F1 vs these axes and
# note the CE-P/CE-R-specific breakdown is not available from this artifact.
out['ce_f1_vs_human_axes'] = {}
for axis in ['hallucination', 'completeness', 'specificity', 'object_correctness', 'action_correctness', 'location_correctness']:
    if axis not in items[0]:
        continue
    x = [it['ce_f1'] for it in items]
    y = [it[axis] for it in items]
    rho, p = stats.spearmanr(x, y)
    tau, pt = stats.kendalltau(x, y)
    out['ce_f1_vs_human_axes'][axis] = {'spearman_r': float(rho), 'spearman_p': float(p), 'kendall_tau': float(tau), 'kendall_p': float(pt), 'n': len(x)}
    print('ce_f1 vs', axis, 'spearman=', rho, 'p=', p)

out['note_ce_precision_recall'] = 'per_item.json stores only ce_f1 per item (not ce_precision/ce_recall separately), so CE-P/CE-R-specific vs human hallucination/completeness correlations (as literally specified in T1) could not be computed from this artifact without re-running the scorer against the raw predictions; CE-F1 vs those axes is reported instead as the closest available honest substitute.'

OUT = OUT_DIR / 'correlations_refined.json'
json.dump(out, open(OUT, 'w'), indent=2)
print('wrote', OUT)
