"""Human study: direct comparison of metrics, and per-annotator robustness.

Primary comparison, stated on 2026-10-26 before any metric-vs-human result was computed:
Spearman correlation of CE-F with the consensus (mean of three annotators) Overall rating,
against CIDEr-D and BERTScore. A correlation for each metric alone does not answer
"does CE-F agree with humans better than metric M?"; this script computes the paired
difference  rho(CE-F, H) - rho(M, H)  on exactly the items both metrics score, with a
cluster bootstrap over image pairs (10,000 resamples, seed 20260923, same scheme as
ricce/evaluation/human_analysis.py). Holm correction across the five comparisons
(CIDEr-D, BERTScore primary; BLEU-4, METEOR, ROUGE-L secondary) within each subset.

Robustness (not used to choose anything): the same correlations with each annotator's
own Overall rating instead of the consensus.

Inputs : results_v3/human_eval/analysis_final_evaluator/consensus_items.csv,
         hidden/answer_key_final_evaluator.jsonl, collected/*.csv
Outputs: results_v3/human_eval/analysis_final_evaluator/{metric_differences.csv, per_annotator_spearman.csv}"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.evaluation.human_analysis import load_annotations, load_key, build_frame, consensus_items, _corr, holm

H = ROOT / 'artifacts_cvpr_final' / 'human_eval' / 'image_grounded'
OUT = ROOT / 'results_v3' / 'human_eval' / 'analysis_final_evaluator'
B, SEED = 10000, 20260923
OTHERS = ['cider_d', 'bertscore_f', 'bleu4', 'meteor', 'rougeL']
df = load_annotations(sorted(glob.glob(str(H / 'collected' / '*.csv'))))
key_items, _ = load_key(H / 'hidden' / 'answer_key_final_evaluator.jsonl')
fr = build_frame(consensus_items(df), key_items)


def paired_diff(d, a, b, h='h_overall'):
    x1, x2, y = d['m_' + a].to_numpy(float), d['m_' + b].to_numpy(float), d[h].to_numpy(float)
    est = _corr(x1, y, 'spearman') - _corr(x2, y, 'spearman')
    cl = pd.factorize(d.cluster)[0]; idx = [np.where(cl == c)[0] for c in range(cl.max() + 1)]
    rng = np.random.default_rng(SEED); vals = np.empty(B)
    for i in range(B):
        ii = np.concatenate([idx[j] for j in rng.integers(0, len(idx), len(idx))])
        vals[i] = _corr(x1[ii], y[ii], 'spearman') - _corr(x2[ii], y[ii], 'spearman')
    v = vals[~np.isnan(vals)]
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = max(min(1.0, 2 * (min((v <= 0).sum(), (v >= 0).sum()) + 1) / (len(v) + 1)), 1 / (len(v) + 1))
    return est, lo, hi, p, len(idx)


rows = []
for sub in ('all', 'model_output', 'hard_case'):
    f = fr if sub == 'all' else fr[fr.subset == sub]
    part = []
    for m in OTHERS:
        d = f[['m_ce_f1', 'm_' + m, 'h_overall', 'cluster']].dropna() if 'm_' + m in f else pd.DataFrame()
        if len(d) < 20:
            part.append(dict(subset=sub, comparison=f'ce_f1 - {m}', n=len(d), note='insufficient items')); continue
        est, lo, hi, p, ncl = paired_diff(d, 'ce_f1', m)
        part.append(dict(subset=sub, comparison=f'ce_f1 - {m}', role='primary' if m in ('cider_d', 'bertscore_f') else 'secondary', n=len(d), n_clusters=ncl,
                         rho_ce_f1=_corr(d.m_ce_f1.to_numpy(float), d.h_overall.to_numpy(float), 'spearman'),
                         rho_other=_corr(d['m_' + m].to_numpy(float), d.h_overall.to_numpy(float), 'spearman'),
                         diff=est, ci_lo=lo, ci_hi=hi, p_boot=p, resamples=B))
    t = pd.DataFrame(part); ok = t.get('p_boot', pd.Series(dtype=float)).notna()
    if ok.any(): t.loc[ok, 'p_holm'] = holm(t.loc[ok, 'p_boot'])
    rows.append(t)
D = pd.concat(rows); D.to_csv(OUT / 'metric_differences.csv', index=False)

# per-annotator robustness: each annotator's own Overall vs each metric (same items)
rows = []
for aid, g in df[df.task == 'A'].groupby('annotator_id'):
    own = g[['item_id', 'overall']].rename(columns={'overall': 'own_overall'})
    f = fr.merge(own, on='item_id')
    for sub in ('all', 'model_output'):
        s = f if sub == 'all' else f[f.subset == sub]
        for m in ['ce_f1'] + OTHERS:
            d = s[['m_' + m, 'own_overall']].dropna() if 'm_' + m in s else pd.DataFrame()
            rows.append(dict(annotator=aid, subset=sub, metric=m, n=len(d),
                             spearman=_corr(d['m_' + m].to_numpy(float), d.own_overall.to_numpy(float), 'spearman') if len(d) >= 20 else np.nan))
P = pd.DataFrame(rows); P.to_csv(OUT / 'per_annotator_spearman.csv', index=False)
pd.set_option('display.width', 250)
print(D.drop(columns=[c for c in ('resamples',) if c in D]).round(3).to_string(index=False))
print(P.pivot_table(index=['subset', 'metric'], columns='annotator', values='spearman').round(3).to_string())
