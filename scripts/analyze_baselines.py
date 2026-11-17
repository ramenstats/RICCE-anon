"""Aggregate per-sample baseline CSVs -> summary tables.
Outputs (results_cvpr_final/baselines/):
  perturbation_summary.csv / .json : per dataset+ALL, perturbation type, metric: mean F, paired drop vs
      P0_original (paired on image_id) with 2000-resample paired bootstrap 95% CI (seed 0)
  diagnostics_summary.csv          : binding/source-target/identity/clause-order diagnostics (mean, bootstrap CI)
  system_summary.csv, system_rankings.json : per dataset x system mean of each metric, ranks, Spearman vs CE-F
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results_cvpr_final' / 'baselines'
PS = R / 'per_sample'
METRICS = ['ricce', 'kiritchenko_micro', 'kiritchenko_macro', 'lca_f', 'hcm', 'taxaware_single', 'taxaware_axisflat']
NB = 2000
rng = np.random.default_rng(0)


def agg_fn(m, df):
    """returns function idx-array -> F for a resampled index set (micro = ratio of sums)."""
    if m == 'kiritchenko_micro':
        I, NP, NR = (df[f'{m}_{k}'].to_numpy(float) for k in ('inter', 'np', 'nr'))
        def f(ix):
            i, p, r = I[ix].sum(), NP[ix].sum(), NR[ix].sum()
            P = i / p if p else 0.0; Rr = i / r if r else 0.0
            return 2 * P * Rr / (P + Rr) if P + Rr else 0.0
        return f
    v = df[f'{m}_F'].to_numpy(float)
    return lambda ix: v[ix].mean()


def load_all(pattern):
    return pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(PS / pattern)))], ignore_index=True)


def boot_ci(fa, fb, n, paired=True):
    """fa,fb: functions of index arrays. returns diffs (fa-fb) bootstrap stats"""
    ds = np.empty(NB)
    for b in range(NB):
        ix = rng.integers(0, n, n)
        ds[b] = fa(ix) - fb(ix)
    return np.percentile(ds, [2.5, 97.5])


def perturbation():
    e = load_all('e04_*.csv'); e['image_id'] = e['id'].str.split('|').str[0]
    out = []
    for ds in ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC', 'ALL']:
        d = e if ds == 'ALL' else e[e.dataset == ds]
        d = d.assign(key=d.dataset + '/' + d.image_id)
        base = d[d.group == 'P0_original'].drop_duplicates('key').set_index('key')
        for t in sorted(d.group.unique()):
            sub = d[d.group == t].drop_duplicates('key')
            sub = sub[sub.key.isin(base.index)].set_index('key')
            b = base.loc[sub.index].reset_index(); s = sub.reset_index()
            n = len(s)
            for m in METRICS:
                fs, fb = agg_fn(m, s), agg_fn(m, b)
                ix = np.arange(n)
                mean_t, mean_0 = fs(ix), fb(ix)
                lo, hi = boot_ci(fb, fs, n)
                out.append(dict(dataset=ds, type=t, metric=m, n=n, mean_type=mean_t, mean_P0=mean_0,
                                drop=mean_0 - mean_t, drop_ci_lo=lo, drop_ci_hi=hi))
    o = pd.DataFrame(out); o.to_csv(R / 'perturbation_summary.csv', index=False)
    return o


def diagnostics():
    rows = []
    for g in ['identity', 'clause_order', 'bind_swap', 'st_swap']:
        p = PS / f'diag_{g}.csv'
        if not p.exists(): continue
        d = pd.read_csv(p)
        for m in METRICS:
            f = agg_fn(m, d); ix = np.arange(len(d))
            v = f(ix)
            bs = np.array([f(rng.integers(0, len(d), len(d))) for _ in range(NB)])
            rows.append(dict(group=g, metric=m, n=len(d), mean_F=v, ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                             frac_F_eq_1=float((d[f'{m}_F'] > 1 - 1e-9).mean())))
    o = pd.DataFrame(rows); o.to_csv(R / 'diagnostics_summary.csv', index=False); return o


def systems():
    s = load_all('sys_*.csv')
    rows = []
    for (ds, sysn), d in s.groupby(['dataset', 'group']):
        r = dict(dataset=ds, system=sysn, n=len(d))
        for m in METRICS:
            r[m] = agg_fn(m, d)(np.arange(len(d)))
        rows.append(r)
    t = pd.DataFrame(rows)
    # collapse flag: modal predicted caption share >= 0.5 on that dataset (from frozen predictions)
    col = {}
    for _, r in t.iterrows():
        preds = [json.loads(l)['prediction'] for l in open(ROOT / 'results' / 'baselines' / r.system / f'{r.dataset}_test.jsonl')]
        from collections import Counter
        col[(r.dataset, r.system)] = Counter(preds).most_common(1)[0][1] / len(preds)
    t['modal_caption_share'] = [col[(a, b)] for a, b in zip(t.dataset, t.system)]
    t['collapsed'] = t.modal_caption_share >= 0.5
    t.to_csv(R / 'system_summary.csv', index=False)
    corr = []
    def sp(sub, tag, ds):
        for m in METRICS[1:]:
            if len(sub) >= 3 and sub[m].nunique() > 1 and sub['ricce'].nunique() > 1:
                rho = spearmanr(sub[m], sub['ricce']).statistic
            else:
                rho = None
            corr.append(dict(dataset=ds, subset=tag, metric=m, n=len(sub), spearman_vs_CE_F=rho))
    for ds in ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC']:
        sub = t[t.dataset == ds]
        sp(sub, 'all_systems', ds); sp(sub[~sub.collapsed], 'non_collapsed', ds)
    sp(t, 'all_systems', 'POOLED_cells'); sp(t[~t.collapsed], 'non_collapsed', 'POOLED_cells')
    c = pd.DataFrame(corr); c.to_csv(R / 'system_spearman.csv', index=False)
    ranks = {}
    for ds, sub in t.groupby('dataset'):
        ranks[ds] = {m: list(sub.sort_values(m, ascending=False).system) for m in METRICS}
    json.dump(ranks, open(R / 'system_rankings.json', 'w'), indent=1)
    return t, c, ranks


if __name__ == '__main__':
    what = sys.argv[1:] or ['perturbation', 'diagnostics', 'systems']
    if 'perturbation' in what: print(perturbation().query("dataset=='ALL'").round(3).to_string())
    if 'diagnostics' in what: print(diagnostics().round(3).to_string())
    if 'systems' in what:
        t, c, r = systems(); print(t.round(3).to_string()); print(c.round(3).to_string()); print(json.dumps(r, indent=1))
