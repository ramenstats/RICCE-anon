"""v3: every real-system number under the FINAL evaluator
(ricce/metrics/final.py), from the shared parse cache.

  0. basis check: PREVIOUS_CONFIG on the cache reproduces
     results_v2/system_benchmark_v2/summary.csv (LEVIR-CC, 7 systems)
     -> the cache is byte-equivalent to the direct parses used before;
  1. system benchmark: 7 systems x LEVIR-CC, 4 x SECOND-CC / Dubai-CC;
     per-image mean over references; 10,000-resample percentile CIs;
     paired bootstrap + Holm per dataset; Spearman vs BLEU-4 / CIDEr-D;
  2. learning runs (SFT / step-matched SFT / swap-augmented SFT seeds,
     DPO / TCPO variants) on LEVIR-CC; swap-vs-control paired bootstrap;
  3. cross-dataset: parse-clean subset AND stratification by whether the
     references describe a change, plus each system's no-change output share;
  4. natural multi-event pairs: final CE-F and the flat baselines
     (Kiritchenko micro, HCM, TaxAware axis-flat) on identical parses.
Outputs -> results_v3/systems/."""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl
from ricce.benchmarks.env import load_taxes, load_weights
from ricce.evaluation.parse_store import ParseStore
from ricce.metrics.final import FINAL_CONFIG, PREVIOUS_CONFIG
from ricce.metrics.event_v2 import ricce_scores_v2

OUT = ROOT / 'results_v3' / 'systems'; (OUT / 'per_image').mkdir(parents=True, exist_ok=True)
T, W = load_taxes(), load_weights()
store = ParseStore(ROOT / 'results_v3' / 'parse_cache.json')
_ev = {}
def ev(t):
    if t not in _ev:
        _ev[t] = store.events(t, 'multi')
    return _ev[t]

MAIN = ['qwen', 'sft_seed42', 'swmcc', 'chg2cap', 'rsiccformer', 'promptcc', 'rscama']
DS = ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC']
SEEDS = [1, 42, 123]
LEARN = {'sft': [f'sft_seed{s}' for s in SEEDS], 'sft2x': [f'sft2x_seed{s}' for s in SEEDS], 'swapsft': [f'swapsft_seed{s}' for s in SEEDS],
         'dpo_original': ['dpo_seed42'], 'tcpo_original': ['tcpo_seed42'],
         'dpo_stab': [f'dpo_stab_seed{s}' for s in SEEDS], 'dpo_stab_kl': [f'dpo_reg_seed{s}' for s in SEEDS],
         'tcpo_true': [f'tcpo_v3_seed{s}' for s in SEEDS], 'tcpo_clipped': [f'tcpo_clipped_severity_seed{s}' for s in SEEDS],
         'tcpo_normalized': [f'tcpo_normalized_severity_seed{s}' for s in SEEDS], 'tcpo_random': [f'tcpo_random_severity_seed{s}' for s in SEEDS],
         'tcpo_uniform': [f'tcpo_no_severity_seed{s}' for s in SEEDS], 'tcpo_inverse': [f'tcpo_inverse_severity_seed{s}' for s in SEEDS],
         'tcpo_curriculum': ['tcpo_curriculum_severity_seed42']}
STAB25 = [s for k in ('dpo_stab', 'dpo_stab_kl', 'tcpo_true', 'tcpo_clipped', 'tcpo_normalized', 'tcpo_random', 'tcpo_uniform', 'tcpo_inverse', 'tcpo_curriculum') for s in LEARN[k]]
assert len(STAB25) == 25
man = {ds: {r['image_id']: r for r in read_jsonl(ROOT / 'artifacts' / 'manifests' / f'{ds}.jsonl') if r.get('split') == 'test'} for ds in DS}
rng = np.random.default_rng(0)
B = 10000


def per_image(ds, sy, cfg):
    pf = ROOT / 'results' / 'baselines' / sy / f'{ds}_test.jsonl'
    if not pf.exists():
        return None
    recs = []
    for r in read_jsonl(pf):
        if r['image_id'] not in man[ds]:
            continue
        pe = ev(str(r['prediction']))
        vals = []
        for g in man[ds][r['image_id']]['references']:
            s = ricce_scores_v2(pe, ev(g), T, W, **cfg)
            vals.append((np.nan, np.nan, np.nan) if s['not_scorable'] else (s['ce_precision'], s['ce_recall'], s['ce_f1']))
        v = np.array(vals, float)
        m = [np.nanmean(v[:, k]) if np.isfinite(v[:, k]).any() else np.nan for k in range(3)]
        recs.append((r['image_id'], *m, sum(np.isnan(v[:, 2]))))
    return pd.DataFrame(recs, columns=['image_id', 'ce_p', 'ce_r', 'ce_f', 'n_ref_not_scorable']).dropna(subset=['ce_p', 'ce_r', 'ce_f'])


def ci(x):
    x = np.asarray(x, float); m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


res = {}
# ------------------------------------------------------------------ 0 basis check
prev = pd.read_csv(ROOT / 'results_v2' / 'system_benchmark_v2' / 'summary.csv')
prev = prev[prev.dataset == 'LEVIR_CC'].set_index('system')
mx = 0.0
for sy in MAIN:
    d = per_image('LEVIR_CC', sy, PREVIOUS_CONFIG)
    for k in ('ce_p', 'ce_r', 'ce_f'):
        mx = max(mx, abs(d[k].mean() - prev.loc[sy, k]))
assert mx < 1e-9, mx
res['basis_check_max_abs_diff_vs_previous_summary'] = mx
print('basis check ok', mx, flush=True)

# ------------------------------------------------------------------ 1 system benchmark
conv = {}
rows, per = [], {}
for ds in DS:
    per[ds] = {}
    for sy in MAIN:
        d = per_image(ds, sy, FINAL_CONFIG)
        if d is None:
            continue
        d.to_csv(OUT / 'per_image' / f'{ds}__{sy}.csv', index=False)
        per[ds][sy] = d.set_index('image_id')
        mj = ROOT / 'results' / 'E08' / ds / sy / 'metrics.json'
        c = json.load(open(mj)) if mj.exists() else {}
        lo, hi = ci(d.ce_f.values)
        rows.append(dict(dataset=ds, system=sy, n=len(d), ce_p=d.ce_p.mean(), ce_r=d.ce_r.mean(), ce_f=d.ce_f.mean(), ce_f_lo=lo, ce_f_hi=hi,
                         n_images_with_unscorable_ref=int((d.n_ref_not_scorable > 0).sum()), bleu4=c.get('bleu4'), cider_d=c.get('cider_d_corpus')))
        print(ds, sy, round(d.ce_f.mean(), 4), flush=True)
S = pd.DataFrame(rows); S.to_csv(OUT / 'summary.csv', index=False)
pw_all = []
for ds in DS:
    pairs = []
    for a, b in itertools.combinations(sorted(per[ds]), 2):
        j = per[ds][a].index.intersection(per[ds][b].index)
        d = (per[ds][a].loc[j, 'ce_f'] - per[ds][b].loc[j, 'ce_f']).values
        bm = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
        p = max(2 * min((bm <= 0).mean(), (bm >= 0).mean()), 1.0 / (B + 1))
        pairs.append(dict(dataset=ds, a=a, b=b, n=len(d), mean_diff=float(d.mean()), ci_lo=float(np.percentile(bm, 2.5)), ci_hi=float(np.percentile(bm, 97.5)), p_raw=p))
    order = np.argsort([q['p_raw'] for q in pairs]); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(pairs) - rank) * pairs[i]['p_raw'])); pairs[i]['p_holm'] = run
    pw_all += pairs
PW = pd.DataFrame(pw_all); PW.to_csv(OUT / 'pairwise.csv', index=False)
corr = {}
for ds in DS:
    s = S[S.dataset == ds]
    corr[ds] = dict(n_systems=len(s))
    for k in ('bleu4', 'cider_d'):
        r, p = spearmanr(s.ce_f, s[k]); corr[ds][f'spearman_vs_{k}'] = float(r); corr[ds][f'p_vs_{k}'] = float(p)
    lv = PW[PW.dataset == ds]
    corr[ds]['n_pairs'] = len(lv); corr[ds]['n_sig_holm'] = int((lv.p_holm < 0.05).sum())
res['rank_correlation_with_conventional'] = corr
lv = S[S.dataset == 'LEVIR_CC'].set_index('system')
res['levir_order'] = list(lv.sort_values('ce_f', ascending=False).index)
res['tau_final_vs_previous_levir'] = float(kendalltau([lv.loc[s, 'ce_f'] for s in MAIN], [prev.loc[s, 'ce_f'] for s in MAIN])[0])
print(json.dumps(corr, indent=1), flush=True)

# ------------------------------------------------------------------ 2 learning runs
lrows, lper = [], {}
for grp, systems in LEARN.items():
    for sy in systems:
        d = per_image('LEVIR_CC', sy, FINAL_CONFIG)
        if d is None:
            print('missing', sy); continue
        d.to_csv(OUT / 'per_image' / f'LEVIR_CC__{sy}.csv', index=False)
        lper[sy] = d.set_index('image_id')
        lrows.append(dict(group=grp, system=sy, n=len(d), ce_p=d.ce_p.mean(), ce_r=d.ce_r.mean(), ce_f=d.ce_f.mean()))
L = pd.DataFrame(lrows); L.to_csv(OUT / 'learning_runs.csv', index=False)
lg = L.groupby('group').ce_f.agg(['mean', 'std', 'count']); lg.to_csv(OUT / 'learning_groups.csv')
res['learning_groups'] = lg.to_dict('index')
st = L[L.system.isin(STAB25)].ce_f
res['stabilised_preference_25'] = dict(n=int(len(st)), min_group_mean=float(lg.loc[[k for k in LEARN if k in ('dpo_stab', 'dpo_stab_kl', 'tcpo_true', 'tcpo_clipped', 'tcpo_normalized', 'tcpo_random', 'tcpo_uniform', 'tcpo_inverse', 'tcpo_curriculum')], 'mean'].min()),
                                       max_group_mean=float(lg.loc[[k for k in ('dpo_stab', 'dpo_stab_kl', 'tcpo_true', 'tcpo_clipped', 'tcpo_normalized', 'tcpo_random', 'tcpo_uniform', 'tcpo_inverse', 'tcpo_curriculum')], 'mean'].max()),
                                       mean=float(st.mean()), std=float(st.std()))


def seedavg(g):
    return pd.concat([lper[s].ce_f for s in LEARN[g]], axis=1).mean(axis=1)


for a, b in [('swapsft', 'sft2x'), ('swapsft', 'sft')]:
    x, y = seedavg(a), seedavg(b); j = x.index.intersection(y.index)
    d = (x.loc[j] - y.loc[j]).values; bm = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    res[f'paired_{a}_minus_{b}'] = dict(diff=float(d.mean()), ci95=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))], n_images=len(d))
print(json.dumps({k: v for k, v in res.items() if k.startswith('paired') or k.startswith('stab')}, indent=1), flush=True)

# ------------------------------------------------------------------ 3 cross-dataset
def stratum(e):
    if e.change_state == 'no_change':
        return 'no_change'
    core = int(e.object is not None and e.object != T['object'].root) + int(e.action is not None and e.action != T['action'].root)
    return {2: 'full', 1: 'partial', 0: 'failed'}[core]


def ev_stratum(es):
    if not es:
        return 'failed'
    if all(e.change_state == 'no_change' for e in es):
        return 'no_change'
    s = [stratum(e) for e in es]
    if all(x in ('full', 'no_change') for x in s):
        return 'full'
    return 'failed' if all(x == 'failed' for x in s) else 'partial'


XS = ['sft_seed42', 'chg2cap', 'swmcc', 'qwen']
xrows = []
for ds in DS:
    for sy in XS:
        if sy not in per[ds]:
            continue
        preds = {r['image_id']: str(r['prediction']) for r in read_jsonl(ROOT / 'results' / 'baselines' / sy / f'{ds}_test.jsonl')}
        for iid, row in per[ds][sy].iterrows():
            refs = man[ds][iid]['references']
            rs = [ev_stratum(ev(g)) for g in refs]
            ps = ev_stratum(ev(preds[iid]))
            xrows.append(dict(dataset=ds, system=sy, image_id=iid, ce_f=row.ce_f, pred_stratum=ps,
                              ref_clean=all(x in ('full', 'no_change') for x in rs),
                              ref_nochange_frac=float(np.mean([x == 'no_change' for x in rs]))))
X = pd.DataFrame(xrows); X.to_csv(OUT / 'cross_dataset_per_image.csv', index=False)
X['ref_changed'] = X.ref_nochange_frac < 0.5
X['clean'] = X.ref_clean & X.pred_stratum.isin(['full', 'no_change'])
xs = {}
for (ds, sy), g in X.groupby(['dataset', 'system']):
    ch = g[g.ref_changed]; nc = g[~g.ref_changed]
    xs[f'{ds}|{sy}'] = dict(n=len(g), ce_f=float(g.ce_f.mean()), n_clean=int(g.clean.sum()), ce_f_clean=float(g[g.clean].ce_f.mean()) if g.clean.any() else None,
                            clean_subset_ref_changed_share=float(g[g.clean].ref_changed.mean()) if g.clean.any() else None,
                            ref_changed_share=float(g.ref_changed.mean()), pred_nochange_share=float((g.pred_stratum == 'no_change').mean()),
                            pred_nochange_share_on_changed=float((ch.pred_stratum == 'no_change').mean()) if len(ch) else None,
                            ce_f_changed=float(ch.ce_f.mean()) if len(ch) else None, n_changed=len(ch),
                            ce_f_unchanged=float(nc.ce_f.mean()) if len(nc) else None, n_unchanged=len(nc),
                            ce_f_changed_clean=float(ch[ch.clean].ce_f.mean()) if ch.clean.any() else None, n_changed_clean=int(ch.clean.sum()))
res['cross_dataset'] = xs
print(pd.DataFrame(xs).T[['n', 'ce_f', 'ce_f_clean', 'clean_subset_ref_changed_share', 'ref_changed_share', 'pred_nochange_share', 'ce_f_changed', 'ce_f_unchanged', 'ce_f_changed_clean', 'n_changed_clean']].round(3).to_string(), flush=True)

# ------------------------------------------------------------------ 4 natural multi-event pairs
from ricce.baselines.registry import BaselineContext, score_pair
ctx = BaselineContext()
NM = ['qwen', 'sft_seed42', 'swmcc', 'chg2cap']
levir_all = list(read_jsonl(ROOT / 'artifacts' / 'manifests' / 'LEVIR_CC.jsonl'))
n_multi = n_tot = 0
for r in levir_all:
    for g in r.get('references', []):
        if g and g.strip():
            n_tot += 1; n_multi += len(ev(g)) >= 2
mref = {iid: [g for g in r['references'] if len(ev(g)) >= 2] for iid, r in man['LEVIR_CC'].items()}
mref = {k: v for k, v in mref.items() if v}
nrows = []
for sy in NM:
    preds = {r['image_id']: str(r['prediction']) for r in read_jsonl(ROOT / 'results' / 'baselines' / sy / 'LEVIR_CC_test.jsonl')}
    for iid, gs in mref.items():
        p = preds[iid]
        for g in gs:
            for t in (p, g):
                ctx._cache.setdefault(t, ev(t))
            s = ricce_scores_v2(ev(p), ev(g), T, W, **FINAL_CONFIG)
            row = dict(system=sy, image_id=iid, prediction=p, reference=g, cef=np.nan if s['not_scorable'] else s['ce_f1'],
                       n_pred_events=len(ev(p)), n_ref_events=len(ev(g)))
            for b in ('kiritchenko_micro', 'hcm', 'taxaware_axisflat'):
                row[b] = score_pair(b, p, g, ctx)['F']
            nrows.append(row)
N = pd.DataFrame(nrows); N.to_csv(OUT / 'natural_multievent_pairs.csv', index=False)
nm = N.groupby('system')[['cef', 'kiritchenko_micro', 'hcm', 'taxaware_axisflat']].mean()
res['natural_multievent'] = dict(n_multi_event_refs_levir_all_splits=int(n_multi), n_refs_levir_all_splits=int(n_tot),
                                 n_test_images_with_multi_ref=len(mref), pairs_per_system=N.groupby('system').size().to_dict(),
                                 mean=nm.to_dict('index'))
print(nm.round(3).to_string(), flush=True)
json.dump(res, open(OUT / 'summary.json', 'w'), indent=1, default=float)
print('DONE')
