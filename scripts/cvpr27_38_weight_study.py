"""Field-weight study (2026-11-09). Exploratory: designed after the human ratings had been analysed. Nothing here
selects or changes a weight; the evaluator and every number in the paper keep the default weights.

The evaluator averages its field scores with fixed weights (configs/project.yaml, unchanged since the first commit):
change state 1.5, action 1.3, object/source/target 1.0, location 0.7, quantity 0.5. Because the average is a weighted
mean, only the ratios of the weights matter. Tab. ablw (scripts/v3_ablations.py) checks 13 settings and 7
leave-one-field-out variants on RICCE-Bench and the eight-system ranking. This script adds:
  1. Wider fixed settings, scored exactly like Tab. ablw (RICCE-Bench core / validity / near>sibling, mean CE-F,
     Kendall tau of the seven LEVIR-CC systems): each field's weight halved and doubled (14) and the priority order
     reversed (the default values assigned to the fields in the opposite order).
  2. Agreement with people under every fixed setting (default, the 20 of Tab. ablw, the 15 of step 1): Spearman with
     the consensus Overall rating on the paper's matched sets (all captions, model outputs, constructed captions), and
     the difference to the default by the paired cluster bootstrap of scripts/cvpr27_05_human_reanalysis.py
     (image pairs as clusters, B = 10,000, seed 20260923, shared resamples).
  3. Random weights: N_RANDOM vectors, each weight independently log-uniform in [1/4, 4] (seed 20261109): the spread
     of the Spearman and the default's rank among them; and five-fold cross-validation over image pairs: on four folds
     take the vector with the highest Spearman (the default competes), score the fifth fold with it; the out-of-fold
     column is compared to the default by the same bootstrap.
  4. How severe people judge each single-field error: each constructed caption changes one thing in an original
     caption. Per error type: mean consensus Overall and the mean rating on the axis the error targets (cluster
     bootstrap CI), and the mean CE-F under the default and under equal weights, next to the weight of the field.
Scores use the captions, references, parses and aggregation of scripts/cvpr27_04_variants_human.py (hard cases
against the first reference, model outputs averaged over the scorable references); checks: the default weights
reproduce the key's CE-F of every caption, the paper's Spearman and the Tab. ablw row of the full evaluator.
Outputs -> submission_cvpr27_new_version/14_WEIGHT_STUDY/
"""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# evaluator variants and the RICCE-Bench / systems data of v3_ablations.py (everything but its __main__ block)
ABL = ROOT / 'scripts' / 'v3_ablations.py'
ns = {'__file__': str(ABL), '__name__': 'v3_ablations_defs'}
exec(compile(ABL.read_text(), str(ABL), 'exec'), ns)
VARS, Variant, score, W, MAIN = ns['VARS'], ns['Variant'], ns['score'], ns['W'], ns['MAIN']
FIELDS = list(W)

S = ROOT / 'submission_cvpr27_new_version'
OUT = S / '14_WEIGHT_STUDY'; OUT.mkdir(parents=True, exist_ok=True)
B, SEED = 10000, 20260923                  # as scripts/cvpr27_05_human_reanalysis.py
N_RANDOM, RSEED, N_FOLDS = 500, 20261109, 5
NAME = {'change_state': 'change state', 'action': 'action', 'object': 'object', 'source': 'source', 'target': 'target',
        'location': 'location', 'quantity': 'quantity'}

# ------------------------------------------------------------------ fixed settings
NEW = []
for ax in FIELDS:
    for f, tag in ((0.5, 'half'), (2.0, 'double')):
        NEW.append(Variant(f'W_{ax}_{tag}', f'weight[{ax}] x{f:g}', group='weights_new', weights={**W, ax: W[ax] * f}))
order = sorted(FIELDS, key=lambda k: -W[k])                       # change_state, action, object, source, target, location, quantity
rev = dict(zip(order, sorted((W[k] for k in order))))            # largest weight to the field with the smallest, ...
NEW.append(Variant('W_reversed', 'priority order reversed: ' + ', '.join(f'{k} {v:g}' for k, v in rev.items()), group='weights_new', weights=rev))
for v in NEW:
    VARS[v.name] = v
TAB_ABLW = [n for n in VARS if VARS[n].group in ('weights', 'axis_loo')]
FIXED = ['A0_full'] + TAB_ABLW + [v.name for v in NEW]


def run_one(name):
    return ns['run'](name)


# ------------------------------------------------------------------ 1. RICCE-Bench and the seven systems
def bench_rows(res):
    from ricce.evaluation.ordering import build_constraints, pairwise_ordering_accuracy
    items = ns['items']
    cons = build_constraints(items)
    core = cons[cons.suite == 'core']; ext = cons[cons.suite == 'extended']
    GOOD = ['P0_paraphrase', 'P1_near_ancestor']; BAD = ['P4_action_reversal', 'P9_change_state_reversal', 'negation_flip', 'source_target_swap']
    a0sys = res['A0_full'][1]
    rows = []
    for name, (bf, sysf, nat) in res.items():          # same summary as the __main__ block of v3_ablations.py
        s = pd.Series(bf, index=items.sample_id.values)
        o = pairwise_ordering_accuracy(s, core, n_boot=500, seed=0)
        e = pairwise_ordering_accuracy(s, ext, n_boot=200, seed=0)
        d = pd.DataFrame({'source_id': items.source_id.values, 'ptype': items.perturbation_type.values, 'F': bf})
        piv = d.pivot_table(index='source_id', columns='ptype', values='F'); w = n = 0
        for gd in GOOD:
            for bd in BAD:
                x = piv[[gd, bd]].dropna(); w += int((x[gd] > x[bd] + 1e-9).sum()); n += len(x)
        rows.append(dict(variant=name, group=VARS[name].group, desc=VARS[name].desc, weights=json.dumps(VARS[name].weights or W),
                         core=o['accuracy_strict'], core_lo=o['accuracy_strict_ci95'][0], core_hi=o['accuracy_strict_ci95'][1],
                         validity=w / n, near_gt_sibling=e['accuracy_strict'], mean_cef_bench=float(np.nanmean(bf)),
                         tau_systems=float(stats.kendalltau([a0sys[x] for x in MAIN], [sysf[x] for x in MAIN])[0]),
                         order=' > '.join(sorted(MAIN, key=lambda x: -sysf[x])), **{f'sys_{k}': v for k, v in sysf.items()}))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ human-study captions
def human_setup():
    from ricce.evaluation.parse_store import ParseStore
    spec = importlib.util.spec_from_file_location('cvpr27_03', ROOT / 'scripts' / 'cvpr27_03_extend_human_metrics.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    J = m.jobs(); J = J[J.kind == 'item'].reset_index(drop=True)
    store = ParseStore(ROOT / 'results_v3' / 'human_eval' / 'key_rescoring' / 'parse_cache_human_eval.json')
    ev = {t: store.events(t, 'multi') for t in set(J.candidate) | {g for rs in J.references for g in rs}}
    F = pd.read_csv(S / '03_HUMAN_REANALYSIS' / 'item_frame.csv').set_index('item_id').loc[J.item_id].reset_index()
    return J, ev, F


def human_scores(J, ev, w):
    v = Variant('w', '', weights=w)
    out = np.full(len(J), np.nan)
    for i, (c, refs) in enumerate(zip(J.candidate, J.references)):
        x = [score(v, ev[c], ev[g]) for g in refs]; x = [y for y in x if y == y]   # mean over the scorable references
        if x:
            out[i] = np.mean(x)
    return out


def picks(clusters, B, seed):                       # as scripts/cvpr27_05_human_reanalysis.py
    cl = pd.factorize(np.asarray(clusters))[0]
    idx = [np.where(cl == c)[0] for c in range(cl.max() + 1)]
    rng = np.random.default_rng(seed)
    return [np.concatenate([idx[j] for j in rng.integers(0, len(idx), len(idx))]) for _ in range(B)]


def spearman_matrix(X, y):                          # as scripts/cvpr27_05_human_reanalysis.py
    rx = stats.rankdata(X, axis=0); ry = stats.rankdata(y)
    rx = rx - rx.mean(0); ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum(0) * (ry ** 2).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        return (rx * ry[:, None]).sum(0) / den


def boot(X, y, clusters):
    est = spearman_matrix(X, y)
    bm = np.vstack([spearman_matrix(X[ii], y[ii]) for ii in picks(clusters, B, SEED)])
    return est, bm


def p_boot(v):                                      # as scripts/cvpr27_05_human_reanalysis.py
    v = v[~np.isnan(v)]
    tail = min((v <= 0).sum(), (v >= 0).sum())
    return max(min(1.0, 2 * (tail + 1) / (len(v) + 1)), 1.0 / (len(v) + 1))


if __name__ == '__main__':
    import multiprocessing as mp
    checks = {}
    t0 = time.time()
    names = ['A0_full'] + [v.name for v in NEW]
    res = {}
    with mp.get_context('fork').Pool(len(names)) as pool:
        for name, bf, sysf, nat, dt in pool.imap_unordered(run_one, names):
            res[name] = (bf, sysf, nat); print(name, f'{dt:.0f}s', flush=True)
    BR = bench_rows({n: res[n] for n in names})
    old = pd.read_csv(ROOT / 'results_v3' / 'ablations' / 'ablation_summary.csv').set_index('variant')
    BR = pd.concat([BR, old.loc[TAB_ABLW].reset_index()[['variant', 'group', 'desc', 'core', 'core_lo', 'core_hi', 'validity',
                                                          'near_gt_sibling', 'mean_cef_bench', 'tau_systems', 'order']]
                    .assign(source='results_v3/ablations/ablation_summary.csv (Tab. ablw)')], ignore_index=True)
    BR['source'] = BR['source'].fillna('this script')
    a0 = BR.set_index('variant').loc['A0_full']
    checks['bench_A0_equals_ablation_summary'] = {c: [float(a0[c]), float(old.loc['A0_full', c])]
                                                  for c in ('core', 'validity', 'near_gt_sibling', 'mean_cef_bench', 'tau_systems')}
    checks['bench_A0_max_abs_diff'] = max(abs(a - b) for a, b in checks['bench_A0_equals_ablation_summary'].values())
    BR.to_csv(OUT / 'bench_fixed_settings.csv', index=False)
    print(f'bench: {time.time() - t0:.0f}s', flush=True)

    # ---- 2. agreement with people under every fixed setting
    J, ev, F = human_setup()
    SC = pd.DataFrame({n: human_scores(J, ev, VARS[n].weights or W) for n in FIXED})
    SC.insert(0, 'item_id', J.item_id)
    SC.to_csv(OUT / 'human_item_scores_fixed.csv', index=False)
    both = F.ce_f1.notna() & SC.A0_full.notna()
    checks['human_A0_vs_key_n'] = int(both.sum())
    checks['human_A0_vs_key_max_abs_diff'] = float((F.ce_f1[both] - SC.A0_full[both]).abs().max())
    checks['human_nan_mismatch'] = int((F.ce_f1.isna() != SC.A0_full.isna()).sum())
    SUB = {'all': F.h_overall.notna() & F.ce_f1.notna(), 'real': (F.subset == 'real') & F.h_overall.notna() & F.ce_f1.notna(),
           'constructed': (F.subset == 'constructed') & F.h_overall.notna() & F.ce_f1.notna()}
    mn = pd.read_csv(S / '03_HUMAN_REANALYSIS' / 'human_matched_n.csv').set_index('subset')
    checks['matched_n'] = {s: [int(m.sum()), int(mn.loc[s, 'n_matched'])] for s, m in SUB.items()}
    mm = pd.read_csv(S / '03_HUMAN_REANALYSIS' / 'human_main_metrics.csv')
    rows, BOOT = [], {}
    for s, m in SUB.items():
        d = F[m]; X = SC.loc[m.values, FIXED].to_numpy(float); y = d.h_overall.to_numpy(float)
        est, bm = boot(X, y, d.cluster.values); BOOT[s] = bm
        paper = mm[(mm.subset == s) & (mm.metric == 'ce_f1')].spearman
        checks[f'spearman_A0_{s}'] = [float(est[0]), float(paper.iloc[0]) if len(paper) else None]
        for j, n in enumerate(FIXED):
            dlt = bm[:, j] - bm[:, 0]
            rows.append(dict(subset=s, variant=n, group=VARS[n].group, desc=VARS[n].desc, n=len(d), n_clusters=d.cluster.nunique(),
                             spearman=est[j], ci_lo=np.percentile(bm[:, j], 2.5), ci_hi=np.percentile(bm[:, j], 97.5),
                             diff_vs_default=est[j] - est[0], diff_lo=np.percentile(dlt, 2.5), diff_hi=np.percentile(dlt, 97.5),
                             p_boot=p_boot(dlt) if j else np.nan))
    HF = pd.DataFrame(rows); HF.to_csv(OUT / 'human_fixed_settings.csv', index=False)
    print(f'human fixed: {time.time() - t0:.0f}s', flush=True)

    # ---- 3. random weights and cross-validated selection
    rng = np.random.default_rng(RSEED)
    RW = np.exp(rng.uniform(np.log(0.25), np.log(4.0), size=(N_RANDOM, len(FIELDS))))
    CAND = [dict(W)] + [dict(zip(FIELDS, r)) for r in RW]                  # column 0 = default
    RS = np.column_stack([SC.A0_full.to_numpy()] + [human_scores(J, ev, w) for w in CAND[1:]])
    print(f'random: {time.time() - t0:.0f}s', flush=True)
    rr = []
    for i, w in enumerate(CAND):
        r = dict(vector=i, is_default=i == 0, **{f'w_{k}': w[k] for k in FIELDS})
        for s, m in SUB.items():
            r[f'spearman_{s}'] = float(spearman_matrix(RS[m.values, i:i + 1], F.h_overall[m].to_numpy(float))[0])
        rr.append(r)
    RR = pd.DataFrame(rr); RR.to_csv(OUT / 'human_random_weights.csv', index=False)
    rsum = {}
    for s in SUB:
        v = RR[~RR.is_default][f'spearman_{s}']; d0 = float(RR.loc[0, f'spearman_{s}'])
        rsum[s] = dict(default=d0, min=float(v.min()), p05=float(v.quantile(.05)), median=float(v.median()), p95=float(v.quantile(.95)),
                       max=float(v.max()), share_random_above_default=float((v > d0).mean()), best_vector=int(v.idxmax()),
                       best_weights={k: float(RR.loc[v.idxmax(), f'w_{k}']) for k in FIELDS})
    # five-fold cross-validation over image pairs on the 'all' set
    m = SUB['all'].values; d = F[m].reset_index(drop=True); X = RS[m]; y = d.h_overall.to_numpy(float)
    cl = d.cluster.unique(); fold_of = dict(zip(np.random.default_rng(RSEED).permutation(cl), np.arange(len(cl)) % N_FOLDS))
    fold = d.cluster.map(fold_of).to_numpy()
    oof = np.full(len(d), np.nan); folds = []
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        rho_tr = spearman_matrix(X[tr], y[tr]); best = int(np.nanargmax(rho_tr))
        oof[te] = X[te, best]
        folds.append(dict(fold=k, n_train=int(tr.sum()), n_test=int(te.sum()), selected_vector=best, selected_is_default=best == 0,
                          train_spearman_selected=float(rho_tr[best]), train_spearman_default=float(rho_tr[0]),
                          test_spearman_selected=float(spearman_matrix(X[te, best:best + 1], y[te])[0]),
                          test_spearman_default=float(spearman_matrix(X[te, :1], y[te])[0]),
                          **{f'w_{f}': CAND[best][f] for f in FIELDS}))
    CV = pd.DataFrame(folds); CV.to_csv(OUT / 'human_cv_folds.csv', index=False)
    est, bm = boot(np.column_stack([X[:, 0], oof]), y, d.cluster.values)
    dlt = bm[:, 1] - bm[:, 0]
    cvsum = dict(n=len(d), default=float(est[0]), cross_validated=float(est[1]), diff=float(est[1] - est[0]),
                 diff_lo=float(np.percentile(dlt, 2.5)), diff_hi=float(np.percentile(dlt, 97.5)), p_boot=p_boot(dlt),
                 mean_test_selected=float(CV.test_spearman_selected.mean()), mean_test_default=float(CV.test_spearman_default.mean()))
    print(f'cv: {time.time() - t0:.0f}s', flush=True)

    # ---- 4. severity of single-field errors as judged by people
    TARGET = {'P0_original': (None, None), 'P0_paraphrase': (None, None), 'P1_ancestor': ('object', 'specificity'),
              'P2_distant_ancestor': ('object', 'specificity'), 'P3_sibling': ('object', 'object'), 'P4_action_reversal': ('action', 'action'),
              'P5_location_swap': ('location', 'location'), 'P6_count_corruption': ('quantity', 'quantity'),
              'P7_event_omission': (None, 'completeness'), 'P8_hallucination': (None, 'hallucination'),
              'P9_change_state_reversal': ('change_state', 'change_nochange')}
    C = F[SUB['constructed'].values].copy(); C['ce_default'] = SC.loc[SUB['constructed'].values, 'A0_full'].values
    C['ce_equal'] = SC.loc[SUB['constructed'].values, 'W_equal'].values
    sv = []
    for t, (fld, axis) in TARGET.items():
        g = C[C.perturbation_type == t]
        r = dict(error_type=t, field=fld, field_weight=W.get(fld) if fld else np.nan, n=len(g), n_clusters=g.cluster.nunique(),
                 human_overall=g.h_overall.mean(), human_overall_sd=g.h_overall.std(),
                 human_target_axis=axis, human_target_rating=g[f'h_{axis}'].mean() if axis else np.nan,
                 ce_f1_default=g.ce_default.mean(), ce_f1_equal_weights=g.ce_equal.mean())
        sv.append(r)
    SV = pd.DataFrame(sv)
    # bootstrap CI of each type's mean Overall (resampling that type's items, nearly one item per image pair; B = 10,000)
    rb = np.random.default_rng(SEED)
    for i, t in enumerate(SV.error_type):
        x = C[C.perturbation_type == t].h_overall.to_numpy(float)
        bm_ = x[rb.integers(0, len(x), (B, len(x)))].mean(1)
        SV.loc[i, 'human_overall_lo'], SV.loc[i, 'human_overall_hi'] = np.percentile(bm_, [2.5, 97.5])
    SV.to_csv(OUT / 'severity_by_error_type.csv', index=False)
    fe = SV.dropna(subset=['field_weight'])
    fe = fe[fe.error_type.isin(['P3_sibling', 'P4_action_reversal', 'P5_location_swap', 'P6_count_corruption', 'P9_change_state_reversal'])]
    sev = dict(types=list(fe.error_type), weights=list(fe.field_weight), human_overall=list(fe.human_overall),
               human_order_most_severe_first=list(fe.sort_values('human_overall').error_type),
               weight_order_largest_first=list(fe.sort_values('field_weight', ascending=False).error_type),
               kendall_tau_weight_vs_human_drop=float(stats.kendalltau(fe.field_weight, -fe.human_overall)[0]))

    summary = dict(date='2026-11-09', status='exploratory (after the human ratings were analysed); no weight selected or changed',
                   default_weights=W, n_random=N_RANDOM, random_range='each weight log-uniform in [1/4, 4]', seeds=dict(bootstrap=SEED, random=RSEED),
                   bootstrap=f'cluster (image pair), B={B}', random_weights=rsum, cross_validation=cvsum, severity=sev, checks=checks,
                   command='python scripts/cvpr27_38_weight_study.py')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    pd.set_option('display.width', 250)
    print(BR[['variant', 'core', 'validity', 'near_gt_sibling', 'mean_cef_bench', 'tau_systems']].round(3).to_string())
    print(HF.pivot(index='variant', columns='subset', values='spearman').loc[FIXED].round(3).to_string())
    print(HF[HF.subset == 'all'][['variant', 'diff_vs_default', 'diff_lo', 'diff_hi']].round(3).to_string())
    print(SV.round(3).to_string())
    print(json.dumps({k: summary[k] for k in ('random_weights', 'cross_validation', 'severity', 'checks')}, indent=1, default=float))
    print(f'DONE {time.time() - t0:.0f}s')
