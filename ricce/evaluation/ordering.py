"""Pairwise semantic-ordering evaluation for ANY metric.

A metric is either (a) a callable f(candidate, reference) -> float (use
`score_items_with_callable`) or (b) a precomputed per-item score column keyed by
`sample_id` (pandas Series / dict). Higher score = better for every metric.

Constraints are built per source caption (`source_id`); bootstrap CIs resample
SOURCE CAPTIONS (clusters), never individual constraints.
Tie semantics (reported explicitly):
  * 'gt'  constraint  hi > lo : strict variant counts ties as VIOLATED (0);
                                tie-credit-half variant counts ties as 0.5.
  * 'ge'  constraint  hi >= lo: ties satisfy the constraint in BOTH variants.
  * 'eq'  invariance          : |hi-lo| <= tol satisfies in both variants.
"""
import math
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

FAM = dict(ident='P0_identity', para='P0_paraphrase', near='P1_near_ancestor', far='P2_distant_ancestor',
           sib='P3_sibling_substitution', act='P4_action_reversal', loc='P5_location_swap', cnt='P6_count_corruption',
           om='P7_event_omission', hal='P8_event_hallucination', cs='P9_change_state_reversal',
           desc='unsupported_descendant_specificity', st='source_target_swap', neg='negation_flip',
           reo='multi_event_reorder', bind='multi_event_source_target_ambiguity')

# (hi family, lo family, kind, suite)
CONSTRAINT_SPEC = (
    [(FAM['ident'], FAM['para'], 'ge', 'core'), (FAM['para'], FAM['near'], 'gt', 'core'), (FAM['near'], FAM['far'], 'gt', 'core')]
    + [(FAM['ident'], FAM[k], 'gt', 'core') for k in ('near', 'far', 'sib', 'act', 'loc', 'cnt', 'om', 'hal', 'cs', 'desc', 'st', 'neg', 'bind')]
    + [(FAM['ident'], FAM['reo'], 'eq', 'invariance'), (FAM['near'], FAM['sib'], 'gt', 'extended')]
)
LADDER = [FAM['ident'], FAM['para'], FAM['near'], FAM['far']]


def build_constraints(items):
    """items: DataFrame with source_id, sample_id, perturbation_type. Returns DataFrame
    [source_id, dataset, hi, lo, hi_family, lo_family, kind, suite, group]."""
    rows = []
    for sid, g in items.groupby('source_id', sort=False):
        by = dict(zip(g['perturbation_type'], g['sample_id']))
        ds = g['dataset'].iloc[0]
        for hi_f, lo_f, kind, suite in CONSTRAINT_SPEC:
            if hi_f in by and lo_f in by:
                op = {'gt': '>', 'ge': '>=', 'eq': '=='}[kind]
                rows.append((sid, ds, by[hi_f], by[lo_f], hi_f, lo_f, kind, suite, f'{hi_f}{op}{lo_f}'))
    return pd.DataFrame(rows, columns=['source_id', 'dataset', 'hi', 'lo', 'hi_family', 'lo_family', 'kind', 'suite', 'group'])


def _satisfaction(shi, slo, kind, tol=1e-9):
    d = shi - slo
    tie = np.abs(d) <= tol
    if kind == 'gt':
        strict = (d > tol).astype(float)
        half = strict + 0.5 * tie
    elif kind == 'ge':
        strict = (d >= -tol).astype(float)
        half = strict.copy()
    else:  # eq
        strict = tie.astype(float)
        half = strict.copy()
    return strict, half, tie


def _boot(cluster_ids, num, den, n_boot, seed):
    """cluster bootstrap of sum(num)/sum(den); returns (lo, hi)."""
    uniq, inv = np.unique(cluster_ids, return_inverse=True)
    S = len(uniq)
    cn = np.bincount(inv, weights=num, minlength=S)
    cd = np.bincount(inv, weights=den, minlength=S)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, S, size=(n_boot, S))
    dd = cd[idx].sum(1)
    acc = np.where(dd > 0, cn[idx].sum(1) / np.where(dd > 0, dd, 1), np.nan)
    return tuple(np.nanpercentile(acc, [2.5, 97.5]))


def pairwise_ordering_accuracy(scores, constraints, n_boot=1000, seed=0, tol=1e-9, items=None):
    """scores: Series/dict sample_id -> float (NaN = not scored; constraints touching NaN are dropped).
    constraints: DataFrame from build_constraints (optionally pre-filtered by suite).
    items: optional items DataFrame; if given, monotonicity violations and Kendall tau are added."""
    s = pd.Series(scores) if not isinstance(scores, pd.Series) else scores
    c = constraints.copy()
    c['shi'] = c['hi'].map(s)
    c['slo'] = c['lo'].map(s)
    n_all = len(c)
    c = c[c['shi'].notna() & c['slo'].notna()]
    out = dict(n_constraints=len(c), n_dropped_nan=n_all - len(c))
    if len(c) == 0:
        return out
    parts = [(_satisfaction(g['shi'].to_numpy(float), g['slo'].to_numpy(float), k, tol), g) for k, g in c.groupby('kind')]
    strict = np.zeros(len(c)); half = np.zeros(len(c)); tie = np.zeros(len(c))
    pos = {ix: i for i, ix in enumerate(c.index)}
    for (st, hf, ti), g in parts:
        ii = [pos[x] for x in g.index]
        strict[ii], half[ii], tie[ii] = st, hf, ti
    cid = c['source_id'].to_numpy()
    ones = np.ones(len(c))
    out.update(n_satisfied_strict=float(strict.sum()), accuracy_strict=float(strict.mean()),
               accuracy_strict_ci95=_boot(cid, strict, ones, n_boot, seed),
               n_ties=int(tie.sum()), tie_rate=float(tie.mean()),
               accuracy_tiehalf=float(half.mean()), accuracy_tiehalf_ci95=_boot(cid, half, ones, n_boot, seed + 1),
               n_source_captions=int(len(np.unique(cid))))
    if items is not None:
        out.update(monotonicity_and_tau(s, items, tol))
    return out


def monotonicity_and_tau(s, items, tol=1e-9):
    """Monotonicity violations along the ladder identity >= paraphrase >= near >= distant (a violation =
    a MORE severe candidate scores strictly higher than a less severe one; ties are not violations), and Kendall tau
    between design severity and score (sign flipped so +1 = score falls as severity rises)."""
    it = items.assign(score=items['sample_id'].map(s))
    it = it[it['score'].notna()]
    n_chain = n_viol = 0
    taus_all, taus_lad = [], []
    for sid, g in it.groupby('source_id', sort=False):
        by = dict(zip(g['perturbation_type'], g['score']))
        seq = [by[f] for f in LADDER if f in by]
        if len(seq) >= 3 and LADDER[0] in by and LADDER[2] in by and LADDER[3] in by:
            n_chain += 1
            if any(seq[i + 1] > seq[i] + tol for i in range(len(seq) - 1)):
                n_viol += 1
        if g['severity_level'].nunique() > 1 and g['score'].nunique() > 1:
            t = kendalltau(g['severity_level'], -g['score'])[0]
            if not math.isnan(t):
                taus_all.append(t)
        gl = g[g['perturbation_type'].isin(LADDER)]
        if len(gl) >= 3 and gl['severity_level'].nunique() > 1 and gl['score'].nunique() > 1:
            t = kendalltau(gl['severity_level'], -gl['score'])[0]
            if not math.isnan(t):
                taus_lad.append(t)
    return dict(monotonic_chains=n_chain, monotonicity_violations=n_viol,
                monotonicity_violation_rate=(n_viol / n_chain if n_chain else float('nan')),
                kendall_tau_severity_all_mean=float(np.mean(taus_all)) if taus_all else float('nan'),
                kendall_tau_severity_ladder_mean=float(np.mean(taus_lad)) if taus_lad else float('nan'),
                kendall_tau_n_sources=len(taus_all))


def score_items_with_callable(items, fn, id_col='sample_id'):
    """fn(candidate, reference) -> float. Returns Series indexed by sample_id."""
    return pd.Series({r[id_col]: float(fn(r['candidate'], r['reference'])) for _, r in items.iterrows()})


def evaluate_metric(items, constraints, scores, name, n_boot=1000, seed=0, suites=('core',)):
    """One row per (dataset, metric) plus ALL; restricted to given suites. Returns list of dict rows."""
    rows = []
    cons = constraints[constraints['suite'].isin(suites)]
    for ds in ['ALL'] + sorted(items['dataset'].unique()):
        ci = cons if ds == 'ALL' else cons[cons['dataset'] == ds]
        ii = items if ds == 'ALL' else items[items['dataset'] == ds]
        r = pairwise_ordering_accuracy(scores, ci, n_boot, seed, items=(ii if tuple(suites) == ('core',) else None))
        r.update(metric=name, dataset=ds, suites='+'.join(suites))
        rows.append(r)
    return rows


def evaluate_by_family(items, constraints, scores, name, n_boot=300, seed=0, suites=('core', 'extended', 'invariance')):
    rows = []
    cons = constraints[constraints['suite'].isin(suites)]
    for grp, g in cons.groupby('group'):
        r = pairwise_ordering_accuracy(scores, g, n_boot, seed)
        r.update(metric=name, group=grp, hi_family=g['hi_family'].iloc[0], lo_family=g['lo_family'].iloc[0], kind=g['kind'].iloc[0], suite=g['suite'].iloc[0])
        rows.append(r)
    return rows


def flatten(rows):
    out = []
    for r in rows:
        r = dict(r)
        for k in list(r):
            if k.endswith('_ci95') and isinstance(r[k], tuple):
                r[k + '_lo'], r[k + '_hi'] = r[k]
                del r[k]
        out.append(r)
    return pd.DataFrame(out)
