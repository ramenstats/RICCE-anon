"""v3: design ablations under the FINAL evaluator (replaces the
historical-evaluator ablation table). Each variant removes one component of
ricce/metrics/final.py by a process-local patch of event_v2 (the module
itself is never edited), and is scored on
  * RICCE-Bench (core / semantic-validity / near>sibling ordering, mean CE-F),
  * the 8-system LEVIR-CC benchmark (Kendall tau vs. the full metric),
  * the 759 x 5 real naturally multi-event pairs (which component accounts for
    the gap between CE-F and flat hierarchical scores on real data).
Outputs -> results_v3/ablations/."""
import contextlib, dataclasses, json, sys, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import numpy as np, pandas as pd
from scipy.stats import kendalltau
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.schema import Event
from ricce.utils import read_jsonl
from ricce.benchmarks.env import load_taxes, load_weights
from ricce.evaluation.ordering import build_constraints, pairwise_ordering_accuracy
from ricce.evaluation.parse_store import ParseStore
from ricce.metrics import event_v2 as V2
from ricce.metrics import event as V1
from ricce.metrics.final import FINAL_CONFIG
from ricce.baselines.ablations import (_family_reversal_only, drop_nochange, make_collapse, collapse_events,
                                          _gate_on_original)

OUT = ROOT / 'results_v3' / 'ablations'; OUT.mkdir(parents=True, exist_ok=True)
T, W = load_taxes(), load_weights()
AXES = ['object', 'action', 'source', 'target', 'location', 'quantity', 'change_state']
_ORIG = dict(_field_pr_v2=V2._field_pr_v2, event_field_scores_v2=V2.event_field_scores_v2,
             is_action_reversal=V2.is_action_reversal, linear_sum_assignment=V2.linear_sum_assignment)


@contextlib.contextmanager
def patched(**kw):
    old = {k: getattr(V2, k) for k in kw}
    try:
        for k, v in kw.items():
            setattr(V2, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(V2, k, v)


# ---------------------------------------------------------------- component removals
def field_exact(tax, a, b, exclude_root, root_mode=None):
    """A1: no hierarchy -- exact node match; same presence and root semantics as the final evaluator."""
    a = None if a == tax.root else a
    b = None if b == tax.root else b
    if not a and not b:
        return None, None
    if a and not b:
        return 0.0, None
    if b and not a:
        return None, 0.0
    if a not in tax.nodes or b not in tax.nodes:
        return 0.0, 0.0
    v = 1.0 if a == b else 0.0
    return v, v


def fields_unordered(pred, gt, taxes, exclude_root=True, root_mode=None):
    """A5: source/target compared as an unordered pair (best of direct / crossed)."""
    f = _ORIG['event_field_scores_v2'](pred, gt, taxes, exclude_root, root_mode)
    if _family_reversal_only(pred, gt, taxes):
        return f
    ot, fp = taxes['object'], _ORIG['_field_pr_v2']
    d = (fp(ot, pred.source, gt.source, exclude_root, root_mode), fp(ot, pred.target, gt.target, exclude_root, root_mode))
    c = (fp(ot, pred.source, gt.target, exclude_root, root_mode), fp(ot, pred.target, gt.source, exclude_root, root_mode))
    tot = lambda pr: sum((x or 0.0) for pair in pr for x in pair)
    f['source'], f['target'] = c if tot(c) > tot(d) else d
    return f


def greedy_lsa(C):
    """A6: greedy assignment on the same score matrices (C = -S)."""
    S = -np.asarray(C); M, N = S.shape
    cells = sorted(((-S[i, j], i, j) for i in range(M) for j in range(N)))
    ur, uc, r, c = set(), set(), [], []
    for _, i, j in cells:
        if i in ur or j in uc:
            continue
        ur.add(i); uc.add(j); r.append(i); c.append(j)
        if len(r) == min(M, N):
            break
    o = np.argsort(r)
    return np.array(r)[o], np.array(c)[o]


def bag(pred, gt, taxes, weights):
    """A3: no event binding -- every field of every event pooled per axis, best match per label."""
    rae = lambda es: all(V2.is_empty_event(e, taxes, True) for e in es) if es else True
    if rae(pred) and rae(gt):
        return np.nan
    if not pred or not gt:
        return 0.0
    tx = {'object': taxes['object'], 'source': taxes['object'], 'target': taxes['object'], 'action': taxes['action'],
          'location': taxes['location'], 'quantity': taxes['quantity']}
    fields = {}
    for f in AXES:
        rt = tx[f].root if f in tx else None
        pv = [getattr(e, f) for e in pred if getattr(e, f) and getattr(e, f) != rt]
        gv = [getattr(e, f) for e in gt if getattr(e, f) and getattr(e, f) != rt]
        if not pv and not gv:
            fields[f] = (None, None)
        elif pv and not gv:
            fields[f] = (0.0, None)
        elif gv and not pv:
            fields[f] = (None, 0.0)
        elif f == 'change_state':
            fields[f] = (sum(p in gv for p in pv) / len(pv), sum(g in pv for g in gv) / len(gv))
        else:
            t = tx[f]
            hs = lambda p, g: t.hierarchical_scores(p, g, exclude_root=True)[:2] if (p in t.nodes and g in t.nodes) else (0.0, 0.0)
            fields[f] = (sum(max(hs(p, g)[0] for g in gv) for p in pv) / len(pv), sum(max(hs(p, g)[1] for p in pv) for g in gv) / len(gv))
    sp, sr = V2.aggregate_pr(fields, weights)
    pcs, gcs = {e.change_state for e in pred if e.change_state}, {e.change_state for e in gt if e.change_state}
    gate = bool(pcs and gcs and not (pcs & gcs))
    if not gate:
        gate = any(V1.is_action_reversal(p, g, taxes) for p in pred for g in gt) and not any(p.action == g.action for p in pred for g in gt)
    if gate:
        return 0.0
    return 2 * sp * sr / (sp + sr) if sp + sr else 0.0


def nostate(pred, gt, taxes, weights):
    if not pred and not gt:
        return 1.0
    s = V2.ricce_scores_v2(pred, gt, taxes, weights, **FINAL_CONFIG)
    return np.nan if s['not_scorable'] else s['ce_f1']


@dataclass
class Variant:
    name: str
    desc: str
    group: str = 'component'
    single_parse: bool = False
    transform: Optional[Callable] = None
    patches: dict = field(default_factory=dict)
    scorer: Optional[Callable] = None
    cfg: dict = field(default_factory=dict)
    weights: Optional[dict] = None
    taxes: Optional[dict] = None


def build():
    V = {}
    add = lambda v: V.__setitem__(v.name, v)
    add(Variant('A0_full', 'final evaluator'))
    add(Variant('A1_no_hierarchy', 'exact node match only', patches=dict(_field_pr_v2=field_exact)))
    add(Variant('A2_no_decomposition', 'one event per caption (single-event parse)', single_parse=True))
    add(Variant('A3_no_binding', 'fields of all events pooled per axis', scorer=bag))
    add(Variant('A4_no_gate', 'contradiction gate off', cfg=dict(gate_mode='none')))
    add(Variant('A5_unordered_source_target', 'source/target unordered; conversion-swap gate off',
                patches=dict(event_field_scores_v2=fields_unordered, is_action_reversal=_family_reversal_only)))
    add(Variant('A6_greedy', 'greedy instead of optimal assignment (both assignments)', patches=dict(linear_sum_assignment=greedy_lsa)))
    add(Variant('A7_no_nochange_state', 'no-change events dropped; empty vs empty = 1', transform=drop_nochange, scorer=nostate))
    add(Variant('A8_single_assignment', 'one shared assignment maximising harmonic(S_P,S_R)', cfg=dict(dual_matching=False, joint_fn='harmonic')))
    add(Variant('A9_root_counted', 'taxonomy root counted as an ancestor', cfg=dict(root_mode='include')))
    add(Variant('A10_root_excluded', 'root removed from ancestor sets, root labels kept as nodes', cfg=dict(root_mode='exclude')))
    add(Variant('A11_gate_any_object', 'contradiction gate also between events about different objects', cfg=dict(gate_object_match=False)))
    for k in (1, 2):
        ctax, maps, caps = make_collapse(T, k)
        add(Variant(f'DEPTH_minus{k}', f'ontology collapsed by {k} level(s); gate on original ids', group='depth',
                    transform=(lambda m: lambda es: collapse_events(es, m))(maps), taxes=ctax,
                    patches=dict(is_action_reversal=_gate_on_original(T))))
    add(Variant('W_equal', 'all field weights 1', group='weights', weights={k: 1.0 for k in W}))
    add(Variant('W_action_x3', 'action weight x3', group='weights', weights={**W, 'action': W['action'] * 3}))
    add(Variant('W_object_x3', 'object weight x3', group='weights', weights={**W, 'object': W['object'] * 3}))
    import random
    rng = random.Random(0)
    for i in range(10):
        f = {ax: rng.choice([0.8, 1.2]) for ax in AXES}
        add(Variant(f'W_random_{i}', f'weights x{f}', group='weights', weights={k: W[k] * f.get(k, 1.0) for k in W}))
    for ax in AXES:
        add(Variant(f'LOO_{ax}', f'weight[{ax}]=0', group='axis_loo', weights={**W, ax: 0.0}))
    return V


VARS = build()


def score(v, p, g):
    tx = v.taxes or T
    w = v.weights or W
    if v.transform:
        p, g = v.transform(p), v.transform(g)
    with patched(**v.patches):
        if v.scorer:
            return v.scorer(p, g, tx, w)
        s = V2.ricce_scores_v2(p, g, tx, w, **{**FINAL_CONFIG, **v.cfg})
        return np.nan if s['not_scorable'] else s['ce_f1']


# ---------------------------------------------------------------- data (loaded before fork)
E = lambda L: [Event(**{k: v for k, v in d.items() if k in Event.__dataclass_fields__}) for d in L]
items = pd.DataFrame([json.loads(l) for l in open(ROOT / 'artifacts_cvpr_final' / 'riccebench' / 'items.jsonl')])
old = ParseStore(ROOT / 'results_cvpr_final' / 'ablations' / 'parse_cache.json')
store = ParseStore(ROOT / 'results_v3' / 'parse_cache.json')
BENCH = {'multi': [(E(c), E(s)) for c, s in zip(items.candidate_event, items.source_event)],
         'single': [(old.events(c, 'single'), old.events(r, 'single')) for c, r in zip(items.candidate, items.reference)]}
MAIN = ['qwen', 'sft_seed42', 'swmcc', 'chg2cap', 'rsiccformer', 'promptcc', 'rscama']
mtest = {r['image_id']: r for r in read_jsonl(ROOT / 'artifacts' / 'manifests' / 'LEVIR_CC.jsonl') if r.get('split') == 'test'}
SYSD = {}
for sy in MAIN:
    rows = []
    for r in read_jsonl(ROOT / 'results' / 'baselines' / sy / 'LEVIR_CC_test.jsonl'):
        p = str(r['prediction']); refs = mtest[r['image_id']]['references']
        rows.append({m: (store.events(p, m), [store.events(g, m) for g in refs]) for m in ('multi', 'single')})
    SYSD[sy] = rows
NAT = pd.read_csv(ROOT / 'results_v3' / 'systems' / 'natural_multievent_pairs.csv', keep_default_na=False)
NATD = [{m: (store.events(p, m), store.events(g, m)) for m in ('multi', 'single')} for p, g in zip(NAT.prediction.astype(str), NAT.reference)]


def run(name):
    v = VARS[name]; mode = 'single' if v.single_parse else 'multi'; t0 = time.time()
    bf = np.array([score(v, c, s) for c, s in BENCH[mode]], float)
    sysf = {}
    for sy, rows in SYSD.items():
        per = []
        for r in rows:
            pe, gs = r[mode]
            x = [score(v, pe, g) for g in gs]; x = [y for y in x if not np.isnan(y)]
            if x:
                per.append(np.mean(x))
        sysf[sy] = float(np.mean(per))
    nat = np.array([score(v, *d[mode]) for d in NATD], float)
    return name, bf, sysf, nat, time.time() - t0


if __name__ == '__main__':
    import multiprocessing as mp
    names = list(VARS)
    res = {}
    with mp.get_context('fork').Pool(min(40, len(names))) as pool:
        for name, bf, sysf, nat, dt in pool.imap_unordered(run, names):
            res[name] = (bf, sysf, nat); print(name, f'{dt:.0f}s', flush=True)
    cons = build_constraints(items)
    core = cons[cons.suite == 'core']; ext = cons[cons.suite == 'extended']
    GOOD = ['P0_paraphrase', 'P1_near_ancestor']; BAD = ['P4_action_reversal', 'P9_change_state_reversal', 'negation_flip', 'source_target_swap']
    a0sys = res['A0_full'][1]
    rows = []
    for name in names:
        bf, sysf, nat = res[name]; s = pd.Series(bf, index=items.sample_id.values)
        o = pairwise_ordering_accuracy(s, core, n_boot=500, seed=0)
        e = pairwise_ordering_accuracy(s, ext, n_boot=200, seed=0)
        d = pd.DataFrame({'source_id': items.source_id.values, 'ptype': items.perturbation_type.values, 'F': bf})
        piv = d.pivot_table(index='source_id', columns='ptype', values='F'); w = n = 0
        for gd in GOOD:
            for bd in BAD:
                x = piv[[gd, bd]].dropna(); w += int((x[gd] > x[bd] + 1e-9).sum()); n += len(x)
        fam = d.groupby('ptype').F.mean()
        NAT[name] = nat
        row = dict(variant=name, group=VARS[name].group, desc=VARS[name].desc, core=o['accuracy_strict'], core_lo=o['accuracy_strict_ci95'][0],
                   core_hi=o['accuracy_strict_ci95'][1], validity=w / n, near_gt_sibling=e['accuracy_strict'], mean_cef_bench=float(np.nanmean(bf)),
                   cef_action_reversal=float(fam['P4_action_reversal']), cef_source_target_swap=float(fam['source_target_swap']),
                   tau_systems=float(kendalltau([a0sys[x] for x in MAIN], [sysf[x] for x in MAIN])[0]),
                   order=' > '.join(sorted(MAIN, key=lambda x: -sysf[x])), natural_multievent_mean=float(np.nanmean(nat)),
                   **{f'sys_{k}': v for k, v in sysf.items()},
                   **{f'nat_{sy}': float(np.nanmean(nat[(NAT.system == sy).values])) for sy in NAT.system.unique()})
        rows.append(row)
    S = pd.DataFrame(rows); S.to_csv(OUT / 'ablation_summary.csv', index=False)
    NAT.drop(columns=['prediction', 'reference']).to_csv(OUT / 'natural_multievent_by_variant.csv', index=False)
    pd.set_option('display.width', 250)
    print(S[['variant', 'core', 'validity', 'near_gt_sibling', 'mean_cef_bench', 'cef_action_reversal', 'tau_systems', 'natural_multievent_mean']].round(3).to_string())
    json.dump({'n_items': len(items), 'variants': {k: v.desc for k, v in VARS.items()}, 'command': 'python scripts/v3_ablations.py'},
              open(OUT / 'config.json', 'w'), indent=1)
    print('DONE')
