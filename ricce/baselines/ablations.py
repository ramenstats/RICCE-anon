"""RICCE ablation variants (A0..A7, leave-one-axis-out, weight robustness, ontology-depth collapse).

Everything is implemented as wrappers / context-manager monkeypatches of `ricce.metrics.event`
(the core file is NOT edited); patches are always restored on exit. NOTE: patching mutates module globals,
so scoring must be single-threaded per process.

A Variant = (parse_multi, event transform, patches, scorer, weights, matcher).
`score_events(variant, pred_events, gt_events, taxes, base_weights)` -> dict(ce_precision, ce_recall, ce_f1).
"""
import contextlib, dataclasses, random
from dataclasses import dataclass, field
from typing import Callable, Optional
from ..metrics import event as E
from ..metrics.event import ricce_scores
from ..ontology.tree import Taxonomy, Node
from ..schema import Event

_ORIG = dict(_field_pr=E._field_pr, is_action_reversal=E.is_action_reversal, event_similarity_pr=E.event_similarity_pr,
             event_field_scores=E.event_field_scores)
AXES = ['object', 'action', 'source', 'target', 'location', 'quantity', 'change_state']


@contextlib.contextmanager
def patched(**kw):
    old = {k: getattr(E, k) for k in kw}
    try:
        for k, v in kw.items():
            setattr(E, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(E, k, v)


# ------------------------------------------------------------------ A1 no hierarchy
def _field_pr_flat(tax, a, b):
    """Same missing-field semantics as core, but exact node match only (no ancestor credit)."""
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


# ------------------------------------------------------------------ A4 no gates
def _no_reversal(pred, gt, taxes=None):
    return False


def _sim_pr_nogate(pred, gt, taxes, weights):
    return E.aggregate_pr(E.event_field_scores(pred, gt, taxes), weights)


# ------------------------------------------------------------------ A5 no source/target distinction
def _family_reversal_only(pred, gt, taxes=None):
    """core is_action_reversal minus the conversion source/target-swap clause."""
    at = taxes.get('action') if taxes else None
    if at is not None:
        pf, gf = E._action_family(pred.action, at), E._action_family(gt.action, at)
        return (pf, gf) in E._FAMILY_REVERSALS
    return (pred.action, gt.action) in E.ACTION_REVERSALS


def _field_scores_unordered(pred, gt, taxes):
    f = _ORIG['event_field_scores'](pred, gt, taxes)
    if _family_reversal_only(pred, gt, taxes):
        return f
    ot = taxes['object']
    d = (E._field_pr(ot, pred.source, gt.source), E._field_pr(ot, pred.target, gt.target))
    c = (E._field_pr(ot, pred.source, gt.target), E._field_pr(ot, pred.target, gt.source))
    tot = lambda pr: sum((x or 0.0) for pair in pr for x in pair)
    best = c if tot(c) > tot(d) else d
    f['source'], f['target'] = best
    return f


# ------------------------------------------------------------------ A3 no event binding (bag of fields)
def bag_scores(pred, gt, taxes, weights):
    if not pred or not gt:
        return dict(ce_precision=0.0, ce_recall=0.0, ce_f1=0.0)
    tx = {'object': taxes['object'], 'source': taxes['object'], 'target': taxes['object'], 'action': taxes['action'],
          'location': taxes['location'], 'quantity': taxes['quantity']}
    fields = {}
    for f in AXES:
        pv, gv = [getattr(e, f) for e in pred if getattr(e, f)], [getattr(e, f) for e in gt if getattr(e, f)]
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
            def hs(p, g):
                if p not in t.nodes or g not in t.nodes:
                    return (0.0, 0.0)
                return t.hierarchical_scores(p, g)[:2]
            P = sum(max(hs(p, g)[0] for g in gv) for p in pv) / len(pv)
            R = sum(max(hs(p, g)[1] for p in pv) for g in gv) / len(gv)
            fields[f] = (P, R)
    sp, sr = E.aggregate_pr(fields, weights)
    pcs, gcs = {e.change_state for e in pred if e.change_state}, {e.change_state for e in gt if e.change_state}
    gate = bool(pcs and gcs and not (pcs & gcs))
    if not gate:
        rev = any(E.is_action_reversal(p, g, taxes) for p in pred for g in gt)
        same = any(p.action == g.action for p in pred for g in gt)
        gate = rev and not same
    if gate:
        sp = sr = 0.0
    return dict(ce_precision=sp, ce_recall=sr, ce_f1=(2 * sp * sr / (sp + sr) if sp + sr else 0.0))


# ------------------------------------------------------------------ A7 no explicit no-change state
def drop_nochange(events):
    return [dataclasses.replace(e, change_state=None) for e in events if e.change_state != 'no_change']


def score_nostate(pred, gt, taxes, weights, matcher='hungarian'):
    if not pred and not gt:
        return dict(ce_precision=1.0, ce_recall=1.0, ce_f1=1.0)
    s = ricce_scores(pred, gt, taxes, weights, matcher=matcher)
    return {k: s[k] for k in ('ce_precision', 'ce_recall', 'ce_f1')}


# ------------------------------------------------------------------ ontology depth collapse
def _max_depth(tax):
    return max(tax.depth(n) for n in tax.nodes)


def collapse_taxonomy(tax, k):
    """Programmatically build a Taxonomy truncated by k levels (cap = max(1, maxdepth-k)); aliases of the dropped
    descendants are merged into their surviving ancestor. Returns (collapsed_taxonomy, {orig_id: surviving_id}, cap)."""
    cap = max(1, _max_depth(tax) - k)
    mp = {n: (tax.ancestors(n)[min(tax.depth(n), cap)]) for n in tax.nodes}
    nodes = {}
    for nid, n in tax.nodes.items():
        if tax.depth(nid) <= cap:
            nodes[nid] = Node(nid, n.label, n.parent, list(n.aliases), n.description)
    for nid, tgt in mp.items():
        if nid != tgt:
            n = tax.nodes[nid]
            nodes[tgt].aliases += [n.label] + list(n.aliases)
    return Taxonomy(list(nodes.values()), tax.root), mp, cap


def make_collapse(taxes, k):
    ctax, maps, caps = {}, {}, {}
    for name, t in taxes.items():
        ctax[name], maps[name], caps[name] = collapse_taxonomy(t, k)
    return ctax, maps, caps


def collapse_events(events, maps):
    out = []
    for e in events:
        orig = dict(action=e.action, source=e.source, target=e.target)
        ne = dataclasses.replace(e, object=maps['object'].get(e.object, e.object) if e.object else None,
                                 action=maps['action'].get(e.action, e.action) if e.action else None,
                                 source=maps['object'].get(e.source, e.source) if e.source else None,
                                 target=maps['object'].get(e.target, e.target) if e.target else None,
                                 location=maps['location'].get(e.location, e.location) if e.location else None,
                                 quantity=maps['quantity'].get(e.quantity, e.quantity) if e.quantity else None)
        ne._orig = orig
        out.append(ne)
    return out


def _gate_on_original(orig_taxes):
    def f(pred, gt, taxes=None):
        pa, ga = getattr(pred, '_orig', None), getattr(gt, '_orig', None)
        if pa is None or ga is None:
            return _ORIG['is_action_reversal'](pred, gt, orig_taxes)
        return _ORIG['is_action_reversal'](dataclasses.replace(pred, **pa), dataclasses.replace(gt, **ga), orig_taxes)
    return f


# ------------------------------------------------------------------ variant registry
@dataclass
class Variant:
    name: str
    desc: str
    parse_multi: bool = True
    transform: Optional[Callable] = None
    patches: dict = field(default_factory=dict)
    scorer: Optional[Callable] = None          # (pred, gt, taxes, weights) -> dict
    weights_fn: Optional[Callable] = None      # base weights -> weights
    matcher: str = 'hungarian'
    taxes_fn: Optional[Callable] = None        # base taxes -> taxes used for scoring
    family: str = 'ablation'                   # ablation | axis_loo | weights | depth


def score_events(v, pred, gt, taxes, base_w):
    w = v.weights_fn(base_w) if v.weights_fn else base_w
    tx = v.taxes_fn(taxes) if v.taxes_fn else taxes
    if v.transform:
        pred, gt = v.transform(pred), v.transform(gt)
    with patched(**v.patches):
        if v.scorer:
            return v.scorer(pred, gt, tx, w)
        s = ricce_scores(pred, gt, tx, w, matcher=v.matcher)
        return dict(ce_precision=s['ce_precision'], ce_recall=s['ce_recall'], ce_f1=s['ce_f1'])


def build_variants(base_taxes, weight_seed=0, n_random=10):
    V = {}
    add = lambda v: V.__setitem__(v.name, v)
    add(Variant('A0_full', 'full RICCE (multi-event parse, hierarchy, binding, gates, src/tgt, Hungarian, no-change state)'))
    add(Variant('A1_no_hierarchy', 'exact node match only (no ancestor credit)', patches=dict(_field_pr=_field_pr_flat)))
    add(Variant('A2_no_event_decomposition', 'single event per caption (parse without multi-event; all fields merged)', parse_multi=False))
    add(Variant('A3_no_event_binding', 'flatten fields of all events into one bag before scoring (many-to-many best-match per field)', scorer=bag_scores))
    add(Variant('A4_no_gates', 'both hard gates (change/no-change, action reversal) off; field-level scoring kept',
                patches=dict(is_action_reversal=_no_reversal, event_similarity_pr=_sim_pr_nogate)))
    add(Variant('A5_no_source_target_distinction', 'source/target compared as an unordered pair; conversion-swap gate off',
                patches=dict(is_action_reversal=_family_reversal_only, event_field_scores=_field_scores_unordered)))
    add(Variant('A6_greedy_matching', 'greedy instead of Hungarian event matching', matcher='greedy'))
    add(Variant('A7_no_nochange_state', 'no explicit no-change state: no-change events dropped (empty event set), change_state field removed; empty-vs-empty=1',
                transform=drop_nochange, scorer=lambda p, g, t, w: score_nostate(p, g, t, w)))
    for ax in AXES:
        add(Variant(f'LOO_{ax}', f'leave-one-axis-out: weight[{ax}]=0 (gates unchanged)', weights_fn=(lambda ax: lambda w: {**w, ax: 0.0})(ax), family='axis_loo'))
    add(Variant('W_equal', 'all field weights = 1', weights_fn=lambda w: {k: 1.0 for k in w}, family='weights'))
    add(Variant('W_action_heavy', 'action weight x3', weights_fn=lambda w: {**w, 'action': w['action'] * 3}, family='weights'))
    add(Variant('W_object_heavy', 'object weight x3', weights_fn=lambda w: {**w, 'object': w['object'] * 3}, family='weights'))
    for ax in AXES:
        for sgn, tag in ((1.2, 'p20'), (0.8, 'm20')):
            add(Variant(f'W_{ax}_{tag}', f'weight[{ax}] x{sgn}', weights_fn=(lambda ax, sgn: lambda w: {**w, ax: w[ax] * sgn})(ax, sgn), family='weights'))
    rng = random.Random(weight_seed)
    for i in range(n_random):
        f = {ax: rng.choice([0.8, 1.2]) for ax in AXES}
        add(Variant(f'W_random_pm20_{i}', 'all weights independently x0.8 or x1.2: ' + str(f),
                    weights_fn=(lambda f: lambda w: {k: w[k] * f.get(k, 1.0) for k in w})(f), family='weights'))
    for k in (1, 2):
        ctax, maps, caps = make_collapse(base_taxes, k)
        add(Variant(f'DEPTH_collapse{k}', f'ontology collapsed by {k} level(s), caps={caps}, nodes={ {n: len(t.nodes) for n, t in ctax.items()} }; gates use uncollapsed ids',
                    transform=(lambda maps: lambda ev: collapse_events(ev, maps))(maps), taxes_fn=(lambda ctax: lambda t: ctax)(ctax),
                    patches=dict(is_action_reversal=_gate_on_original(base_taxes)), family='depth'))
    return V
