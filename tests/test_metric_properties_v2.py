"""Property tests for the v2 evaluator variant (P1 dual matching, P2
exclude_root, P3 not-scorable empty pairs; see ricce/metrics/event_v2.py
and the experiment status notes). Mirrors tests/test_metric_properties.py
exactly wherever the property is about the frozen evaluator's *behaviour*
rather than its specific internals (single shared `matches` list, None-as-0
scoring of doubly-empty inputs), so a reader can diff the two files to see
exactly what changed. The three properties that are xfail(strict=True) in
test_metric_properties.py (test_identity_holds_for_unparsed_empty_event,
test_omission_monotonicity_recall, test_root_only_prediction_not_scored_high)
are NOT marked xfail here: they are expected to pass under v2, and the test
suite is run with `-rf` so a silently-newly-failing case still shows up."""
import random
import numpy as np
import pytest
from ricce.schema import Event
from ricce.benchmarks.env import load_taxes, load_weights
from ricce.metrics.event_v2 import ricce_scores_v2

TAXES = load_taxes()
W = load_weights()
EPS = 1e-9
V2_KW = dict(gate_mode='hard', soft_gamma=0.1, exclude_root=True, dual_matching=True)


def sc(pred, gt):
    return ricce_scores_v2(pred, gt, TAXES, W, **V2_KW)


def ev(obj='building', act='constructed', loc='left', qty='several', st='change', src=None, tgt='__auto__', cnt=None):
    if tgt == '__auto__':
        tgt = obj if act in ('constructed', 'appearance') else None
        if act in ('demolished', 'disappearance') and src is None:
            src = obj
    return Event(object=obj, action=act, source=src, target=tgt, location=loc, quantity=qty, exact_count=cnt, change_state=st)


def no_change():
    return Event(change_state='no_change', action='no_change')


def rand_event(rng):
    T = TAXES
    if rng.random() < 0.15:
        return no_change()
    obj = rng.choice(sorted(T['object'].nodes))
    act = rng.choice([a for a in sorted(T['action'].nodes) if a not in ('no_change', 'temporal_state')])
    e = ev(obj, act, rng.choice([None] + sorted(T['location'].nodes)), rng.choice([None] + sorted(T['quantity'].nodes)))
    if act == 'conversion':
        e.source, e.target = rng.choice(sorted(T['object'].nodes)), rng.choice(sorted(T['object'].nodes))
    return e


def rand_events(rng, lo=1, hi=3):
    return [rand_event(rng) for _ in range(rng.randint(lo, hi))]


# ---------------------------------------------------------------- basic
def test_bounds_random():
    rng = random.Random(0)
    checked = 0
    for _ in range(300):
        s = sc(rand_events(rng, 0, 4), rand_events(rng, 0, 4))
        if s['not_scorable']:
            continue  # P3: both sides empty -> intentionally not a [0,1] score
        checked += 1
        for k in ('ce_precision', 'ce_recall', 'ce_f1'):
            assert np.isfinite(s[k]) and -EPS <= s[k] <= 1 + EPS
    assert checked > 200


def test_identity_is_one():
    rng = random.Random(1)
    cases = [[ev()], [no_change()], [ev(), ev('road', 'appearance', 'right')]]
    cases += [[rand_event(rng)] for _ in range(50)]
    for g in cases:
        s = sc(list(g), list(g))
        assert s['ce_precision'] == pytest.approx(1) and s['ce_recall'] == pytest.approx(1) and s['ce_f1'] == pytest.approx(1)


def test_identity_holds_for_unparsed_empty_event():
    """P3 fix: both sides a single all-None event -> not_scorable, not a
    silent 0. This is the correct generalisation of 'identity' for an input
    that carries no assertable information at all."""
    e = Event()
    s = sc([e], [Event()])
    assert s['not_scorable'] is True
    assert s['ce_precision'] is None and s['ce_recall'] is None and s['ce_f1'] is None


def test_correct_ancestor_precision_one_recall_below():
    gt = [ev('residential_building')]
    for anc in ('building', 'built_environment'):
        s = sc([ev(anc)], gt)
        assert s['ce_precision'] == pytest.approx(1.0)
        assert s['ce_recall'] < 1.0


def test_unsupported_descendant_precision_below_recall_one():
    gt = [ev('building')]
    s = sc([ev('residential_building')], gt)
    assert s['ce_precision'] < 1.0
    assert s['ce_recall'] == pytest.approx(1.0)


def test_sibling_both_below_one():
    s = sc([ev('road')], [ev('bridge')])
    assert s['ce_precision'] < 1.0 and s['ce_recall'] < 1.0


def test_ancestor_ladder_monotone():
    gt = [ev('residential_building')]
    f = [sc([ev(o)], gt)['ce_f1'] for o in ('residential_building', 'building', 'built_environment', 'entity')]
    assert all(f[i] > f[i + 1] for i in range(3))


# ---------------------------------------------------------------- monotonicity (randomised metamorphic)
def _unmatched_add_cases(n=400, seed=2):
    rng = random.Random(seed)
    for _ in range(n):
        pred, gt = rand_events(rng, 1, 3), rand_events(rng, 1, 3)
        yield pred, gt, rand_event(rng)


def test_hallucination_monotonicity_precision():
    """Adding a predicted event that ends up UNMATCHED (on the precision-side
    matching) never raises CE-P."""
    checked = 0
    for pred, gt, extra in _unmatched_add_cases():
        base = sc(pred, gt)
        new = sc(pred + [extra], gt)
        matched_pred = {m[0] for m in new['matches_p']}
        if len(pred) in matched_pred:
            continue  # the extra event was matched on the precision side -> not a pure hallucination
        checked += 1
        assert new['ce_precision'] <= base['ce_precision'] + EPS, (pred, gt, extra)
    assert checked > 50


def test_hallucination_monotonicity_zero_overlap_event():
    gt = [ev('building', 'constructed', 'left')]
    pred = [ev('building', 'constructed', 'left')]
    halluc = ev('water', 'demolished', 'center', 'many')
    assert sc(pred + [halluc], gt)['ce_precision'] < sc(pred, gt)['ce_precision']


def test_omission_monotonicity_recall():
    """P1 fix: removing a MATCHED predicted event never raises CE-R, because
    CE-R now comes from its own S_R-maximising assignment (dual_matching),
    which is monotone in row removal by construction (Hungarian's optimum
    over a matrix with one fewer row can only be <= the optimum over the
    full matrix, since every feasible assignment of the smaller problem is
    also feasible in the larger one)."""
    rng = random.Random(3)
    checked = 0
    for _ in range(400):
        pred, gt = rand_events(rng, 2, 3), rand_events(rng, 1, 3)
        base = sc(pred, gt)
        if base['not_scorable']:
            continue
        for (i, j, sr) in base['matches_r']:
            if sr <= 0:
                continue
            new = sc([p for k, p in enumerate(pred) if k != i], gt)
            if new['not_scorable']:
                continue
            checked += 1
            assert new['ce_recall'] <= base['ce_recall'] + EPS, (pred, gt, i)
    assert checked > 50


def test_prediction_and_reference_order_invariance():
    rng = random.Random(4)
    for _ in range(200):
        pred, gt = rand_events(rng, 1, 4), rand_events(rng, 1, 4)
        a = sc(pred, gt)
        p2, g2 = pred[:], gt[:]
        rng.shuffle(p2); rng.shuffle(g2)
        b = sc(p2, g2)
        for k in ('ce_precision', 'ce_recall', 'ce_f1'):
            assert a[k] == pytest.approx(b[k], abs=1e-9), (pred, gt, p2, g2, k)


def test_one_to_one_matching_prediction_side():
    e = ev()
    s = sc([e], [e, ev(*(e.object, e.action, e.location, e.quantity))])
    assert s['ce_recall'] == pytest.approx(0.5) and s['ce_precision'] == pytest.approx(1.0)


def test_one_to_one_matching_reference_side():
    e = ev()
    s = sc([e, ev()], [e])
    assert s['ce_precision'] == pytest.approx(0.5) and s['ce_recall'] == pytest.approx(1.0)


def test_one_prediction_cannot_get_full_credit_against_two_distinct_refs():
    g1, g2 = ev('building', 'constructed', 'left'), ev('road', 'appearance', 'right')
    s = sc([ev('building', 'constructed', 'left')], [g1, g2])
    assert s['ce_recall'] < 1.0 and len(s['matches_r']) == 1


# ---------------------------------------------------------------- contradictions
def test_change_no_change_contradiction():
    for a, b in ((no_change(), ev()), (ev(), no_change())):
        s = sc([a], [b])
        assert s['ce_precision'] == 0 and s['ce_recall'] == 0 and s['ce_f1'] == 0


@pytest.mark.parametrize('a,b', [('constructed', 'demolished'), ('demolished', 'constructed'),
                                 ('appearance', 'disappearance'), ('disappearance', 'appearance'),
                                 ('expansion', 'contraction'), ('contraction', 'expansion'),
                                 ('constructed', 'disappearance'), ('expansion', 'contraction')])
def test_directional_reversal_contradiction(a, b):
    s = sc([ev('building', a)], [ev('building', b)])
    assert s['ce_f1'] == 0 and s['ce_precision'] == 0 and s['ce_recall'] == 0


def test_conversion_source_target_swap_contradiction():
    g = ev('building', 'conversion', tgt='road', src='vegetation')
    p = ev('building', 'conversion', tgt='vegetation', src='road')
    assert sc([p], [g])['ce_f1'] == 0


# ---------------------------------------------------------------- missing-field asymmetry
def test_prediction_only_field_hurts_precision_not_recall():
    gt = [ev('building', 'constructed', None, 'several')]
    pred = [ev('building', 'constructed', 'left', 'several')]
    s = sc(pred, gt)
    assert s['ce_precision'] < 1.0 and s['ce_recall'] == pytest.approx(1.0)


def test_reference_only_field_hurts_recall_not_precision():
    gt = [ev('building', 'constructed', 'left', 'several')]
    pred = [ev('building', 'constructed', None, 'several')]
    s = sc(pred, gt)
    assert s['ce_recall'] < 1.0 and s['ce_precision'] == pytest.approx(1.0)


# ---------------------------------------------------------------- root-only / degenerate
def test_root_only_prediction_not_scored_high():
    """P2 fix: with the taxonomy root excluded from the ancestor-overlap
    count, a root-only prediction shares no ancestor with any specific
    reference node on any axis and scores 0, not ~0.60."""
    gt = [ev('residential_building', 'constructed', 'upper_left', 'several')]
    root = Event(object='entity', action='temporal_state', location='image', quantity='quantity', change_state='change')
    s = sc([root], gt)
    assert s['ce_f1'] < 0.5, s


def test_empty_prediction_scores_zero():
    s = sc([], [ev()])
    assert s['ce_f1'] == 0


# ---------------------------------------------------------------- determinism
def test_deterministic_repeat_and_tie_breaking():
    rng = random.Random(5)
    for _ in range(50):
        pred, gt = rand_events(rng, 1, 4), rand_events(rng, 1, 4)
        r = [sc(pred, gt) for _ in range(3)]
        assert r[0]['matches_p'] == r[1]['matches_p'] == r[2]['matches_p']
        assert r[0]['matches_r'] == r[1]['matches_r'] == r[2]['matches_r']
    e = ev()
    outs = [sc([e, ev('road', 'appearance', 'right')], [e, e]) for _ in range(5)]
    assert all(o['matches_p'] == outs[0]['matches_p'] for o in outs)
