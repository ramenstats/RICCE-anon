import pytest
from ricce.baselines.flatten import FlatContext
from ricce.baselines.hierarchical.kiritchenko import kiritchenko_labels, aggregate_micro

@pytest.fixture(scope='module')
def ctx():
    return FlatContext(use_embeddings=False)

def L(*xs): return set(xs)

def test_exact(ctx):
    s = L(('object', 'building'), ('action', 'constructed'))
    r = kiritchenko_labels(s, s, ctx)
    assert (r['P'], r['R'], r['F']) == (1, 1, 1)

def test_correct_ancestor(ctx):  # predicts the coarser (correct) ancestor
    r = kiritchenko_labels(L(('object', 'building')), L(('object', 'residential_building')), ctx)
    assert r['P'] == 1 and r['R'] < 1

def test_unsupported_descendant(ctx):
    r = kiritchenko_labels(L(('object', 'residential_building')), L(('object', 'building')), ctx)
    assert r['P'] < 1 and r['R'] == 1

def test_sibling(ctx):
    r = kiritchenko_labels(L(('object', 'residential_building')), L(('object', 'industrial_building')), ctx)
    assert 0 < r['P'] < 1 and 0 < r['R'] < 1
    # shared aug = entity, built_environment, building of 5 each
    assert r['P'] == pytest.approx(3 / 4)

def test_extra_label_lowers_precision(ctx):
    ref = L(('object', 'road'))
    r = kiritchenko_labels(ref | L(('object', 'tree')), ref, ctx)
    assert r['P'] < 1 and r['R'] == 1

def test_missing_label_lowers_recall(ctx):
    pred = L(('object', 'road'))
    r = kiritchenko_labels(pred, pred | L(('action', 'constructed')), ctx)
    assert r['R'] < 1 and r['P'] == 1

def test_axes_do_not_overlap(ctx):
    r = kiritchenko_labels(L(('source', 'building')), L(('target', 'building')), ctx)
    assert r['P'] == 0 == r['R']

def test_empty(ctx):
    assert kiritchenko_labels(set(), set(), ctx)['F'] == 1
    assert kiritchenko_labels(set(), L(('object', 'road')), ctx)['F'] == 0

def test_micro_ratio_of_sums(ctx):
    a = kiritchenko_labels(L(('object', 'road')), L(('object', 'road')), ctx)
    b = kiritchenko_labels(L(('object', 'tree')), L(('object', 'road'), ('action', 'constructed')), ctx)
    m = aggregate_micro([a, b])
    assert m['P'] == pytest.approx((a['_inter'] + b['_inter']) / (a['_np'] + b['_np']))
