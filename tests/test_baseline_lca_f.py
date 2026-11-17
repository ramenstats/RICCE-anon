import random, shutil, pytest
from ricce.baselines.flatten import FlatContext, AXES
from ricce.baselines.hierarchical import lca_f as L
from ricce.baselines.hierarchical.kiritchenko import kiritchenko_labels

pytestmark = pytest.mark.skipif(not L.HEMKIT_DIR.exists() or (shutil.which('g++') is None and not L.HEMKIT_BIN.exists()),
                                reason='HEMKit not in third_party/ or not buildable (see README)')

@pytest.fixture(scope='module')
def ctx(): return FlatContext(use_embeddings=False)
@pytest.fixture(scope='module')
def hem(ctx): return L.LCAScorer(ctx, 'hemkit')
@pytest.fixture(scope='module')
def py(ctx): return L.LCAScorer(ctx, 'python')

def S(*x): return set(x)

FIXTURES = [
 # (pred, ref, (LCA-P, LCA-R))  values checked against the authors' HEMKit binary
 (S(('object','road')), S(('object','road')), (1.0, 1.0)),
 # predicting the ancestor of the true label: LCA path = {anc}, {true, anc}  -> P=1, R=1/2
 (S(('object','building')), S(('object','residential_building')), (1.0, 0.5)),
 (S(('object','residential_building')), S(('object','building')), (0.5, 1.0)),
 # siblings under building: aug = {res,building} vs {ind,building}
 (S(('object','residential_building')), S(('object','industrial_building')), (0.5, 0.5)),
]

@pytest.mark.parametrize('p,r,exp', FIXTURES)
def test_fixtures_hemkit_and_python(p, r, exp, hem, py):
    for sc in (hem, py):
        s = sc(p, r)
        assert (s['P'], s['R']) == pytest.approx(exp, abs=1e-5)

def test_empty_rules(hem):
    assert hem(set(), set())['F'] == 1 and hem(set(), S(('object','road')))['F'] == 0

def test_hemkit_hier_prf_matches_kiritchenko(ctx, hem):
    """HEMKit's own hierarchical P/R/F = ours once its two implicit roots (virtual root + ROOT) are added."""
    random.seed(0)
    allL = [(a, n) for a in AXES for n in ctx.axis_tax[a].nodes]
    for _ in range(40):
        p = set(random.sample(allL, random.randint(1, 4))); r = set(random.sample(allL, random.randint(1, 4)))
        k = kiritchenko_labels(p, r, ctx); h = hem(p, r)
        assert h['_hemkit_hP'] == pytest.approx((k['_inter'] + 2) / (k['_np'] + 2), abs=1e-5)
        assert h['_hemkit_hR'] == pytest.approx((k['_inter'] + 2) / (k['_nr'] + 2), abs=1e-5)

def test_python_port_agreement_same_axis(ctx, hem, py):
    """Non-tied, same-axis cases: the experimental port must match HEMKit."""
    random.seed(3); ok = tot = 0
    for ax in ('object', 'action', 'location'):
        nodes = list(ctx.axis_tax[ax].nodes)
        for _ in range(30):
            p, r = {(ax, random.choice(nodes))}, {(ax, random.choice(nodes))}
            a, b = hem(p, r), py(p, r); tot += 1
            ok += abs(a['P'] - b['P']) < 1e-5 and abs(a['R'] - b['R']) < 1e-5
    assert ok == tot
