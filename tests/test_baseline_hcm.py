import networkx as nx, pytest
from pathlib import Path

if not (Path(__file__).resolve().parents[1] / 'third_party' / 'HierarchicalConfusionMatrix').exists():
    pytest.skip('official HCM code not in third_party/ (see README)', allow_module_level=True)

from ricce.baselines.hierarchical.hcm_adapter import (determineHierarchicalConfusionMatrix,
    HCMScorer, metrics_from_confusion)
from ricce.baselines.flatten import FlatContext

# Expected values are the recorded outputs of the official repo's
# JupyterNotebooks/Example_Figure4.ipynb (and README for problem A).
def _g(edges):
    g = nx.DiGraph(); g.add_edges_from(edges); return g
GA = _g([("root","A"),("root","B"),("A","C"),("A","D"),("A","E"),("C","H"),("D","I"),("D","J"),("D","K"),("E","L"),("E","M"),("B","F"),("B","G")])
GB = _g([("root","A"),("root","B"),("root","D"),("A","C"),("A","D"),("A","E"),("C","H"),("D","I"),("D","J"),("D","K"),("E","K"),("E","L"),("E","M"),("B","F"),("B","G")])
GC = _g([("root","A"),("root","B"),("A","C"),("A","D"),("A","E"),("C","H"),("D","I"),("D","J"),("D","K"),("E","L"),("E","M"),("B","E"),("B","F"),("B","G")])

CASES = [
 (GA, ["I"], [["root","A","D","I"]], (3,5,0,0)),
 (GA, ["I"], [["root","A","D","J"]], (2,4,1,1)),
 (GA, ["I"], [["root","A","E","L"]], (1,2,2,2)),
 (GB, ["I"], [["root","A","D","I"]], (3,5,0,0)),
 (GB, ["I"], [["root","D","J"]], (1,5,1,1)),
 (GB, ["I"], [["root","A","E","L"]], (1,2,2,2)),
 (GC, ["I","F"], [["root","A","D","I"],["root","B","F"]], (5,8,0,0)),
 (GC, ["I","F"], [["root","A","D","I"],["root","A","D","J"]], (3,5,3,2)),
 (GC, ["I","F"], [["root","A","C","H"],["root","B","E","M"]], (2,4,4,3)),
]

@pytest.mark.parametrize('g,t,p,exp', CASES)
def test_official_examples(g, t, p, exp):
    assert tuple(int(x) for x in determineHierarchicalConfusionMatrix(g, t, p)) == exp

def test_metrics_formula():
    ppv, rec, f1, mcc = metrics_from_confusion(2, 4, 1, 1)
    assert ppv == pytest.approx(2/3) and rec == pytest.approx(2/3) and f1 == pytest.approx(2/3)
    assert mcc == pytest.approx((2*4-1*1)/((3*3*5*5)**.5))

def test_adapter_flat_labels():
    ctx = FlatContext(use_embeddings=False); h = HCMScorer(ctx)
    s = {('object','building'),('action','constructed')}
    assert h(s, s)['F'] == 1
    r = h({('object','residential_building')}, {('object','building')})
    assert r['P'] < 1 and r['R'] == 1 or r['R'] < 1  # deeper prediction is penalised
    assert h({('source','building')}, {('target','building')})['P'] == pytest.approx(0)  # axes disjoint after root
