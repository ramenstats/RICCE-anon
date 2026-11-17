"""Adapter for the Hierarchical Confusion Matrix (Riehl, Neunteufel, Hemberg
2023; https://github.com/DerKevinRiehl/HierarchicalConfusionMatrix, commit
2256f4c, cloned to third_party/). The official function
`determineHierarchicalConfusionMatrix` is imported UNMODIFIED.

Graph: node "root" (required by the official code) -> axis-prefixed copy of each
RICCE taxonomy for object / action / source / target / location / quantity
/ change_state.  Each resolved field of every event becomes one multi-path
label; event binding is ignored (flattened set).  Reference labels are node
names, predictions are root->node paths (non-mandatory-leaf semantics).
Metrics from the four counts as in the authors' example scripts:
PPV=TP/(TP+FP), REC=TP/(TP+FN), F1, MCC (None when its denominator is 0).
Empty-set convention: both empty -> P=R=F=1, MCC None; one empty -> 0.
"""
from __future__ import annotations
import sys, math
import networkx as nx
from .. import flatten as fl

_TP = fl.ROOT_DIR / 'third_party' / 'HierarchicalConfusionMatrix'
if str(_TP) not in sys.path:
    sys.path.insert(0, str(_TP))
from hierarchical_confusion_matrix import determineHierarchicalConfusionMatrix  # noqa: E402


def build_graph(ctx):
    g = nx.DiGraph()
    for axis in fl.AXES:
        for n in ctx.axis_tax[axis].nodes.values():
            child = f'{axis}:{n.id}'
            parent = 'root' if n.parent is None else f'{axis}:{n.parent}'
            g.add_edge(parent, child)
    return g


def path_of(graph, name):
    return nx.shortest_path(graph, 'root', name)


def metrics_from_confusion(tp, tn, fp, fn):
    ppv = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * ppv * rec / (ppv + rec) if ppv + rec else 0.0
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / math.sqrt(den) if den > 0 else None
    return ppv, rec, f1, mcc


class HCMScorer:
    def __init__(self, ctx):
        self.graph = build_graph(ctx)

    def confusion(self, pred, ref):
        true_labels = [f'{a}:{n}' for a, n in sorted(ref)]
        pred_paths = [path_of(self.graph, f'{a}:{n}') for a, n in sorted(pred)]
        return [int(x) for x in determineHierarchicalConfusionMatrix(self.graph, true_labels, pred_paths)]

    def __call__(self, pred, ref):
        if not pred and not ref:
            return {'P': 1.0, 'R': 1.0, 'F': 1.0, 'MCC': None}
        if not pred or not ref:
            # official code requires >=1 pred path; an empty prediction is all-FN, empty ref all-FP
            tp = tn = 0
            fp = sum(len(path_of(self.graph, f'{a}:{n}')) - 1 for a, n in pred)
            fn = sum(len(path_of(self.graph, f'{a}:{n}')) - 1 for a, n in ref)
            m = (tp, tn, fp, fn)
        else:
            m = self.confusion(pred, ref)
        ppv, rec, f1, mcc = metrics_from_confusion(*m)
        return {'P': ppv, 'R': rec, 'F': f1, 'MCC': mcc, '_tp': m[0], '_tn': m[1], '_fp': m[2], '_fn': m[3]}
