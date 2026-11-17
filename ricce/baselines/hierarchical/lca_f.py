"""LCA-P / LCA-R / LCA-F (Kosmopoulos, Partalas, Gaussier, Paliouras,
Androutsopoulos 2015, "Evaluation measures for hierarchical classification: a
unified view through two generic frameworks") on the flattened label set.

Backends
  * 'hemkit'  (default, REFERENCE): the authors' own C++ HEMKit
    (third_party/Evaluation-Measures/hierarchical/software, BioASQ mirror),
    called once per unique (pred, ref) pair. Built with `make` on first use.
  * 'python'  (EXPERIMENTAL port): a simplified deterministic re-implementation
    for trees. It is cross-checked against HEMKit in
    tests/test_baseline_lca_f.py; agreement statistics are in
    results_cvpr_final/baselines/lca_validation/. Use only for speed / fixtures.

Global hierarchy: ROOT -> <axis taxonomy root> -> ... with axis-prefixed node
names (see flatten.py).  Because a single global root is required by the
'flattened labels' setting, unmatched labels may be linked to the closest
label of ANOTHER axis through ROOT (path of length >=2 above the axis roots);
this is inherent to LCA-F on a flattened hierarchy and is documented.
Empty-set convention: both empty -> 1/1/1; exactly one empty -> 0/0/0.
"""
from __future__ import annotations
import os, subprocess, tempfile, subprocess as sp
from pathlib import Path
from functools import lru_cache
from .. import flatten as fl

HEMKIT_DIR = fl.ROOT_DIR / 'third_party' / 'Evaluation-Measures' / 'hierarchical'
HEMKIT_BIN = HEMKIT_DIR / 'bin' / 'HEMKit'


def ensure_hemkit():
    if not HEMKIT_BIN.exists():
        (HEMKIT_DIR / 'bin').mkdir(exist_ok=True)
        sp.run(['make'], cwd=HEMKIT_DIR / 'software', check=True, capture_output=True)
    return HEMKIT_BIN


class GlobalHierarchy:
    def __init__(self, ctx):
        self.ids = {fl.GLOBAL_ROOT: 1}
        self.parent = {}
        self.edges = []
        def nid(name):
            if name not in self.ids: self.ids[name] = len(self.ids) + 1
            return self.ids[name]
        for axis in fl.AXES:
            tax = ctx.axis_tax[axis]
            for n in tax.nodes.values():
                name = f'{axis}:{n.id}'
                par = fl.GLOBAL_ROOT if n.parent is None else f'{axis}:{n.parent}'
                self.parent[name] = par
                self.edges.append((nid(par), nid(name)))
        self.names = {v: k for k, v in self.ids.items()}
        self.parent[fl.GLOBAL_ROOT] = None
        self._hier_text = '\n'.join(f'{a} {b}' for a, b in self.edges) + '\n'

    def name(self, label):
        return f'{label[0]}:{label[1]}'

    def path_to_root(self, name):
        out = []
        while name is not None:
            out.append(name); name = self.parent[name]
        return out  # node ... ROOT


def _empty_rule(pred, ref):
    if not pred and not ref: return {'P': 1.0, 'R': 1.0, 'F': 1.0}
    if not pred or not ref: return {'P': 0.0, 'R': 0.0, 'F': 0.0}
    return None


def lca_hemkit(pred, ref, H, tmpdir=None):
    e = _empty_rule(pred, ref)
    if e: return e
    binp = ensure_hemkit()
    with tempfile.TemporaryDirectory(dir=tmpdir or os.environ.get('CE_TMPDIR')) as d:
        d = Path(d)
        (d / 'h.txt').write_text(H._hier_text)
        (d / 'g.txt').write_text(' '.join(str(H.ids[H.name(l)]) for l in sorted(ref)) + '\n')
        (d / 'p.txt').write_text(' '.join(str(H.ids[H.name(l)]) for l in sorted(pred)) + '\n')
        out = sp.run([str(binp), str(d / 'h.txt'), str(d / 'g.txt'), str(d / 'p.txt'), '100000', '5'],
                     capture_output=True, text=True, check=True).stdout.split()
    hp, hr, hf, lp, lr, lf = map(float, out[:6])
    return {'P': lp, 'R': lr, 'F': lf, '_hemkit_hP': hp, '_hemkit_hR': hr, '_hemkit_hF': hf}


def lca_python(pred, ref, H):
    """Simplified LCA-F for a tree (EXPERIMENTAL; see module docstring)."""
    e = _empty_rule(pred, ref)
    if e: return e
    P0 = {H.name(l) for l in pred}; T0 = {H.name(l) for l in ref}
    def reduce(S):  # drop labels that are ancestors of other labels of the same set
        anc = set()
        for n in S: anc.update(H.path_to_root(n)[1:])
        return {n for n in S if n not in anc}
    P, T = reduce(P0), reduce(T0)
    AP, AT = set(P & T), set(P & T)
    def link(src_set, other_set, aug_src, aug_other):
        for g in sorted(src_set - other_set):
            pg = H.path_to_root(g); best = None
            for p in sorted(other_set):
                pp = H.path_to_root(p)
                anc = next(a for a in pg if a in set(pp))
                d = pg.index(anc) + pp.index(anc)
                if best is None or d < best[0]: best = (d, anc, pp)
            _, anc, pp = best
            aug_src.update(pg[:pg.index(anc) + 1]); aug_other.update(pp[:pp.index(anc) + 1])
    link(T, P, AT, AP); link(P, T, AP, AT)
    inter = len(AP & AT)
    p, r = inter / len(AP), inter / len(AT)
    return {'P': p, 'R': r, 'F': 2 * p * r / (p + r) if p + r else 0.0}


class LCAScorer:
    """Caches per unique (pred, ref) label-set pair."""
    def __init__(self, ctx, backend='hemkit', tmpdir=None):
        self.H = GlobalHierarchy(ctx); self.backend = backend; self.tmpdir = tmpdir; self.cache = {}
    def __call__(self, pred, ref):
        k = (frozenset(pred), frozenset(ref))
        if k not in self.cache:
            self.cache[k] = (lca_hemkit(pred, ref, self.H, self.tmpdir) if self.backend == 'hemkit'
                             else lca_python(pred, ref, self.H))
        return self.cache[k]
