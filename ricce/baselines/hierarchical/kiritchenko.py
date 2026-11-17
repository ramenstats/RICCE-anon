"""Ancestor-augmented hierarchical P/R/F (Kiritchenko, Matwin, Famili 2006).

hP = |Aug(P) & Aug(T)| / |Aug(P)|,  hR = |Aug(P) & Aug(T)| / |Aug(T)|, where
Aug(S) is S together with all taxonomy ancestors of every label in S.
Applied to the flattened, axis-prefixed label SET of a caption (see
ricce.baselines.flatten): no event grouping, no event matching.

Two aggregations:
  * per-example (macro): hP/hR/hF computed per caption pair, then averaged;
  * micro: sums of |inter|, |Aug(P)|, |Aug(T)| over the evaluation set are
    accumulated first (`aggregate_micro`), exactly as in the original paper.
Empty-set convention: both empty -> 1/1/1; exactly one empty -> 0/0/0.
"""
from __future__ import annotations


def augment(labels, ctx):
    out = set()
    for axis, node in labels:
        out.update(ctx.ancestors(axis, node))
    return out


def _f(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def kiritchenko_labels(pred, ref, ctx):
    ap, ar = augment(pred, ctx), augment(ref, ctx)
    inter = len(ap & ar)
    if not ap and not ar:
        p = r = 1.0
    else:
        p = inter / len(ap) if ap else 0.0
        r = inter / len(ar) if ar else 0.0
    return {'P': p, 'R': r, 'F': _f(p, r), '_inter': inter, '_np': len(ap), '_nr': len(ar)}


def kiritchenko_text(candidate, reference, ctx):
    return kiritchenko_labels(ctx.labels(candidate), ctx.labels(reference), ctx)


def aggregate_micro(rows):
    """rows: iterable of dicts carrying _inter/_np/_nr. Returns micro P/R/F."""
    ti = sum(r['_inter'] for r in rows); tp = sum(r['_np'] for r in rows); tr = sum(r['_nr'] for r in rows)
    p = ti / tp if tp else 0.0
    r = ti / tr if tr else 0.0
    return {'P': p, 'R': r, 'F': _f(p, r)}
