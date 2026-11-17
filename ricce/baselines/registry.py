"""Uniform scoring API for RICCE and the prior-work baselines.

    ctx = BaselineContext()                       # one parser, shared caches
    score_pair('lca_f', candidate, reference, ctx) -> {'P':..,'R':..,'F':..,...}

Names: ricce, kiritchenko_micro, kiritchenko_macro, lca_f, hcm,
taxaware_single, taxaware_axisflat.  Per-item results for kiritchenko_micro and
kiritchenko_macro are identical (one pair); they differ only in aggregation --
use `aggregate(name, rows)` for set-level numbers (micro = ratio of sums).
Every returned dict carries P/R/F; keys starting with '_' are auxiliary.
"""
from __future__ import annotations
from ..metrics.event import ricce_scores
from .flatten import FlatContext
from .hierarchical import kiritchenko as K
from .hierarchical.lca_f import LCAScorer
from .hierarchical.hcm_adapter import HCMScorer
from .taxaware_vlm_flat import Mapper, score_single, score_axisflat

NAMES = ['ricce', 'kiritchenko_micro', 'kiritchenko_macro', 'lca_f', 'hcm',
         'taxaware_single', 'taxaware_axisflat']


class BaselineContext(FlatContext):
    def __init__(self, cfg_path=None, use_embeddings=True, parser=None, lca_backend='hemkit', tmpdir=None):
        super().__init__(cfg_path, use_embeddings, parser)
        self._lca = LCAScorer(self, lca_backend, tmpdir)
        self._hcm = HCMScorer(self)
        self._mapper = Mapper(self)

    @property
    def lca(self): return self._lca
    @property
    def hcm(self): return self._hcm
    @property
    def mapper(self): return self._mapper


def score_pair(name, candidate, reference, ctx):
    if name == 'ricce':
        s = ricce_scores(ctx.parse(candidate), ctx.parse(reference), ctx.taxes, ctx.weights)
        return {'P': s['ce_precision'], 'R': s['ce_recall'], 'F': s['ce_f1']}
    if name in ('kiritchenko_micro', 'kiritchenko_macro'):
        return K.kiritchenko_text(candidate, reference, ctx)
    if name == 'lca_f':
        return ctx.lca(ctx.labels(candidate), ctx.labels(reference))
    if name == 'hcm':
        return ctx.hcm(ctx.labels(candidate), ctx.labels(reference))
    if name == 'taxaware_single':
        return score_single(candidate, reference, ctx.mapper)
    if name == 'taxaware_axisflat':
        return score_axisflat(candidate, reference, ctx.mapper)
    raise KeyError(name)


def aggregate(name, rows):
    """Set-level P/R/F. rows = list of score_pair outputs."""
    if name == 'kiritchenko_micro':
        return K.aggregate_micro(rows)
    n = max(1, len(rows))
    return {k: sum(r[k] for r in rows) / n for k in ('P', 'R', 'F')}
