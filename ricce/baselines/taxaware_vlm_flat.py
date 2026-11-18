from __future__ import annotations
import re
import numpy as np
from ..metrics.event import _field_pr, aggregate_pr
from ..parsing.hybrid import HybridParser, NUMBER_WORDS
from .flatten import FlatContext, AXES

_norm = HybridParser.normalize
EMB_THRESHOLD = 0.58
MARGIN = 0.05
TOPK = 3
EMB_AXES = {'object', 'action', 'location', 'quantity'}
_FAMILIES = ('appearance', 'disappearance')


class Mapper:
    def __init__(self, ctx: FlatContext):
        self.ctx = ctx
        self._lex = {}
        self._cache = {}

    def _lexicon(self, tax_key):
        if tax_key not in self._lex:
            tax = self.ctx.taxes[tax_key]
            ph = {}
            for nid, p in tax.labels_and_aliases():
                p = _norm(p)
                if p:
                    ph.setdefault(p, set()).add(nid)
            # ambiguous phrase (same phrase, several nodes): keep deepest node
            self._lex[tax_key] = [(p, max(ids, key=lambda n: tax.depth(n))) for p, ids in ph.items()]
        return self._lex[tax_key]

    def lexical(self, text, tax_key, skip_root=True):
        """Dominant phrase-containment match (official stage 3, D1/D2)."""
        tax = self.ctx.taxes[tax_key]
        stats = {}
        for p, nid in self._lexicon(tax_key):
            ms = list(re.finditer(rf'\b{re.escape(p)}\b', text))
            if ms and not (skip_root and nid == tax.root):
                # a longer phrase covering the same span suppresses shorter ones
                cnt, ln, first = stats.get(nid, (0, 0, 10**9))
                stats[nid] = (cnt + len(ms), max(ln, len(p)), min(first, ms[0].start()))
        if not stats:
            return None
        return max(stats, key=lambda n: (stats[n][0], stats[n][1], -stats[n][2]))

    def embedding(self, text, tax_key):
        emb = self.ctx.parser.embedder
        if emb is None:
            return None
        tax = self.ctx.taxes[tax_key]
        ids, lab = self.ctx.parser._label_embeddings(tax)
        te = emb.encode([text], normalize_embeddings=True)[0]
        s = lab @ te
        order = np.argsort(-s)[:TOPK]
        if s[order[0]] < EMB_THRESHOLD:
            return None
        if len(order) > 1 and s[order[0]] - s[order[1]] < MARGIN:
            return tax.deepest_common_ancestor([ids[i] for i in order])
        return ids[order[0]]

    def map_axis(self, text, tax_key, allow_emb=True):
        key = (text, tax_key, allow_emb)
        if key not in self._cache:
            n = self.lexical(text, tax_key)
            if n is None and allow_emb:
                n = self.embedding(text, tax_key)
            self._cache[key] = n
        return self._cache[key]

    def quantity(self, text):
        q, _ = self.ctx.parser._quantity(text)  # bin only; exact_count discarded (D7)
        return q


def map_caption_axes(caption, mp: Mapper):
    """E2 mapping: at most one node per axis; NO event splitting."""
    ctx = mp.ctx; text = _norm(caption)
    obj = mp.map_axis(text, 'object')
    act = mp.map_axis(text, 'action')
    loc = mp.map_axis(text, 'location')
    qty = mp.quantity(text)
    if qty is None:
        qty = mp.map_axis(text, 'quantity', allow_emb=False)
    src = tgt = None
    fam = set(ctx.taxes['action'].ancestors(act)) if act else set()
    if act == 'conversion':
        m = HybridParser._CONVERSION_DIR_RE.search(text)
        if m:
            src = mp.map_axis(m.group(1).strip(), 'object', allow_emb=False)
            tgt = mp.map_axis(m.group(2).strip(), 'object', allow_emb=False)
    elif 'appearance' in fam:
        tgt = obj
    elif 'disappearance' in fam:
        src = obj
    cs = None
    if act == 'no_change' or 'no change' in text or 'unchanged' in text:
        cs = 'no_change'
    elif act:
        cs = 'change'
    if cs == 'no_change':  # mirror RICCE: a no-change statement carries no other field (D7)
        return {'object': None, 'action': 'no_change', 'source': None, 'target': None,
                'location': None, 'quantity': None, 'change_state': 'no_change'}
    return {'object': obj, 'action': act, 'source': src, 'target': tgt,
            'location': loc, 'quantity': qty, 'change_state': cs}


def _hpr(tax, a, b):
    if not a and not b: return None, None
    return _field_pr(tax, a, b)


def score_axisflat(candidate, reference, mp: Mapper):
    ctx = mp.ctx
    pa, ra = map_caption_axes(candidate, mp), map_caption_axes(reference, mp)
    fields = {}
    for ax in AXES:
        fields[ax] = _hpr(ctx.axis_tax[ax], pa[ax], ra[ax])
    p, r = aggregate_pr(fields, ctx.weights)
    if all(v == (None, None) for v in fields.values()):
        p = r = 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {'P': p, 'R': r, 'F': f}


def score_single(candidate, reference, mp: Mapper):
    tax = mp.ctx.taxes['object']
    a = mp.map_axis(_norm(candidate), 'object'); b = mp.map_axis(_norm(reference), 'object')
    if a is None and b is None: return {'P': 1.0, 'R': 1.0, 'F': 1.0}
    if a is None or b is None: return {'P': 0.0, 'R': 0.0, 'F': 0.0}
    hp, hr, hf = tax.hierarchical_scores(a, b)
    return {'P': hp, 'R': hr, 'F': hf}
