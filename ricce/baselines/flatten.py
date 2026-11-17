"""Shared machinery for the flat (event-agnostic) prior-work baselines.

A caption is parsed with the SAME HybridParser RICCE uses; event grouping
is then discarded and every resolved node of every field of every event is
collected into one *set* of axis-prefixed labels, e.g. ``("object","building")``.
Axis prefixes act as per-axis super-roots, so labels of different axes can
never overlap (a source=building label never matches a target=building one).

Conventions (documented deviations from RICCE, not from the prior work):
  * exact_count is ignored; quantity enters only through its taxonomy bin.
  * change_state is a 2-leaf taxonomy (change_state -> {change, no_change}).
  * labels that the parser left unresolved (None) or that are outside a
    taxonomy are dropped.
  * the taxonomy root of each axis IS kept as an ancestor (same as
    Taxonomy.hierarchical_scores used by RICCE); the global ROOT is not.
"""
from __future__ import annotations
from pathlib import Path
import yaml
from ..ontology.tree import Taxonomy, Node
from ..parsing.hybrid import HybridParser

ROOT_DIR = Path(__file__).resolve().parents[2]
AXES = ['object', 'action', 'source', 'target', 'location', 'quantity', 'change_state']
GLOBAL_ROOT = 'ROOT'


def _change_state_tax():
    return Taxonomy([Node('change_state', 'change state', None, []),
                     Node('change', 'change', 'change_state', []),
                     Node('no_change', 'no change', 'change_state', [])], 'change_state')


class FlatContext:
    """Holds taxonomies, weights and one (lazily built) parser; caches parses."""

    def __init__(self, cfg_path=None, use_embeddings=True, parser=None):
        cfg = yaml.safe_load(open(cfg_path or ROOT_DIR / 'configs' / 'project.yaml'))
        self.cfg = cfg
        self.taxes = {k: Taxonomy.from_yaml(ROOT_DIR / v) for k, v in cfg['ontology'].items()}
        self.weights = cfg['metric']['weights']
        self._use_emb = use_embeddings
        self._parser = parser
        self._cache = {}
        self.axis_tax = {'object': self.taxes['object'], 'action': self.taxes['action'],
                         'source': self.taxes['object'], 'target': self.taxes['object'],
                         'location': self.taxes['location'], 'quantity': self.taxes['quantity'],
                         'change_state': _change_state_tax()}

    @property
    def parser(self):
        if self._parser is None:
            p = self.cfg['parser']; t = self.taxes
            self._parser = HybridParser(t['object'], t['action'], t['location'], t['quantity'],
                                        use_embeddings=self._use_emb and p.get('use_embeddings', True),
                                        embedding_model=p.get('embedding_model'),
                                        similarity_threshold=float(p.get('similarity_threshold', 0.58)))
        return self._parser

    def parse(self, text):
        if text not in self._cache:
            self._cache[text] = self.parser.parse(text, allow_multi_event=True)
        return self._cache[text]

    def labels(self, text):
        return flatten_events(self.parse(text), self)

    def ancestors(self, axis, node):
        """Axis-prefixed root->node path (taxonomy root included)."""
        return [(axis, a) for a in self.axis_tax[axis].ancestors(node)]


def flatten_events(events, ctx):
    out = set()
    for e in events:
        for axis in AXES:
            v = getattr(e, axis, None)
            if v and v in ctx.axis_tax[axis].nodes:
                out.add((axis, v))
    return out
