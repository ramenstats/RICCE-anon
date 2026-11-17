"""Shared loader: taxonomies, metric weights and a single reusable HybridParser
(embeddings are cached per parser instance, so reuse one per process)."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_cfg():
    return yaml.safe_load(open(ROOT / 'configs' / 'project.yaml'))


def load_taxes():
    from ..ontology.tree import Taxonomy
    cfg = load_cfg()
    return {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}


def load_weights():
    return dict(load_cfg()['metric']['weights'])


def make_parser(taxes=None, use_embeddings=True):
    from ..parsing.hybrid import HybridParser
    cfg = load_cfg()
    taxes = taxes or load_taxes()
    p = cfg['parser']
    return HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'],
                        use_embeddings=use_embeddings and p.get('use_embeddings', True),
                        embedding_model=p.get('embedding_model'),
                        similarity_threshold=float(p.get('similarity_threshold', 0.58)))


class CachedParser:
    """Wraps HybridParser.parse with a per-(text, multi) cache."""
    def __init__(self, parser):
        self.parser = parser
        self._c = {}

    def parse(self, text, allow_multi_event=True):
        k = (text, allow_multi_event)
        if k not in self._c:
            self._c[k] = self.parser.parse(text, allow_multi_event=allow_multi_event)
        return self._c[k]

    def __getattr__(self, n):
        return getattr(self.parser, n)
