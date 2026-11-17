"""Mapper-validation v2 helpers.

Human labels in results/mapper_v1/mapper_validation/e03_mapper_validation_to_fill.csv
are single-annotator FREE TEXT, not ontology node ids. Everything here that
compares parser nodes with human labels is therefore a PROXY that depends on a
conservative phrase->node conversion (see HumanMapper). True node-level labels
must come from artifacts_cvpr_final/mapper_v2/node_label_template.csv.

Human-phrase -> node procedure (deliberately conservative, no guessing):
  1. normalize the WHOLE phrase with HybridParser.normalize (lowercase, '-'->' ',
     strip punctuation, collapse spaces);
  2. exact match against the normalized labels/aliases of the axis taxonomy
     (Taxonomy.labels_and_aliases()); object/source/target use the object
     taxonomy. No stemming, no substring, no embeddings;
  3. a phrase matching aliases of >1 distinct nodes is 'ambiguous' (unmappable);
  4. no exact match -> 'unmappable' (never guessed);
  5. quantity only: a bare integer n is binned with the SAME rule as
     HybridParser._quantity (0 none, 1 single, 2-3 few, 4-6 several, >6 many)
     and exact_count=n; multi-valued cells ('4; 2') are 'unmappable';
  6. change_state: exactly 'change' or 'no_change'.
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')  # keep the shared GPU free unless caller overrides
import math
from pathlib import Path
import pandas as pd
import yaml
from ..ontology.tree import Taxonomy
from ..parsing.hybrid import HybridParser
from ..schema import Event
from ..utils import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
AXES = ['object', 'action', 'source', 'target', 'location', 'quantity', 'change_state']
DATASETS = ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC']
LABEL_CSV = ROOT / 'results/mapper_v1/mapper_validation/e03_mapper_validation_to_fill.csv'
AXIS_TAX = {'object': 'object', 'source': 'object', 'target': 'object', 'action': 'action',
            'location': 'location', 'quantity': 'quantity'}
norm = HybridParser.normalize


def load_cfg():
    return yaml.safe_load(open(ROOT / 'configs/project.yaml'))


def load_taxes(cfg=None):
    cfg = cfg or load_cfg()
    return {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}


def build_parser(taxes, cfg=None, **kw):
    cfg = cfg or load_cfg()
    p = dict(cfg['parser']); p.update(kw)
    return HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'], **p)


def canonical_image_id(dataset, image_id):
    """SECOND_CC ids lost leading zeros in the label CSV; manifests use 7 digits."""
    image_id = str(image_id)
    return image_id.zfill(7) if dataset == 'SECOND_CC' and image_id.isdigit() else image_id


def load_labels():
    """The 480 unique human-labelled captions (duplicate '__retest' rows dropped)."""
    d = pd.read_csv(LABEL_CSV, dtype=str)
    d = d[~d['id'].str.endswith('__retest')].copy()
    d['image_id'] = [canonical_image_id(a, b) for a, b in zip(d['dataset'], d['image_id'])]
    for c in AXES:
        d[c] = pd.Series([v.strip() if isinstance(v, str) and v.strip() else None for v in d[c]], dtype=object, index=d.index)
    return d.reset_index(drop=True)


def label_records(d):
    """Rows as dicts with NaN -> None (pandas re-introduces NaN in frames)."""
    return [{k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()} for r in d.to_dict('records')]


def quantity_bin(n):
    if n == 0: return 'none'
    if n == 1: return 'single'
    if n <= 3: return 'few'
    if n <= 6: return 'several'
    return 'many'


class HumanMapper:
    def __init__(self, taxes):
        self.taxes = taxes
        self.index = {}
        for k, tax in taxes.items():
            idx = {}
            for nid, ph in tax.labels_and_aliases():
                idx.setdefault(norm(ph), set()).add(nid)
            self.index[k] = idx

    def map(self, axis, phrase):
        """-> (node_or_None, status, exact_count); status in absent/mapped/unmappable/ambiguous."""
        if phrase is None or not str(phrase).strip():
            return None, 'absent', None
        if axis == 'change_state':
            v = norm(phrase).replace(' ', '_')
            return (v, 'mapped', None) if v in ('change', 'no_change') else (None, 'unmappable', None)
        tk = AXIS_TAX[axis]
        if axis == 'quantity':
            s = str(phrase).strip()
            if s.isdigit():
                return quantity_bin(int(s)), 'mapped', int(s)
        hits = self.index[tk].get(norm(phrase), set())
        if len(hits) == 1:
            return next(iter(hits)), 'mapped', None
        return None, ('ambiguous' if hits else 'unmappable'), None

    def map_row(self, row):
        out = {}
        for ax in AXES:
            node, st, cnt = self.map(ax, row.get(ax))
            out[ax] = {'node': node, 'status': st, 'exact_count': cnt}
        return out

    @staticmethod
    def fully_mappable(mapped):
        return all(v['status'] in ('absent', 'mapped') for v in mapped.values())

    @staticmethod
    def to_event(mapped, caption=''):
        """Human-structured Event. no_change is encoded like the parser does
        (action='no_change'), since the human schema carries it in change_state."""
        g = lambda a: mapped[a]['node']
        if g('change_state') == 'no_change':
            return Event(change_state='no_change', action='no_change', text=caption)
        return Event(object=g('object'), action=g('action'), source=g('source'), target=g('target'),
                     location=g('location'), quantity=g('quantity'), exact_count=mapped['quantity']['exact_count'],
                     change_state=g('change_state'), text=caption)


def parser_field(ev, axis):
    return getattr(ev, axis)


def relation(tax, parser_node, human_node):
    """How the parser node relates to the human node."""
    if parser_node == human_node: return 'exact'
    if tax is None or parser_node not in tax.nodes or human_node not in tax.nodes: return 'other'
    if parser_node in tax.ancestors(human_node): return 'ancestor'      # parser coarser
    if human_node in tax.ancestors(parser_node): return 'descendant'    # parser finer
    if tax.nodes[parser_node].parent == tax.nodes[human_node].parent: return 'sibling'
    return 'other'


def classify_axis(axis, human_phrase, human_node, hstatus, parser_node, human_nochange=False):
    """Categories a..e (+f): a_null_correct, b_consistent, c_different,
    d_unresolved, e_spurious, f_resolved_human_unmappable."""
    if axis == 'action' and human_phrase is None and human_nochange and parser_node == 'no_change':
        return 'a_null_correct'  # human schema encodes no-change via change_state, not action
    if human_phrase is None:
        return 'a_null_correct' if parser_node is None else 'e_spurious'
    if parser_node is None: return 'd_unresolved'
    if hstatus != 'mapped': return 'f_resolved_human_unmappable'
    return 'b_consistent' if parser_node == human_node else 'c_different'


def wilson(k, n, z=1.96):
    if n == 0: return (None, None)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))
