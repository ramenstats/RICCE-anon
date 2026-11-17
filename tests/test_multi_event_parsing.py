"""Unit tests for HybridParser's opt-in multi-event clause splitting
(parse(caption, allow_multi_event=True)), added for the v2
multi-event benchmark. Default parse(caption) behaviour (allow_multi_event
defaults to False) must remain byte-identical to the original single-event
parser -- every already-published result in this project depends on that.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ricce.ontology.tree import Taxonomy, Node
from ricce.parsing.hybrid import HybridParser


def object_tax():
    return Taxonomy([
        Node('entity', 'entity', None, []),
        Node('built_environment', 'built environment', 'entity', []),
        Node('building', 'building', 'built_environment', ['buildings', 'house', 'houses']),
        Node('transportation', 'transportation infrastructure', 'built_environment', []),
        Node('road', 'road', 'transportation', ['roads', 'lane', 'street']),
        Node('vegetation', 'vegetation', 'entity', ['plants']),
        Node('tree', 'tree', 'vegetation', ['trees']),
    ], 'entity')


def action_tax():
    return Taxonomy([
        Node('temporal_state', 'temporal state', None, []),
        Node('no_change', 'no change', 'temporal_state', ['unchanged', 'no change has occurred']),
        Node('change', 'change', 'temporal_state', []),
        Node('appearance', 'appearance', 'change', ['appeared', 'appears', 'added', 'new']),
        Node('constructed', 'constructed', 'appearance', ['built', 'are built']),
        Node('disappearance', 'disappearance', 'change', ['disappeared', 'removed']),
        Node('demolished', 'demolished', 'disappearance', ['destroyed']),
    ], 'temporal_state')


def location_tax():
    return Taxonomy([
        Node('image', 'image', None, ['scene']),
        Node('left', 'left', 'image', []),
        Node('bottom', 'bottom', 'image', []),
        Node('center', 'center', 'image', []),
    ], 'image')


def quantity_tax():
    return Taxonomy([Node('quantity', 'quantity', None, [])], 'quantity')


def make_parser():
    return HybridParser(object_tax(), action_tax(), location_tax(), quantity_tax(), use_embeddings=False)


def test_default_parse_is_single_event_unchanged():
    p = make_parser()
    caption = 'a lane appears on the left and several houses are built at the bottom.'
    events = p.parse(caption)
    assert len(events) == 1


def test_compound_two_action_caption_splits_when_opted_in():
    p = make_parser()
    caption = 'a lane appears on the left and several houses are built at the bottom.'
    events = p.parse(caption, allow_multi_event=True)
    assert len(events) == 2
    actions = {e.action for e in events}
    assert actions == {'appearance', 'constructed'}
    locs = {e.location for e in events}
    assert locs == {'left', 'bottom'}


def test_compound_subject_single_action_stays_one_event():
    # "some houses and vegetation appear around the road" has ONE action verb
    # (appear) shared across a compound subject -- must NOT be split into two
    # events even with allow_multi_event=True.
    p = make_parser()
    caption = 'some houses and vegetation appear around the road.'
    events = p.parse(caption, allow_multi_event=True)
    assert len(events) == 1


def test_no_change_caption_not_split():
    p = make_parser()
    caption = 'no change has occurred.'
    events = p.parse(caption, allow_multi_event=True)
    assert len(events) == 1
    assert events[0].change_state == 'no_change'


def test_semicolon_compound_splits():
    p = make_parser()
    caption = 'a road is built through the scene; some trees are removed.'
    events = p.parse(caption, allow_multi_event=True)
    assert len(events) == 2
    actions = {e.action for e in events}
    assert actions == {'constructed', 'disappearance'}


def test_three_clause_caption_splits_into_three():
    p = make_parser()
    caption = 'a house is built on the left and a tree is removed at the center and a road appears at the bottom.'
    events = p.parse(caption, allow_multi_event=True)
    assert len(events) == 3


def test_single_clause_no_conjunction_not_split():
    p = make_parser()
    caption = 'a new building is built at the center.'
    events = p.parse(caption, allow_multi_event=True)
    assert len(events) == 1
