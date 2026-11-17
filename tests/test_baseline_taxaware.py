import pytest
from ricce.baselines.flatten import FlatContext
from ricce.baselines.taxaware_vlm_flat import Mapper, score_single, score_axisflat, map_caption_axes

@pytest.fixture(scope='module')
def mp(): return Mapper(FlatContext(use_embeddings=False))

def test_lexical_mapping_and_dominance(mp):
    assert mp.lexical('roads and a road and a tree', 'object') == 'road'   # most frequent
    assert mp.lexical('a residential building', 'object') == 'residential_building'  # longest phrase

def test_single_object(mp):
    assert score_single('new houses appear', 'a house is built', mp)['F'] == 1
    r = score_single('a road appears', 'a river appears', mp)
    assert 0 < r['P'] < 1
    assert score_single('nothing', 'nothing here', mp)['F'] == 1  # both unmapped
    assert score_single('a road appears', 'nothing', mp)['F'] == 0

def test_axisflat_no_event_binding(mp):
    # two events: (road constructed left) + (trees demolished right) vs swapped objects -> flat cannot tell
    a = 'a road was built on the left and trees were removed on the right'
    b = 'trees were built on the left and a road was removed on the right'
    assert score_axisflat(a, a, mp)['F'] == 1
    m = map_caption_axes(a, mp)
    assert m['object'] in ('road', 'tree') and m['location'] in ('left', 'right')

def test_axisflat_no_change(mp):
    m = map_caption_axes('there is no difference', mp)
    assert m['change_state'] == 'no_change' and m['object'] is None
    assert score_axisflat('no change', 'there is no difference', mp)['F'] == 1
