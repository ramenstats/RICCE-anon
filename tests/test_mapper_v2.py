from ricce.evaluation.mapper_v2 import (HumanMapper, load_taxes, classify_axis, relation, wilson, canonical_image_id, quantity_bin, label_records)
import pandas as pd

TAXES = load_taxes()
HM = HumanMapper(TAXES)

def test_conservative_mapping():
    assert HM.map('object', 'Houses')[:2] == ('building', 'mapped')
    assert HM.map('object', 'artificial island')[1] == 'unmappable'   # never guessed
    assert HM.map('action', 'appeared')[0] == 'appearance'
    assert HM.map('object', None)[1] == 'absent'
    assert HM.map('quantity', '2') == ('few', 'mapped', 2)
    assert HM.map('quantity', '4; 2')[1] == 'unmappable'
    assert HM.map('change_state', 'no_change')[0] == 'no_change'
    assert quantity_bin(7) == 'many' and quantity_bin(0) == 'none'

def test_classify_axis_categories():
    assert classify_axis('object', None, None, 'absent', None) == 'a_null_correct'
    assert classify_axis('object', None, None, 'absent', 'road') == 'e_spurious'
    assert classify_axis('object', 'road', 'road', 'mapped', None) == 'd_unresolved'
    assert classify_axis('object', 'road', 'road', 'mapped', 'road') == 'b_consistent'
    assert classify_axis('object', 'road', 'road', 'mapped', 'bridge') == 'c_different'
    assert classify_axis('object', 'desert', None, 'unmappable', 'soil') == 'f_resolved_human_unmappable'
    assert classify_axis('action', None, None, 'absent', 'no_change', human_nochange=True) == 'a_null_correct'

def test_relation_and_event():
    t = TAXES['object']
    assert relation(t, 'building', 'residential_building') == 'ancestor'
    assert relation(t, 'residential_building', 'building') == 'descendant'
    assert relation(t, 'road', 'bridge') == 'sibling'
    m = HM.map_row({'object': None, 'action': None, 'source': None, 'target': None, 'location': None, 'quantity': None, 'change_state': 'no_change'})
    assert HM.fully_mappable(m) and HM.to_event(m).action == 'no_change'

def test_misc():
    assert canonical_image_id('SECOND_CC', '12520') == '0012520' and canonical_image_id('DUBAI_CC', '8x38') == '8x38'
    lo, hi = wilson(5, 10); assert 0 < lo < 0.5 < hi < 1 and wilson(0, 0) == (None, None)
    assert label_records(pd.DataFrame({'a': [float('nan'), 'x']})) == [{'a': None}, {'a': 'x'}]
