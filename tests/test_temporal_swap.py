import pytest
from ricce.training.temporal_swap import reverse_caption, RHO_ACTION


@pytest.mark.parametrize('src,dst,grp', [
    ('some houses are built at the corner of the road .', 'some houses are demolished at the corner of the road .', 'app'),
    ('a road has been constructed in the jungle .', 'a road has been demolished in the jungle .', 'app'),
    ('many buildings appear around the road .', 'many buildings disappear around the road .', 'app'),
    ('a road has appeared .', 'a road has disappeared .', 'app'),
    ('some trees are removed .', 'some trees appear .', 'dis'),
    ('a small house is removed .', 'a small house appears .', 'dis'),
    ('trees disappear .', 'trees appear .', 'dis'),
    ('the road has been widened .', 'the road has been narrowed .', 'exp'),
    ('the trees have been reduced .', 'the trees have been increased .', 'con'),
    ('vegetation has decreased .', 'vegetation has increased .', 'con'),
])
def test_reverse(src, dst, grp):
    out, info = reverse_caption(src)
    assert out == dst and info['group'] == grp


@pytest.mark.parametrize('src', [
    'a new house is built .',                          # 'new' not covered
    'houses are built and trees are removed .',        # compound directions
    'the bareland has been replaced by houses .',      # conversion
    'the ground has flattened out .',                  # uncovered verb
    'no houses are built .',                           # negation
    'a house stands here .',                           # no directional word
    'the road is extended .',                          # uncovered
])
def test_dropped(src):
    out, info = reverse_caption(src)
    assert out is None and isinstance(info, str)


def test_involution_of_families():
    for a, b in RHO_ACTION.items():
        assert RHO_ACTION[b] == a


def test_purity_with_real_parser():
    from ricce.utils import load_yaml
    from ricce.ontology.tree import Taxonomy
    from ricce.parsing.hybrid import HybridParser
    from ricce.training.temporal_swap import check_purity
    cfg = load_yaml('configs/project.yaml')
    t = {k: Taxonomy.from_yaml(v) for k, v in cfg['ontology'].items()}
    P = HybridParser(t['object'], t['action'], t['location'], t['quantity'], use_embeddings=False)
    for s in ['some houses are built at the corner of the road .', 'many buildings appear around the road .',
              'some trees are removed .', 'the road has been widened .']:
        r, _ = reverse_caption(s)
        ok, why, a, b = check_purity(s, r, P, t)
        assert ok, (s, r, why, a, b)
    ok, why, _, _ = check_purity('some houses are built .', 'some houses are built .', P, t)
    assert not ok
