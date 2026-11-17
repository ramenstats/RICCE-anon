"""RICCE-Bench text perturbations. Each family function takes
(text, ctx, rng) and returns None (not applicable) or a dict with
candidate/changed_axis/expected_relation/severity_level (+ 'meta').
`ctx` = dict(taxes=..., parser=CachedParser, ref_events=[Event...]).
All randomness comes from the supplied random.Random (deterministic seed)."""
import re
from ..perturbations.generator import _find_taxonomy_mention, REVERSALS

FAMILIES = [
    'P0_identity', 'P0_paraphrase', 'P1_near_ancestor', 'P2_distant_ancestor', 'P3_sibling_substitution',
    'P4_action_reversal', 'P5_location_swap', 'P6_count_corruption', 'P7_event_omission',
    'P8_event_hallucination', 'P9_change_state_reversal', 'unsupported_descendant_specificity',
    'source_target_swap', 'negation_flip', 'multi_event_reorder', 'multi_event_source_target_ambiguity',
]

# A-priori ordinal design severity (documented assumption, not learned):
SEVERITY = {
    'P0_identity': 0, 'P0_paraphrase': 0, 'multi_event_reorder': 0,
    'P1_near_ancestor': 1, 'P2_distant_ancestor': 2, 'unsupported_descendant_specificity': 1,
    'P3_sibling_substitution': 3, 'P5_location_swap': 3, 'P6_count_corruption': 3,
    'P7_event_omission': 4, 'P8_event_hallucination': 4, 'multi_event_source_target_ambiguity': 4,
    'P4_action_reversal': 5, 'P9_change_state_reversal': 5, 'negation_flip': 5, 'source_target_swap': 5,
}


def _sub(text, s, e, new):
    return text[:s] + new + text[e:]


def surface(tax, nid, plural=False):
    n = tax.nodes[nid]
    forms = [n.label] + list(n.aliases)
    if plural:
        if n.label + 's' in forms:
            return n.label + 's'
        for f in forms:
            if f.endswith('s') and ' ' not in f.strip():
                return f
    return n.label


def _obj_mention(text, tax):
    hit = _find_taxonomy_mention(text, tax)
    if not hit:
        return None
    nid, s, e = hit
    if nid == tax.root:
        return None
    return nid, s, e


def _is_plural(surf):
    return surf.lower().endswith('s') and not surf.lower().endswith('ss')


def paraphrase(text, ctx, rng):
    for key in ('action', 'object'):
        tax = ctx['taxes'][key]
        hit = _find_taxonomy_mention(text, tax)
        if not hit:
            continue
        nid, s, e = hit
        n = tax.nodes[nid]
        cur = text[s:e].lower()
        alts = [v for v in [n.label] + list(n.aliases) if v.lower() != cur and v.strip()]
        if alts:
            return dict(candidate=_sub(text, s, e, rng.choice(alts)), changed_axis='none',
                        expected_relation='meaning_preserving', meta={'paraphrase_axis': key})
    return None


def ancestor(text, ctx, rng, hops):
    tax = ctx['taxes']['object']
    hit = _obj_mention(text, tax)
    if not hit:
        return None
    nid, s, e = hit
    anc = tax.ancestors(nid)
    idx = len(anc) - 1 - hops
    if idx < 0 or anc[idx] == nid:
        return None
    tgt = anc[idx]
    # root label 'entity' is not a natural surface form; use its alias 'land cover'
    surf = 'land cover' if tgt == tax.root else surface(tax, tgt, _is_plural(text[s:e]))
    return dict(candidate=_sub(text, s, e, surf), changed_axis='object',
                expected_relation='correct_but_less_specific(CE-P=1,CE-R<1)',
                meta={'from': nid, 'to': tgt, 'expected_object': tgt})


def descendant(text, ctx, rng):
    tax = ctx['taxes']['object']
    hit = _obj_mention(text, tax)
    if not hit:
        return None
    nid, s, e = hit
    kids = list(tax.children.get(nid, []))
    if not kids:
        return None
    tgt = rng.choice(sorted(kids))
    return dict(candidate=_sub(text, s, e, surface(tax, tgt, _is_plural(text[s:e]))), changed_axis='object',
                expected_relation='unsupported_over_specific(CE-P<1,CE-R=1)',
                meta={'from': nid, 'to': tgt, 'expected_object': tgt})


def sibling(text, ctx, rng):
    tax = ctx['taxes']['object']
    hit = _obj_mention(text, tax)
    if not hit:
        return None
    nid, s, e = hit
    par = tax.nodes[nid].parent
    if par is None:
        return None
    sibs = sorted(x for x in tax.children[par] if x != nid)
    if not sibs:
        return None
    tgt = rng.choice(sibs)
    return dict(candidate=_sub(text, s, e, surface(tax, tgt, _is_plural(text[s:e]))), changed_axis='object',
                expected_relation='wrong_sibling(CE-P<1,CE-R<1)', meta={'from': nid, 'to': tgt, 'expected_object': tgt})


_REV = {}
for _a, _b in REVERSALS.items():
    _REV[_a] = _b
_REV_RE = re.compile(r'\b(' + '|'.join(sorted(map(re.escape, _REV), key=len, reverse=True)) + r')\b', re.I)


def action_reversal(text, ctx, rng):
    m = _REV_RE.search(text)
    if not m:
        return None
    return dict(candidate=_sub(text, m.start(), m.end(), _REV[m.group(1).lower()]), changed_axis='action',
                expected_relation='contradiction(gate)', meta={'from': m.group(1), 'to': _REV[m.group(1).lower()]})


_OPP_LOC = {'left': 'right', 'right': 'left', 'upper_left': 'lower_right', 'lower_right': 'upper_left',
            'upper_right': 'lower_left', 'lower_left': 'upper_right', 'center_left': 'center_right',
            'center_right': 'center_left', 'upper_center': 'lower_center', 'lower_center': 'upper_center'}


def location_swap(text, ctx, rng):
    tax = ctx['taxes']['location']
    hit = _find_taxonomy_mention(text, tax)
    if not hit:
        return None
    nid, s, e = hit
    if nid not in _OPP_LOC:
        return None
    tgt = _OPP_LOC[nid]
    return dict(candidate=_sub(text, s, e, tax.nodes[tgt].label), changed_axis='location',
                expected_relation='wrong_location', meta={'from': nid, 'to': tgt, 'expected_location': tgt})


_NUMW = {'one': 'several', 'two': 'many', 'three': 'many', 'four': 'one', 'five': 'one', 'six': 'one', 'few': 'many',
         'several': 'many', 'multiple': 'two', 'numerous': 'two', 'many': 'two', 'some': 'many'}
_NUM_RE = re.compile(r'\b(\d+|one|two|three|four|five|six|few|several|multiple|numerous|many|some)\b', re.I)


def count_corruption(text, ctx, rng):
    m = _NUM_RE.search(text)
    if not m:
        return None
    w = m.group(1).lower()
    if w.isdigit():
        n = int(w)
        new = str(n + 8 if n < 5 else 1)
    else:
        new = _NUMW[w]
    return dict(candidate=_sub(text, m.start(), m.end(), new), changed_axis='quantity',
                expected_relation='wrong_quantity', meta={'from': w, 'to': new})


HALLUC_POOL = ['a new road was constructed', 'several trees were removed', 'a bridge appeared on the right',
               'the water body expanded', 'a parking area was built at the bottom', 'some buildings were demolished',
               'a new railway appeared', 'the forest contracted']


def _clauses(text, ctx):
    return ctx['parser']._split_clauses(text)


def _join(parts):
    return ' and '.join(parts) + ' .'


def omission(text, ctx, rng):
    if len(ctx['ref_events']) < 2:
        return None
    parts = _clauses(text, ctx)
    if not parts or len(parts) < 2:
        return None
    drop = rng.randrange(len(parts))
    return dict(candidate=_join([p for i, p in enumerate(parts) if i != drop]), changed_axis='event_set',
                expected_relation='incomplete(CE-R<1)', meta={'dropped_clause': parts[drop]})


def hallucination(text, ctx, rng):
    ref = ctx['ref_events']
    if not ref or any(e.change_state == 'no_change' for e in ref):
        return None
    pool = list(HALLUC_POOL)
    rng.shuffle(pool)
    txt = text.rstrip(' .')
    return dict(candidate=txt + ' , and ' + pool[0] + ' .', changed_axis='event_set',
                expected_relation='hallucinated(CE-P<1)', meta={'added_clause': pool[0]})


_CHANGE_POOL = ['a new building was constructed in the area', 'several trees were removed', 'a road appeared on the left',
                'many houses were built at the bottom', 'the river expanded']
_NOCHANGE_POOL = ['there is no change', 'no significant change occurred', 'the scene is unchanged', 'nothing has changed']


def change_state_reversal(text, ctx, rng):
    ref = ctx['ref_events']
    if not ref:
        return None
    if all(e.change_state == 'no_change' for e in ref):
        c = rng.choice(_CHANGE_POOL)
    elif any(e.change_state == 'no_change' for e in ref):
        return None
    else:
        c = rng.choice(_NOCHANGE_POOL)
    return dict(candidate=c + ' .', changed_axis='change_state', expected_relation='contradiction(gate)',
                meta={'direction': 'nochange->change' if all(e.change_state == 'no_change' for e in ref) else 'change->nochange'})


_CONV_RE = re.compile(r'(.+?)(\s+(?:was |were |is |are |has been |have been )?(?:converted into|converted to|transformed into|'
                      r'changed into|turned into|replaced by|replaced with)\s+)(.+)', re.I)


def source_target_swap(text, ctx, rng):
    ref = ctx['ref_events']
    if len(ref) != 1 or ref[0].action != 'conversion' or not (ref[0].source and ref[0].target) or ref[0].source == ref[0].target:
        return None
    m = _CONV_RE.search(text.lower())
    if not m:
        return None
    tax = ctx['taxes']['object']
    l, mid, r = m.group(1), m.group(2), m.group(3)
    hl, hr = _find_taxonomy_mention(l, tax), _find_taxonomy_mention(r, tax)
    if not hl or not hr or hl[0] == hr[0]:
        return None
    ls, le, rs, re_ = hl[1], hl[2], hr[1], hr[2]
    lsurf, rsurf = l[ls:le], r[rs:re_]
    new_l = l[:ls] + rsurf + l[le:]
    new_r = r[:rs] + lsurf + r[re_:]
    return dict(candidate=new_l + mid + new_r, changed_axis='source_target',
                expected_relation='contradiction(gate)', meta={'swapped': [hl[0], hr[0]]})


def negation_flip(text, ctx, rng):
    ref = ctx['ref_events']
    if not ref or any(e.change_state == 'no_change' or e.negated for e in ref):
        return None
    m = re.search(r'\b(is|are|was|were|has been|have been)\b', text, re.I)
    if m:
        cand = text[:m.end()] + ' not' + text[m.end():]
    else:
        cand = 'it is not the case that ' + text.strip()
    return dict(candidate=cand, changed_axis='change_state', expected_relation='contradiction(gate)',
                meta={'mode': 'aux_not' if m else 'prefix'})


def reorder(text, ctx, rng):
    ref = ctx['ref_events']
    if len(ref) < 2:
        return None
    parts = _clauses(text, ctx)
    if not parts or len(parts) != len(ref):
        return None
    new = parts[1:] + parts[:1]
    return dict(candidate=_join(new), changed_axis='event_order', expected_relation='order_invariant(equal)',
                meta={})


def binding_swap(text, ctx, rng):
    """Swap the object mentions between the first two clauses: bag of objects
    is unchanged but event-object bindings (who did what) are wrong."""
    ref = ctx['ref_events']
    if len(ref) < 2:
        return None
    parts = _clauses(text, ctx)
    if not parts or len(parts) != len(ref):
        return None
    tax = ctx['taxes']['object']
    h0, h1 = _find_taxonomy_mention(parts[0], tax), _find_taxonomy_mention(parts[1], tax)
    if not h0 or not h1 or h0[0] == h1[0]:
        return None
    s0, e0 = parts[0][h0[1]:h0[2]], parts[1][h1[1]:h1[2]]
    p0 = _sub(parts[0], h0[1], h0[2], e0)
    p1 = _sub(parts[1], h1[1], h1[2], s0)
    return dict(candidate=_join([p0, p1] + parts[2:]), changed_axis='event_binding',
                expected_relation='misbound_events(CE-P<1,CE-R<1)', meta={'swapped_objects': [h0[0], h1[0]]})


FAMILY_FN = {
    'P0_paraphrase': paraphrase,
    'P1_near_ancestor': lambda t, c, r: ancestor(t, c, r, 1),
    'P2_distant_ancestor': lambda t, c, r: ancestor(t, c, r, 2),
    'P3_sibling_substitution': sibling,
    'P4_action_reversal': action_reversal,
    'P5_location_swap': location_swap,
    'P6_count_corruption': count_corruption,
    'P7_event_omission': omission,
    'P8_event_hallucination': hallucination,
    'P9_change_state_reversal': change_state_reversal,
    'unsupported_descendant_specificity': descendant,
    'source_target_swap': source_target_swap,
    'negation_flip': negation_flip,
    'multi_event_reorder': reorder,
    'multi_event_source_target_ambiguity': binding_swap,
}
