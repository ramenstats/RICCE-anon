"""Purity validator: re-parse reference and candidate and verify that a
perturbation changed ONLY the intended axis (all other parsed fields equal)."""
from collections import Counter

EVENT_FIELDS = ['object', 'action', 'source', 'target', 'location', 'quantity', 'exact_count', 'change_state', 'negated']

# family -> (allowed changed fields, required-changed fields (at least one))
SPEC = {
    'P0_identity': (set(), set()),
    'P0_paraphrase': (set(), set()),
    'P1_near_ancestor': ({'object', 'source', 'target'}, {'object'}),
    'P2_distant_ancestor': ({'object', 'source', 'target'}, {'object'}),
    'P3_sibling_substitution': ({'object', 'source', 'target'}, {'object'}),
    'unsupported_descendant_specificity': ({'object', 'source', 'target'}, {'object'}),
    'P4_action_reversal': ({'action', 'source', 'target'}, {'action'}),
    'P5_location_swap': ({'location'}, {'location'}),
    'P6_count_corruption': ({'quantity', 'exact_count'}, {'quantity', 'exact_count'}),
    'P9_change_state_reversal': (set(EVENT_FIELDS), {'change_state'}),
    'negation_flip': ({'change_state', 'negated', 'source', 'target'}, {'change_state', 'negated'}),
    'source_target_swap': ({'source', 'target'}, {'source', 'target'}),
}


def sig(e):
    return tuple(getattr(e, f) for f in EVENT_FIELDS)


def _diff(a, b):
    return {f for f in EVENT_FIELDS if getattr(a, f) != getattr(b, f)}


def check_purity(family, ref_events, cand_events, meta=None, taxes=None):
    from ..metrics.event import is_action_reversal
    meta = meta or {}
    out = {'family': family, 'n_ref_events': len(ref_events), 'n_cand_events': len(cand_events)}
    ok_intended, ok_other, notes = True, True, []
    if family == 'P9_change_state_reversal':
        r_no = bool(ref_events) and all(e.change_state == 'no_change' for e in ref_events)
        c_no = bool(cand_events) and all(e.change_state == 'no_change' for e in cand_events)
        r_ch = any(e.change_state == 'change' for e in ref_events)
        c_ch = any(e.change_state == 'change' for e in cand_events)
        ok_intended = (r_no and c_ch and not c_no) or (r_ch and not r_no and c_no and len(cand_events) == 1)
        out.update(intended_axis_changed=bool(ok_intended), other_axes_unchanged=True, pure=bool(ok_intended))
        return out
    if family in SPEC:
        allowed, required = SPEC[family]
        if len(ref_events) != len(cand_events):
            ok_other = False
            notes.append('event_count_changed')
        else:
            changed = set()
            for a, b in zip(ref_events, cand_events):
                changed |= _diff(a, b)
            unexpected = sorted(changed - allowed)
            out['changed_fields'] = sorted(changed)
            out['unexpected_changed_fields'] = unexpected
            ok_other = not unexpected
            ok_intended = (not required) or bool(changed & required)
            if family == 'P9_change_state_reversal':
                ok_intended = (all(e.change_state == 'no_change' for e in ref_events) !=
                               all(e.change_state == 'no_change' for e in cand_events))
            if family == 'P0_paraphrase' or family == 'P0_identity':
                ok_intended = True
            if family in ('P1_near_ancestor', 'P2_distant_ancestor', 'P3_sibling_substitution', 'unsupported_descendant_specificity'):
                exp = meta.get('expected_object')
                got = [b.object for a, b in zip(ref_events, cand_events) if a.object != b.object]
                out['expected_target_ok'] = bool(got) and got[0] == exp
                ok_intended = ok_intended and out['expected_target_ok']
            if family == 'P5_location_swap':
                exp = meta.get('expected_location')
                got = [b.location for a, b in zip(ref_events, cand_events) if a.location != b.location]
                out['expected_target_ok'] = bool(got) and got[0] == exp
                ok_intended = ok_intended and out['expected_target_ok']
            if family == 'P4_action_reversal':
                r = any(is_action_reversal(b, a, taxes) for a, b in zip(ref_events, cand_events))
                out['action_reversal_detected'] = r
                ok_intended = ok_intended and r
            if family == 'source_target_swap':
                a, b = ref_events[0], cand_events[0]
                sw = (b.source == a.target and b.target == a.source and a.source != a.target)
                out['exact_swap'] = sw
                ok_intended = ok_intended and sw
    else:
        rc, cc = Counter(map(sig, ref_events)), Counter(map(sig, cand_events))
        if family == 'P7_event_omission':
            ok_intended = len(cand_events) == len(ref_events) - 1
            ok_other = not (cc - rc)
        elif family == 'P8_event_hallucination':
            ok_intended = len(cand_events) == len(ref_events) + 1
            ok_other = not (rc - cc)
        elif family == 'multi_event_reorder':
            ok_other = (rc == cc)
            ok_intended = len(set(rc)) >= 2 and [sig(e) for e in ref_events] != [sig(e) for e in cand_events]
        elif family == 'multi_event_source_target_ambiguity':
            if len(ref_events) != len(cand_events):
                ok_other, notes = False, ['event_count_changed']
            else:
                changed = set()
                for a, b in zip(ref_events, cand_events):
                    changed |= _diff(a, b)
                out['changed_fields'] = sorted(changed)
                ok_other = changed <= {'object', 'source', 'target'}
                ok_intended = 'object' in changed and Counter(e.object for e in ref_events) == Counter(e.object for e in cand_events)
        else:
            raise KeyError(family)
    out['intended_axis_changed'] = bool(ok_intended)
    out['other_axes_unchanged'] = bool(ok_other)
    out['pure'] = bool(ok_intended and ok_other)
    if notes:
        out['notes'] = notes
    return out
