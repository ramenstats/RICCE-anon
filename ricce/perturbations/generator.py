import re

REVERSALS = {'constructed': 'demolished', 'built': 'demolished', 'appeared': 'disappeared', 'added': 'removed',
             'demolished': 'constructed', 'destroyed': 'constructed', 'disappeared': 'appeared', 'removed': 'added',
             'expanded': 'contracted', 'increased': 'decreased', 'contracted': 'expanded', 'decreased': 'increased'}
LOCATION_SWAPS = {'upper left': 'lower right', 'upper-left': 'lower-right', 'top left': 'bottom right', 'left': 'right',
                   'upper right': 'lower left', 'upper-right': 'lower-left', 'top right': 'bottom left', 'right': 'left',
                   'lower left': 'upper right', 'lower-left': 'upper-right', 'lower right': 'upper left', 'lower-right': 'upper-left'}
SIBLING_OBJECTS = {'building': 'road', 'buildings': 'roads', 'road': 'building', 'roads': 'buildings',
                    'vegetation': 'water', 'tree': 'building', 'water': 'vegetation', 'cropland': 'road'}


def replace_first(text, mapping):
    low = text.lower()
    for a, b in mapping.items():
        i = low.find(a)
        if i >= 0:
            return text[:i] + b + text[i + len(a):]
    return None


def _find_taxonomy_mention(text, tax):
    """Longest word-boundary match of any label/alias from `tax` in `text`,
    returned as (node_id, start, end). Reuses the same longest-alias-wins
    rule as HybridParser._exact so perturbations and parsing agree on what a
    caption is "about"."""
    low = text.lower()
    best = None
    for nid, n in tax.nodes.items():
        for phrase in [n.label] + list(n.aliases):
            p = phrase.lower().strip()
            if not p:
                continue
            m = re.search(rf'\b{re.escape(p)}\b', low)
            if m and (best is None or (m.end() - m.start()) > (best[2] - best[1])):
                best = (nid, m.start(), m.end())
    return best


def synonym_paraphrase(text, taxes):
    """P0: a genuine meaning-preserving paraphrase, not the identity
    function. Finds an action or object mention and substitutes a
    *different* alias of the *same* taxonomy node -- by construction this
    cannot change which node the caption maps to, so it is a true
    paraphrase rather than a P1-style generalization."""
    for tax in (taxes['action'], taxes['object']):
        hit = _find_taxonomy_mention(text, tax)
        if not hit:
            continue
        nid, s, e = hit
        n = tax.nodes[nid]
        variants = [n.label] + list(n.aliases)
        cur = text[s:e].lower()
        alts = [v for v in variants if v.lower() != cur]
        if alts:
            return text[:s] + alts[0] + text[e:]
    return None


def ancestor_generalize(text, object_tax, hops=1):
    """P1 (hops=1) / P2 (hops=2): replace an object mention with its
    taxonomy ancestor `hops` levels up. Returns None if the object has no
    ancestor that far up (e.g. it is already within `hops` of the root)."""
    hit = _find_taxonomy_mention(text, object_tax)
    if not hit:
        return None
    nid, s, e = hit
    anc = object_tax.ancestors(nid)  # root-first, self-inclusive
    idx = len(anc) - 1 - hops
    if idx < 0:
        return None
    target = anc[idx]
    if target == nid:
        return None
    return text[:s] + object_tax.nodes[target].label + text[e:]


def action_reversal(text): return replace_first(text, REVERSALS)
def location_swap(text): return replace_first(text, LOCATION_SWAPS)
def sibling_substitution(text): return replace_first(text, SIBLING_OBJECTS)


def count_corruption(text):
    m = re.search(r'\b(\d+)\b', text)
    if m:
        n = int(m.group(1))
        n2 = n + 3 if n < 5 else max(1, n - 3)
        return text[:m.start()] + str(n2) + text[m.end():]
    return replace_first(text, {'one': 'several', 'two': 'many', 'three': 'many', 'few': 'many', 'several': 'one', 'many': 'one'})


NO_CHANGE_TEMPLATES = ('no change', 'unchanged', 'nothing has changed', 'same as before', 'seem identical', 'no difference')


def no_change_reversal(text):
    low = text.lower()
    return 'A new building was constructed in the area.' if any(p in low for p in NO_CHANGE_TEMPLATES) else 'No significant change occurred.'


def hallucinate_event(text): return text.rstrip('.') + ', and a new road was constructed.'


def omit_event(text):
    p = re.split(r',?\s+and\s+', text, maxsplit=1)
    return p[0].rstrip('.') + '.' if len(p) > 1 else text


def mixed_multi_axis(text, taxes):
    """P11: hard-negative curriculum level 6 -- compose two independent
    single-axis corruptions (object/sibling substitution + count corruption,
    falling back to action reversal + location swap) so the rejected caption
    differs from the chosen one on more than one taxonomy field at once,
    while staying just as fluent and lexically close as any single-axis
    corruption (each component substitution is itself fluency-preserving).
    This is deliberately *not* purity-checked against _ALLOWED_CHANGED_FIELDS
    since by construction it is allowed -- expected -- to touch multiple
    fields."""
    step1 = sibling_substitution(text) or action_reversal(text)
    if not step1:
        return None
    step2 = count_corruption(step1) or location_swap(step1)
    return step2 if (step2 and step2 != text) else None


# Curriculum difficulty level (design-review A4 hard-negative
# curriculum), 1 = coarsest/easiest-to-spot corruption, 6 = hardest/most
# entangled. P7 (omission) and P9 (no-change<->change reversal) are
# structural utility perturbations used elsewhere in the pipeline but are
# not part of the graded 1-6 ladder, so they are left unleveled (None).
CURRICULUM_LEVEL = {
    'P1_ancestor': 1, 'P2_distant_ancestor': 1,
    'P3_sibling': 2, 'P5_location_swap': 2, 'P6_count_corruption': 2,
    'P4_action_reversal': 3,
    'P10_source_target_swap': 4,  # image-order swap; injected by the trainer, not here
    'P8_hallucination': 5,
    'P11_mixed_multi_axis': 6,
    'P0_paraphrase': None, 'P0_original': None, 'P7_event_omission': None, 'P9_change_state_reversal': None,
}


def _field_ids(text, taxes):
    """Cheap exact/alias-only match per field, used only to police
    perturbation purity below -- not the full HybridParser (no embeddings,
    no negation handling), just enough to catch the failure mode this
    guards against: a substituted ontology label accidentally containing
    another field's alias as a substring (e.g. the object label "built
    environment" contains the action alias "built"), silently turning a
    single-field perturbation into a multi-field one and confounding what
    the perturbation is supposed to isolate."""
    return {k: (_find_taxonomy_mention(text, tax) or (None,))[0] for k, tax in taxes.items()}


# Which taxonomy fields each perturbation type is allowed to change; every
# other field's exact/alias match must be identical before and after, or the
# candidate is rejected (internal design review Sec.7.2 B3: "verify that
# perturbations preserve or alter exactly the intended semantic dimension").
_ALLOWED_CHANGED_FIELDS = {
    'P0_paraphrase': set(), 'P1_ancestor': {'object'}, 'P2_distant_ancestor': {'object'},
    'P3_sibling': {'object'}, 'P4_action_reversal': {'action'},
}


def _is_pure(ptype, orig_text, new_text, taxes):
    if ptype not in _ALLOWED_CHANGED_FIELDS:
        return True  # structural perturbations (omission/hallucination/reversal/count) aren't field-substitution based
    before, after = _field_ids(orig_text, taxes), _field_ids(new_text, taxes)
    allowed = _ALLOWED_CHANGED_FIELDS[ptype]
    return all(before[k] == after[k] for k in taxes if k not in allowed)


def generate_all(text, taxes):
    """taxes: the {'object','action','location','quantity'} Taxonomy dict
    used everywhere else in the pipeline, so perturbations are generated
    against the exact same ontology that scores them. Every substitution
    perturbation is purity-checked (see _is_pure) before being kept, so a
    lexical collision between two taxonomies' aliases can't silently turn
    a "P1 ancestor" sample into a sample that also corrupts the action
    field."""
    rows = [{'type': 'P0_original', 'text': text}]
    p0 = synonym_paraphrase(text, taxes)
    if p0 and p0 != text and _is_pure('P0_paraphrase', text, p0, taxes):
        rows.append({'type': 'P0_paraphrase', 'text': p0})
    p1 = ancestor_generalize(text, taxes['object'], hops=1)
    if p1 and p1 != text and _is_pure('P1_ancestor', text, p1, taxes):
        rows.append({'type': 'P1_ancestor', 'text': p1})
    p2 = ancestor_generalize(text, taxes['object'], hops=2)
    if p2 and p2 != text and p2 != p1 and _is_pure('P2_distant_ancestor', text, p2, taxes):
        rows.append({'type': 'P2_distant_ancestor', 'text': p2})
    subst_funcs = {'P3_sibling': sibling_substitution, 'P4_action_reversal': action_reversal}
    for k, f in subst_funcs.items():
        y = f(text)
        if y and y != text and _is_pure(k, text, y, taxes):
            rows.append({'type': k, 'text': y})
    structural_funcs = {'P5_location_swap': location_swap, 'P6_count_corruption': count_corruption,
                         'P7_event_omission': omit_event, 'P8_hallucination': hallucinate_event,
                         'P9_change_state_reversal': no_change_reversal}
    for k, f in structural_funcs.items():
        y = f(text)
        if y and y != text:
            rows.append({'type': k, 'text': y})
    p11 = mixed_multi_axis(text, taxes)
    if p11 and p11 != text:
        rows.append({'type': 'P11_mixed_multi_axis', 'text': p11})
    return rows


def generate_at_level(text, taxes, level):
    """Hard-negative curriculum helper: return only the candidates from
    generate_all(text, taxes) whose CURRICULUM_LEVEL equals `level`
    (1..6, per the A4 graded-difficulty ladder), or [] if none apply to
    this caption (e.g. it has no object mention for an ancestor-level
    corruption)."""
    return [r for r in generate_all(text, taxes) if CURRICULUM_LEVEL.get(r['type']) == level]
