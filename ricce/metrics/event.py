import numpy as np
from scipy.optimize import linear_sum_assignment

# Preference-order (reversed) action pairs: a hard contradiction gate, not a
# hierarchical-distance penalty, because e.g. "constructed" vs "demolished"
# are taxonomic siblings but describe opposite temporal events. Kept as a
# literal fallback for callers that pass bare action ids with no taxonomy
# (e.g. some unit tests); is_action_reversal below generalizes this to whole
# action *families* via the taxonomy so that e.g. "constructed" (a child of
# appearance) vs "disappearance" (used directly, with no more specific leaf
# resolved) is also caught, not just exact-leaf-vs-exact-leaf pairs.
ACTION_REVERSALS = {('constructed', 'demolished'), ('demolished', 'constructed'),
                     ('appearance', 'disappearance'), ('disappearance', 'appearance'),
                     ('expansion', 'contraction'), ('contraction', 'expansion')}

# Top-level action families that stand in opposition; a resolved action
# belongs to a family if the family node is one of its taxonomy ancestors
# (so both the family node itself, e.g. "appearance", and its more specific
# children, e.g. "constructed", count as that family).
_FAMILY_REVERSALS = {('appearance', 'disappearance'), ('disappearance', 'appearance'),
                      ('expansion', 'contraction'), ('contraction', 'expansion')}
_FAMILY_NODES = {'appearance', 'disappearance', 'expansion', 'contraction'}


def _action_family(action, action_tax):
    if not action or action_tax is None or action not in action_tax.nodes:
        return action
    for anc in action_tax.ancestors(action):
        if anc in _FAMILY_NODES:
            return anc
    return action


def _field_pr(tax, a, b):
    """Hierarchical precision/recall for one ontology field, kept separate
    (never collapsed to a single scalar before aggregation). Missing/unknown
    handling: both absent -> field omitted from both P and R (None, None);
    predicted-but-not-in-reference -> precision penalty only, recall N/A for
    this field (0.0, None); reference-but-not-predicted -> recall penalty
    only, precision N/A (None, 0.0); either label outside the taxonomy
    (parser-unknown) -> explicit full miss on both sides (0.0, 0.0)."""
    if not a and not b:
        return None, None
    if a and not b:
        return 0.0, None
    if b and not a:
        return None, 0.0
    if a not in tax.nodes or b not in tax.nodes:
        return 0.0, 0.0
    hp, hr, _ = tax.hierarchical_scores(a, b)
    return hp, hr


def quantity_pr(pred, gt, qtax):
    if pred.exact_count is not None and gt.exact_count is not None:
        m = max(pred.exact_count, gt.exact_count, 1)
        s = max(0.0, 1.0 - abs(pred.exact_count - gt.exact_count) / m)
        return s, s  # numeric closeness has no natural P/R asymmetry
    return _field_pr(qtax, pred.quantity, gt.quantity)


def change_state_pr(pred, gt):
    if pred.change_state and gt.change_state:
        v = 1.0 if pred.change_state == gt.change_state else 0.0
        return v, v
    if pred.change_state and not gt.change_state:
        return 0.0, None
    if gt.change_state and not pred.change_state:
        return None, 0.0
    return None, None


def is_action_reversal(pred, gt, taxes=None):
    """True when pred/gt describe opposite temporal events: either a
    family-level action reversal (appearance<->disappearance, expansion<->
    contraction -- computed via the action taxonomy's ancestors so a leaf
    like "constructed" is recognized as belonging to the "appearance"
    family even when compared against the bare "disappearance" node, not
    just exact-leaf-vs-exact-leaf pairs), or the same directional action
    (currently: conversion) with source and target swapped between pred and
    gt -- e.g. "vegetation converted into building" vs "building converted
    into vegetation" describe opposite real-world transitions even though
    both resolve to the single action node "conversion" and would otherwise
    be scored as a perfect action-field match. Falls back to the literal
    ACTION_REVERSALS pair set when no action taxonomy is available (e.g.
    bare-id callers in unit tests)."""
    action_tax = taxes.get('action') if taxes else None
    if action_tax is not None:
        pf, gf = _action_family(pred.action, action_tax), _action_family(gt.action, action_tax)
        if (pf, gf) in _FAMILY_REVERSALS:
            return True
    elif (pred.action, gt.action) in ACTION_REVERSALS:
        return True
    if pred.action == gt.action == 'conversion':
        if pred.source and pred.target and gt.source and gt.target and pred.source != pred.target:
            if pred.source == gt.target and pred.target == gt.source:
                return True
    return False


def event_field_scores(pred, gt, taxes):
    fields = {
        'object': _field_pr(taxes['object'], pred.object, gt.object),
        'action': _field_pr(taxes['action'], pred.action, gt.action),
        'source': _field_pr(taxes['object'], pred.source, gt.source),
        'target': _field_pr(taxes['object'], pred.target, gt.target),
        'location': _field_pr(taxes['location'], pred.location, gt.location),
        'quantity': quantity_pr(pred, gt, taxes['quantity']),
        'change_state': change_state_pr(pred, gt),
    }
    if is_action_reversal(pred, gt, taxes):
        fields['action'] = (0.0, 0.0)
        fields['source'] = (0.0, 0.0)
        fields['target'] = (0.0, 0.0)
    return fields


def aggregate_pr(fields, weights):
    np_, dp = 0.0, 0.0
    nr, dr = 0.0, 0.0
    for k, (hp, hr) in fields.items():
        w = float(weights.get(k, 1.0))
        if hp is not None:
            np_ += w * hp
            dp += w
        if hr is not None:
            nr += w * hr
            dr += w
    sp = np_ / dp if dp else 0.0
    sr = nr / dr if dr else 0.0
    return sp, sr


def event_similarity_pr(pred, gt, taxes, weights):
    """Returns (S_P, S_R): separate hierarchical precision and recall for one
    predicted/reference event pair, preserved through to the final CE-P/CE-R
    (see match_events / ricce_scores). Two hard gates apply to the
    *whole* event rather than being absorbed into the weighted field
    average, since both describe getting the fundamental temporal judgment
    wrong rather than a partial/noisy match: (1) a top-level CHANGE-vs-
    NO_CHANGE contradiction, and (2) an action-reversal contradiction
    (constructed<->demolished, appeared<->disappeared, expanded<->
    contracted, or a conversion with source/target swapped) -- an event
    that asserts the opposite of what actually happened should not earn
    partial credit through unrelated fields like matching object/location.
    Zeroing only the action/source/target fields (the previous behavior)
    left 5-6 unrelated weighted fields (object, location, quantity,
    change_state) free to still score close to 1.0, so a hand-written
    reversal stress test found predicted ce_f1 sitting at 0.57-0.71 for
    exactly the pairs this gate exists to catch -- see
    artifacts_r1/action_reversal_stress.jsonl and
    results_r1/parser_repair/ for the before/after evidence."""
    if pred.change_state and gt.change_state and pred.change_state != gt.change_state:
        return 0.0, 0.0
    if is_action_reversal(pred, gt, taxes):
        return 0.0, 0.0
    fields = event_field_scores(pred, gt, taxes)
    return aggregate_pr(fields, weights)


def event_similarity(pred, gt, taxes, weights):
    """Backward-compatible scalar (harmonic mean of S_P, S_R): used only to
    rank candidate matches, never to compute CE-P/CE-R directly."""
    sp, sr = event_similarity_pr(pred, gt, taxes, weights)
    return 2 * sp * sr / (sp + sr) if (sp + sr) else 0.0


def match_events(pred_events, gt_events, taxes, weights):
    """Hungarian-matches on S_F = harmonic(S_P, S_R) so a single predicted
    event cannot double-count against two reference events, then returns the
    matched pairs together with their separate S_P/S_R (not just S_F) so
    CE-P and CE-R can be computed from the precision and recall mass
    respectively, rather than from one shared scalar."""
    M, N = len(pred_events), len(gt_events)
    if M == 0 or N == 0:
        return [], np.zeros((M, N), dtype=float)
    SP = np.zeros((M, N), dtype=float)
    SR = np.zeros((M, N), dtype=float)
    SF = np.zeros((M, N), dtype=float)
    for i, p in enumerate(pred_events):
        for j, g in enumerate(gt_events):
            sp, sr = event_similarity_pr(p, g, taxes, weights)
            SP[i, j] = sp
            SR[i, j] = sr
            SF[i, j] = 2 * sp * sr / (sp + sr) if (sp + sr) else 0.0
    r, c = linear_sum_assignment(-SF)
    pairs = [(int(i), int(j), float(SP[i, j]), float(SR[i, j]), float(SF[i, j])) for i, j in zip(r, c)]
    return pairs, SF


def match_events_greedy(pred_events, gt_events, taxes, weights):
    """Greedy matching baseline for the Hungarian-vs-greedy comparison:
    repeatedly picks the single highest-S_F remaining (pred, gt) cell and
    removes both its row and column, until no pairs remain. This is the
    textbook greedy bipartite-matching heuristic -- locally optimal at each
    step but not globally optimal, unlike linear_sum_assignment in
    match_events above. Returns the same (pairs, SF) shape as match_events
    so downstream scoring code (ricce_scores) is agnostic to which
    matcher produced the pairs."""
    M, N = len(pred_events), len(gt_events)
    if M == 0 or N == 0:
        return [], np.zeros((M, N), dtype=float)
    SP = np.zeros((M, N), dtype=float)
    SR = np.zeros((M, N), dtype=float)
    SF = np.zeros((M, N), dtype=float)
    for i, p in enumerate(pred_events):
        for j, g in enumerate(gt_events):
            sp, sr = event_similarity_pr(p, g, taxes, weights)
            SP[i, j] = sp
            SR[i, j] = sr
            SF[i, j] = 2 * sp * sr / (sp + sr) if (sp + sr) else 0.0
    used_rows, used_cols = set(), set()
    pairs = []
    cells = sorted(((SF[i, j], i, j) for i in range(M) for j in range(N)), key=lambda x: -x[0])
    for score, i, j in cells:
        if i in used_rows or j in used_cols:
            continue
        used_rows.add(i)
        used_cols.add(j)
        pairs.append((int(i), int(j), float(SP[i, j]), float(SR[i, j]), float(SF[i, j])))
    return pairs, SF


def match_events_order(pred_events, gt_events, taxes, weights):
    """Naive order-based baseline: predicted event k is matched to reference
    event k for k in range(min(M, N)), with no regard to score at all --
    i.e. purely positional assignment. Included as the weakest-possible
    baseline alongside Hungarian and greedy."""
    M, N = len(pred_events), len(gt_events)
    if M == 0 or N == 0:
        return [], np.zeros((M, N), dtype=float)
    SP = np.zeros((M, N), dtype=float)
    SR = np.zeros((M, N), dtype=float)
    SF = np.zeros((M, N), dtype=float)
    for i, p in enumerate(pred_events):
        for j, g in enumerate(gt_events):
            sp, sr = event_similarity_pr(p, g, taxes, weights)
            SP[i, j] = sp
            SR[i, j] = sr
            SF[i, j] = 2 * sp * sr / (sp + sr) if (sp + sr) else 0.0
    pairs = [(k, k, float(SP[k, k]), float(SR[k, k]), float(SF[k, k])) for k in range(min(M, N))]
    return pairs, SF


_MATCHERS = {'hungarian': match_events, 'greedy': match_events_greedy, 'order': match_events_order}


def ricce_scores(pred_events, gt_events, taxes, weights, matcher='hungarian'):
    match_fn = _MATCHERS[matcher] if isinstance(matcher, str) else matcher
    pairs, _ = match_fn(pred_events, gt_events, taxes, weights)
    p_mass = sum(x[2] for x in pairs)
    r_mass = sum(x[3] for x in pairs)
    p = p_mass / max(1, len(pred_events))
    r = r_mass / max(1, len(gt_events))
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        'ce_precision': p, 'ce_recall': r, 'ce_f1': f,
        'matched_precision_mass': p_mass, 'matched_recall_mass': r_mass,
        'matches': pairs, 'num_pred_events': len(pred_events), 'num_gt_events': len(gt_events),
        'hallucination_rate': max(0, len(pred_events) - len(pairs)) / max(1, len(pred_events)),
        'omission_rate': max(0, len(gt_events) - len(pairs)) / max(1, len(gt_events)),
    }
