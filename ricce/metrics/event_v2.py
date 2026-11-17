"""Parameterised RICCE evaluator.

The paper's final evaluator is this module run with the configuration in
ricce/metrics/final.py (FINAL_CONFIG); import it from there. The
historical evaluator (ricce/metrics/event.py) is unchanged and kept for
basis checks. Every option below defaults to the behaviour of the earliest
call site that used it, so all previously released numbers stay
reproducible bit for bit (verified on all RICCE-Bench items).

Options and the property comparisons that decided them:
  dual_matching    -- two separately optimal assignments (final) vs. one
                      shared assignment (scripts/v3_dual_vs_joint_matching.py)
  root_mode        -- 'unspecified' (final): a root-level label is an absent
                      field; 'include' / 'exclude' / 'legacy' for comparison
                      (scripts/v3_root_mode_comparison.py)
  gate_object_match-- contradictions only between events about compatible
                      objects (final) (scripts/v3_gate_object_match.py)
  gate_mode        -- 'hard' (final), 'soft' (scale by soft_gamma), 'none'
                      (scripts/v3_gate_analysis.py)

History of the three fixes this module first introduced:

  P1 (recall/precision monotonicity): the frozen evaluator matches once, by
     maximising S_F = harmonic(S_P, S_R), then reads BOTH the precision mass
     and the recall mass off that single assignment. Because Hungarian only
     guarantees the assignment is S_F-optimal, not S_P-optimal or
     S_R-optimal individually, removing a matched predicted event can change
     the feasible assignments in a way that raises the achieved S_R mass
     (or, symmetrically, removing a reference event can raise S_P mass) --
     this is the documented CE-R non-monotonicity. Fix tested here: solve
     TWO separate one-to-one assignments, one Hungarian-optimal for S_P
     (used only to compute CE-P and the hallucination set) and one for S_R
     (used only to compute CE-R and the omission set). Removing a row from a
     matching problem can only weakly decrease the optimal value of that
     same objective (the removed row's contribution is never required), so
     each of CE-P and CE-R is monotone in its own removal direction by
     construction. This changes what "the match" means (precision and
     recall may now point at different reference/prediction events for the
     same item), which is why it is evaluated as a variant, not silently
     substituted for the frozen matcher.
  P2 (root-only inflation): Taxonomy.hierarchical_scores(..., exclude_root=True)
     (see ontology/tree.py). A root-only prediction shares no post-exclusion
     ancestor with any specific reference node and scores 0 on that axis
     instead of receiving free credit for the taxonomy root.
  P3 (empty-parse identity): if every event on both the predicted and the
     reference side is an "empty" event (every field None -- the parser's
     representation of a caption it could not resolve into any structured
     content at all), the item is marked not_scorable=True with
     ce_precision/ce_recall/ce_f1 = None (NaN) rather than silently scored
     as either a perfect match or, as the frozen evaluator does, a total
     miss. Callers must exclude not_scorable items from any mean/aggregate
     rather than treating None as 0.
  G1/G2/G3 (gate strength): gate_mode='hard' reproduces the frozen gate
     (contradiction -> S_P=S_R=0); 'soft' scales the field-aggregated S_P/S_R
     by a fixed gamma in (0,1) instead of zeroing them; 'none' disables the
     contradiction gate entirely (pure hierarchical field scoring).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from ..schema import Event
from .event import is_action_reversal, quantity_pr, change_state_pr

_EMPTY_FIELDS = ('object', 'action', 'source', 'target', 'location', 'quantity', 'exact_count', 'change_state')


_ROOT_MODES = ('legacy', 'include', 'exclude', 'unspecified')
_TAX_OF = {'object': 'object', 'action': 'action', 'source': 'object', 'target': 'object',
           'location': 'location', 'quantity': 'quantity'}


def _resolve_root_mode(exclude_root, root_mode):
    """root_mode=None keeps the behaviour of every earlier call site exactly:
    exclude_root=True -> 'legacy' (root excluded on object/action/source/
    target/location; the quantity axis still went through the historical
    quantity_pr and so kept the root), exclude_root=False -> 'include'.
      'include'     : taxonomy root counted in ancestor sets on every axis.
      'exclude'     : root removed from ancestor sets on every axis
                      (identity kept for root==root).
      'unspecified' : a label equal to its taxonomy root is treated as an
                      absent field on that side (it asserts nothing beyond
                      the axis existing); remaining nodes are compared with
                      the root removed from their ancestor sets."""
    if root_mode is None:
        return 'legacy' if exclude_root else 'include'
    assert root_mode in _ROOT_MODES, root_mode
    return root_mode


def is_empty_event(e: Event, taxes=None, root_as_empty=False) -> bool:
    if not root_as_empty:
        return all(getattr(e, f) is None for f in _EMPTY_FIELDS)
    for f in _EMPTY_FIELDS:
        v = getattr(e, f)
        if v is None:
            continue
        if f in _TAX_OF and taxes is not None and v == taxes[_TAX_OF[f]].root:
            continue
        return False
    return True


def _field_pr_v2(tax, a, b, exclude_root, root_mode=None):
    mode = _resolve_root_mode(exclude_root, root_mode)
    if mode == 'unspecified':
        a = None if a == tax.root else a
        b = None if b == tax.root else b
    if not a and not b:
        return None, None
    if a and not b:
        return 0.0, None
    if b and not a:
        return None, 0.0
    if a not in tax.nodes or b not in tax.nodes:
        return 0.0, 0.0
    hp, hr, _ = tax.hierarchical_scores(a, b, exclude_root=(mode != 'include'))
    return hp, hr


def _quantity_pr_v2(pred, gt, qtax, exclude_root, root_mode):
    """Same exact-count rule as the historical quantity_pr; the binned
    quantity field goes through _field_pr_v2 so it follows root_mode like
    every other hierarchical axis. In 'legacy' mode the historical
    quantity_pr is called unchanged (bit-identical to earlier results)."""
    mode = _resolve_root_mode(exclude_root, root_mode)
    if mode == 'legacy':
        return quantity_pr(pred, gt, qtax)
    if pred.exact_count is not None and gt.exact_count is not None:
        m = max(pred.exact_count, gt.exact_count, 1)
        s = max(0.0, 1.0 - abs(pred.exact_count - gt.exact_count) / m)
        return s, s
    return _field_pr_v2(qtax, pred.quantity, gt.quantity, exclude_root, mode)


def event_field_scores_v2(pred, gt, taxes, exclude_root=True, root_mode=None):
    return {
        'object': _field_pr_v2(taxes['object'], pred.object, gt.object, exclude_root, root_mode),
        'action': _field_pr_v2(taxes['action'], pred.action, gt.action, exclude_root, root_mode),
        'source': _field_pr_v2(taxes['object'], pred.source, gt.source, exclude_root, root_mode),
        'target': _field_pr_v2(taxes['object'], pred.target, gt.target, exclude_root, root_mode),
        'location': _field_pr_v2(taxes['location'], pred.location, gt.location, exclude_root, root_mode),
        'quantity': _quantity_pr_v2(pred, gt, taxes['quantity'], exclude_root, root_mode),
        'change_state': change_state_pr(pred, gt),
    }


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


def objects_compatible(pred, gt, taxes):
    """Two events can contradict each other only if they are about the same
    thing: their object labels are equal, one is an ancestor of the other, or
    either is unspecified (absent or the object-taxonomy root)."""
    ot = taxes['object']
    a = None if pred.object == ot.root else pred.object
    b = None if gt.object == ot.root else gt.object
    if a is None or b is None:
        return True
    if a not in ot.nodes or b not in ot.nodes:
        return a == b
    return a in ot.ancestors(b) or b in ot.ancestors(a)


def contradiction_kind(pred, gt, taxes, object_match=False):
    """None, 'change_state', 'action_family' or 'conversion_swap'. With
    object_match=True, change-state and action-family contradictions count
    only between events about compatible objects (objects_compatible); the
    conversion-swap test already requires the swapped source/target to match."""
    if pred.action == gt.action == 'conversion' and is_action_reversal(pred, gt, taxes):
        return 'conversion_swap'
    if object_match and not objects_compatible(pred, gt, taxes):
        return None
    if pred.change_state and gt.change_state and pred.change_state != gt.change_state:
        return 'change_state'
    if is_action_reversal(pred, gt, taxes):
        return 'action_family'
    return None


def event_similarity_pr_v2(pred, gt, taxes, weights, gate_mode='hard', soft_gamma=0.1, exclude_root=True, root_mode=None,
                           gate_object_match=False):
    """Per-pair (S_P, S_R). gate_mode: 'hard' (contradiction -> 0), 'soft'
    (scale by soft_gamma), 'none' (no contradiction gate). gate_object_match
    restricts change-state / action-family contradictions to events about
    compatible objects (default False = behaviour of every earlier call)."""
    if gate_object_match:
        contradiction = contradiction_kind(pred, gt, taxes, object_match=True) is not None
    else:
        contradiction = bool(pred.change_state and gt.change_state and pred.change_state != gt.change_state)
        contradiction = contradiction or is_action_reversal(pred, gt, taxes)
    fields = event_field_scores_v2(pred, gt, taxes, exclude_root, root_mode)
    sp, sr = aggregate_pr(fields, weights)
    if contradiction:
        if gate_mode == 'hard':
            return 0.0, 0.0
        if gate_mode == 'soft':
            return sp * soft_gamma, sr * soft_gamma
        # gate_mode == 'none': fall through, no penalty beyond the field scores
    return sp, sr


def _score_matrix(pred_events, gt_events, taxes, weights, gate_mode, soft_gamma, exclude_root, root_mode=None, gate_object_match=False):
    M, N = len(pred_events), len(gt_events)
    SP = np.zeros((M, N), dtype=float)
    SR = np.zeros((M, N), dtype=float)
    for i, p in enumerate(pred_events):
        for j, g in enumerate(gt_events):
            sp, sr = event_similarity_pr_v2(p, g, taxes, weights, gate_mode, soft_gamma, exclude_root, root_mode, gate_object_match)
            SP[i, j] = sp
            SR[i, j] = sr
    return SP, SR


_JOINT_FNS = {
    'harmonic': lambda sp, sr: np.divide(2 * sp * sr, sp + sr, out=np.zeros_like(sp), where=(sp + sr) > 0),
    'mean': lambda sp, sr, alpha=0.5: alpha * sp + (1 - alpha) * sr,
    'min': lambda sp, sr: np.minimum(sp, sr),
    'geometric': lambda sp, sr: np.sqrt(np.clip(sp, 0, None) * np.clip(sr, 0, None)),
}


def match_events_v2(pred_events, gt_events, taxes, weights, gate_mode='hard', soft_gamma=0.1,
                     exclude_root=True, dual_matching=True, joint_fn='harmonic', joint_alpha=0.5, root_mode=None,
                     gate_object_match=False):
    """dual_matching=True (V2-A / P1 fix): solve separate Hungarian
    assignments maximising total S_P and total S_R independently, so CE-P
    and CE-R may correspond to different event correspondences.
    dual_matching=False (V2-B, joint assignment): solve ONE Hungarian
    assignment maximising a single combined objective (joint_fn: 'harmonic'
    reproduces the frozen single-S_F-assignment behaviour; 'mean' is
    joint_alpha*S_P+(1-joint_alpha)*S_R; 'min'; 'geometric' is sqrt(S_P*S_R)),
    and CE-P/CE-R are then read off that one shared assignment. See
    Appendix (dual-vs-joint matching) for the controlled comparison this
    parameterisation supports and the adoption decision."""
    M, N = len(pred_events), len(gt_events)
    if M == 0 or N == 0:
        z = np.zeros((M, N), dtype=float)
        return [], [], z, z
    SP, SR = _score_matrix(pred_events, gt_events, taxes, weights, gate_mode, soft_gamma, exclude_root, root_mode, gate_object_match)
    if dual_matching:
        rP, cP = linear_sum_assignment(-SP)
        rR, cR = linear_sum_assignment(-SR)
        pairs_p = [(int(i), int(j), float(SP[i, j])) for i, j in zip(rP, cP)]
        pairs_r = [(int(i), int(j), float(SR[i, j])) for i, j in zip(rR, cR)]
        return pairs_p, pairs_r, SP, SR
    fn = _JOINT_FNS[joint_fn]
    J = fn(SP, SR, joint_alpha) if joint_fn == 'mean' else fn(SP, SR)
    r, c = linear_sum_assignment(-J)
    pairs = [(int(i), int(j), float(SP[i, j])) for i, j in zip(r, c)]
    pairs_r = [(int(i), int(j), float(SR[i, j])) for i, j in zip(r, c)]
    return pairs, pairs_r, SP, SR


def ricce_scores_v2(pred_events, gt_events, taxes, weights, gate_mode='hard', soft_gamma=0.1,
                         exclude_root=True, dual_matching=True, joint_fn='harmonic', joint_alpha=0.5, root_mode=None,
                         gate_object_match=False):
    rae = _resolve_root_mode(exclude_root, root_mode) == 'unspecified'
    pred_empty_all = all(is_empty_event(e, taxes, rae) for e in pred_events) if pred_events else True
    gt_empty_all = all(is_empty_event(e, taxes, rae) for e in gt_events) if gt_events else True
    if pred_empty_all and gt_empty_all:
        return {'ce_precision': None, 'ce_recall': None, 'ce_f1': None, 'not_scorable': True,
                'matches_p': [], 'matches_r': [], 'num_pred_events': len(pred_events), 'num_gt_events': len(gt_events)}
    pairs_p, pairs_r, SP, SR = match_events_v2(pred_events, gt_events, taxes, weights, gate_mode,
                                                soft_gamma, exclude_root, dual_matching, joint_fn, joint_alpha, root_mode,
                                                gate_object_match)
    p_mass = sum(x[2] for x in pairs_p)
    r_mass = sum(x[2] for x in pairs_r)
    p = p_mass / max(1, len(pred_events))
    r = r_mass / max(1, len(gt_events))
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        'ce_precision': p, 'ce_recall': r, 'ce_f1': f, 'not_scorable': False,
        'matches_p': pairs_p, 'matches_r': pairs_r,
        'num_pred_events': len(pred_events), 'num_gt_events': len(gt_events),
        'hallucination_rate': max(0, len(pred_events) - len(pairs_p)) / max(1, len(pred_events)),
        'omission_rate': max(0, len(gt_events) - len(pairs_r)) / max(1, len(gt_events)),
    }
