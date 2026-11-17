"""The final RICCE evaluator used for every headline number in the paper.

One place defines it, so no script can drift from it:
  * dual assignment: CE-P from an S_P-optimal one-to-one matching, CE-R from
    an independently S_R-optimal one (recall/precision monotone by
    construction; chosen over four single-assignment alternatives,
    scripts/run_dual_vs_joint_matching.py);
  * root as unspecified: a label equal to its axis's taxonomy root asserts
    nothing beyond the axis and is scored as an absent field; all other
    nodes are compared with the root removed from their ancestor sets
    (chosen over include / exclude / legacy-exclude on four a-priori
    properties, scripts/v3_root_mode_comparison.py);
  * not-scorable: a pair whose events are all empty on both sides is
    excluded from aggregates instead of being scored;
  * hard contradiction gate: a change/no-change disagreement or an action
    reversal (opposed action families, or a conversion with source and
    target swapped) zeroes S_P and S_R of that event pair; change-state and
    action-family contradictions count only between events about compatible
    objects (equal, ancestor/descendant, or unspecified), so two different
    events that an assignment happens to align are not treated as opposite
    claims (chosen on four a-priori criteria,
    scripts/v3_gate_object_match.py).

The implementation lives in event_v2.py; this module only fixes its
configuration and adds the gate-activation diagnostic.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .event_v2 import ricce_scores_v2, event_similarity_pr_v2, contradiction_kind

FINAL_CONFIG = dict(gate_mode='hard', dual_matching=True, root_mode='unspecified', gate_object_match=True)
# the formulation the previous manuscript version called final (kept for basis checks)
PREVIOUS_CONFIG = dict(gate_mode='hard', dual_matching=True, exclude_root=True)


def final_scores(pred_events, gt_events, taxes, weights, **override):
    return ricce_scores_v2(pred_events, gt_events, taxes, weights, **{**FINAL_CONFIG, **override})


def final_prf(pred_events, gt_events, taxes, weights, **override):
    """(CE-P, CE-R, CE-F), NaN triple for a not-scorable pair."""
    s = final_scores(pred_events, gt_events, taxes, weights, **override)
    if s['not_scorable']:
        return np.nan, np.nan, np.nan
    return s['ce_precision'], s['ce_recall'], s['ce_f1']


def gate_activations(pred_events, gt_events, taxes, weights, root_mode='unspecified', object_match=None):
    """Event pairs the hard gate acts on. Alignment is taken from the UNGATED
    similarities (the correspondence the evaluator would use if no gate
    existed); a gate activation is an aligned pair, under either the S_P- or
    the S_R-optimal assignment, whose events contradict each other under the
    evaluator's own contradiction test. Returns a list of (i, j, kind)."""
    if object_match is None:
        object_match = FINAL_CONFIG.get('gate_object_match', False)
    M, N = len(pred_events), len(gt_events)
    if M == 0 or N == 0:
        return []
    SP = np.zeros((M, N)); SR = np.zeros((M, N))
    for i, p in enumerate(pred_events):
        for j, g in enumerate(gt_events):
            SP[i, j], SR[i, j] = event_similarity_pr_v2(p, g, taxes, weights, gate_mode='none', root_mode=root_mode)
    pairs = set()
    for S in (SP, SR):
        r, c = linear_sum_assignment(-S)
        pairs |= {(int(i), int(j)) for i, j in zip(r, c)}
    out = []
    for i, j in sorted(pairs):
        k = contradiction_kind(pred_events[i], gt_events[j], taxes, object_match)
        if k:
            out.append((i, j, k))
    return out
