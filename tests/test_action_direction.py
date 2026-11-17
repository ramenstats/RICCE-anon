"""Unit tests for the repaired action-reversal contradiction gate: the
family-generalized is_action_reversal() in ricce/metrics/event.py, the
negation handling and directional conversion source/target resolution in
HybridParser (ricce/parsing/hybrid.py), and the enlarged alias coverage
in configs/ontology/action.yaml.

These are deliberately independent of the v2 stress-test JSONL
(artifacts_r1/action_reversal_stress.jsonl) -- that file is a
large, generated regression fixture; this file is a small, hand-picked set
of one or two representative cases per reversal family plus the two
specific bugs the repair fixes (bare-infinitive negation forms, and
whole-event vs field-only gating), meant to fail loudly and specifically if
either regresses.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
from ricce.metrics.event import ricce_scores, is_action_reversal

CFG = yaml.safe_load(open(ROOT / 'configs' / 'project.yaml'))
TAXES = {k: Taxonomy.from_yaml(ROOT / v) for k, v in CFG['ontology'].items()}
WEIGHTS = CFG['metric']['weights']
_P = CFG['parser']


def make_parser():
    # use_embeddings=False: these tests exercise the exact/alias resolution
    # path deterministically, independent of whether a sentence-transformers
    # model is installed/downloadable in the test environment.
    return HybridParser(TAXES['object'], TAXES['action'], TAXES['location'], TAXES['quantity'],
                         use_embeddings=False)


def gated(gt_text, pred_text, parser=None):
    parser = parser or make_parser()
    ge = parser.parse(gt_text, allow_multi_event=True)
    pe = parser.parse(pred_text, allow_multi_event=True)
    sc = ricce_scores(pe, ge, TAXES, WEIGHTS)
    return sc['ce_f1'] < 0.15, sc['ce_f1']


# --- one representative pair per reversal family, both directions ---

def test_appearance_disappearance_gates():
    g, f1 = gated("a house appeared next to the river", "a house disappeared next to the river")
    assert g, f'expected gate, got ce_f1={f1}'
    g, f1 = gated("a house disappeared next to the river", "a house appeared next to the river")
    assert g, f'expected gate, got ce_f1={f1}'


def test_constructed_demolished_gates():
    g, f1 = gated("a building was constructed near the road", "a building was demolished near the road")
    assert g, f'expected gate, got ce_f1={f1}'


def test_constructed_vs_bare_disappearance_family_gates():
    # "constructed" is a *child* of the "appearance" family; this checks the
    # family-level gate generalization (is_action_reversal via taxonomy
    # ancestors), not just the exact-leaf ACTION_REVERSALS pair.
    g, f1 = gated("a building was constructed near the road", "a building disappeared near the road")
    assert g, f'expected gate, got ce_f1={f1}'


def test_expansion_contraction_gates():
    g, f1 = gated("the parking lot expanded to the north", "the parking lot contracted to the north")
    assert g, f'expected gate, got ce_f1={f1}'


def test_conversion_source_target_swap_gates():
    g, f1 = gated("the vegetation was converted into a building", "the building was converted into vegetation")
    assert g, f'expected gate, got ce_f1={f1}'


def test_conversion_same_direction_does_not_gate():
    g, f1 = gated("the vegetation was converted into a building", "vegetation was replaced by a building")
    assert not g, f'expected no gate, got ce_f1={f1}'


# --- negation ---

def test_negation_did_not_construct_gates_against_positive():
    g, f1 = gated("a building was constructed near the road", "a building was not constructed near the road")
    assert g, f'expected gate, got ce_f1={f1}'


def test_negation_no_x_form_gates_against_positive():
    g, f1 = gated("a building was constructed near the road", "no building was constructed near the road")
    assert g, f'expected gate, got ce_f1={f1}'


def test_negation_bare_infinitive_form_resolves_action():
    # Regression test for the specific bug found during the repair: "did not
    # contract" uses the bare infinitive "contract", which was previously
    # absent from action.yaml's alias list, so the action field silently
    # failed to resolve (action=None) and the gate never engaged.
    parser = make_parser()
    ev = parser.parse("the reservoir did not contract to the south", allow_multi_event=True)[0]
    assert ev.action == 'contraction', f'expected contraction, got {ev.action}'
    assert ev.negated is True
    assert ev.change_state == 'no_change'


def test_no_longer_present_is_not_misread_as_negation():
    # "no longer present" is a genuine disappearance alias, not a negated
    # action -- the negation regex must not treat the "no" in "no longer" as
    # a negation trigger.
    parser = make_parser()
    ev = parser.parse("a house is no longer present next to the river", allow_multi_event=True)[0]
    assert ev.action == 'disappearance', f'expected disappearance, got {ev.action}'
    assert ev.negated is False
    assert ev.change_state == 'change'


def test_negation_same_polarity_does_not_gate():
    g, f1 = gated("a building was not constructed near the road", "no building was constructed near the road")
    assert not g, f'expected no gate (both negated), got ce_f1={f1}'


# --- whole-event vs field-only gating (the core scoring-logic fix) ---

def test_reversal_gates_whole_event_not_just_action_field():
    # Regression test for the root cause found during the repair: zeroing
    # only the action field left object/location/quantity/change_state free
    # to still score close to 1.0, so ce_f1 for a genuine reversal with
    # everything else identical sat at ~0.6 instead of near 0.
    g, f1 = gated("a building was constructed in the center of the image",
                  "a building was demolished in the center of the image")
    assert g, f'expected near-zero ce_f1 for a full reversal, got {f1}'
    assert f1 < 0.15


def test_is_action_reversal_direct():
    from ricce.schema import Event
    pred = Event(action='demolished')
    gt = Event(action='constructed')
    assert is_action_reversal(pred, gt, TAXES)
    pred2 = Event(action='constructed')
    gt2 = Event(action='constructed')
    assert not is_action_reversal(pred2, gt2, TAXES)


# --- false-contradiction guard: paraphrase / identical pairs must NOT gate ---

def test_identical_captions_do_not_gate():
    g, f1 = gated("a small building was constructed near the parking area",
                  "a small building was constructed near the parking area")
    assert not g, f'expected no gate for identical captions, got ce_f1={f1}'


def test_paraphrase_does_not_gate():
    g, f1 = gated("a new building was built on the bare land", "a building was constructed on the bare land")
    assert not g, f'expected no gate for a paraphrase, got ce_f1={f1}'


def test_quantity_location_modifiers_do_not_break_reversal_gate():
    g, f1 = gated("three buildings were constructed in the upper right corner",
                  "three buildings were demolished in the upper right corner")
    assert g, f'expected gate despite quantity/location modifiers, got ce_f1={f1}'


def test_compound_sentence_full_reversal_gates():
    g, f1 = gated("houses were built and trees appeared near the lake",
                  "houses were demolished and trees disappeared near the lake")
    assert g, f'expected gate when every clause reverses, got ce_f1={f1}'


def test_compound_sentence_partial_reversal_does_not_zero_whole_caption():
    # Only one of two clauses reverses; a correctly-designed event-level
    # metric should give partial credit, not gate the entire caption to
    # near-zero (that would be indistinguishable from a caption that got
    # everything wrong).
    g, f1 = gated("houses were built and trees were removed near the lake",
                  "houses were demolished and trees were removed near the lake")
    assert not g, f'expected partial credit (not a full gate), got ce_f1={f1}'
    assert 0.0 < f1 < 0.9
