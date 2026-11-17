import pandas as pd
import pytest
from ricce.evaluation.ordering import build_constraints, pairwise_ordering_accuracy
from ricce.benchmarks.purity import check_purity
from ricce.benchmarks.env import load_taxes
from ricce.schema import Event

TAXES = load_taxes()


def _items():
    rows = []
    for s in range(4):
        for fam, sev in [('P0_identity', 0), ('P0_paraphrase', 0), ('P1_near_ancestor', 1), ('P2_distant_ancestor', 2), ('P4_action_reversal', 5)]:
            rows.append(dict(source_id=f's{s}', dataset='D', sample_id=f's{s}:{fam}', perturbation_type=fam, severity_level=sev))
    return pd.DataFrame(rows)


def test_perfect_metric_and_tie_semantics():
    it = _items()
    cons = build_constraints(it)
    val = {'P0_identity': 1.0, 'P0_paraphrase': 1.0, 'P1_near_ancestor': 0.8, 'P2_distant_ancestor': 0.6, 'P4_action_reversal': 0.0}
    s = pd.Series({r.sample_id: val[r.perturbation_type] for r in it.itertuples()})
    r = pairwise_ordering_accuracy(s, cons[cons.suite == 'core'], n_boot=50, items=it)
    assert r['accuracy_strict'] == 1.0 and r['monotonicity_violations'] == 0
    # metric that gives everything the same score: gt-constraints tie
    z = pd.Series({r.sample_id: 0.5 for r in it.itertuples()})
    r0 = pairwise_ordering_accuracy(z, cons[cons.suite == 'core'], n_boot=50)
    n_ge = (cons[cons.suite == 'core'].kind == 'ge').sum()
    n = r0['n_constraints']
    assert r0['accuracy_strict'] == pytest.approx(n_ge / n)                       # only 'ge' constraints satisfied
    assert r0['accuracy_tiehalf'] == pytest.approx((n_ge + 0.5 * (n - n_ge)) / n)  # ties get half credit


def test_purity_flags_impure_and_pure():
    ref = [Event(object='building', action='constructed', target='building', location='left', quantity='several', change_state='change')]
    good = [Event(object='road', action='constructed', target='road', location='left', quantity='several', change_state='change')]
    bad = [Event(object='road', action='demolished', source='road', location='left', quantity='several', change_state='change')]
    assert check_purity('P3_sibling_substitution', ref, good, {'expected_object': 'road'}, TAXES)['pure']
    assert not check_purity('P3_sibling_substitution', ref, bad, {'expected_object': 'road'}, TAXES)['pure']
