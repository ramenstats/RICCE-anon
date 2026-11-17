"""Collapsing per-event hP/hR into one harmonic scalar before aggregation loses
the coarse-vs-over-specific distinction that separate CE-P/CE-R preserve."""
import pytest
from ricce.benchmarks.env import load_taxes, load_weights
from ricce.metrics.event import ricce_scores, match_events
from test_metric_properties import ev, sc, TAXES, W


def collapsed_F(pred, gt):
    pairs, _ = match_events(pred, gt, TAXES, W)
    m = sum(p[4] for p in pairs)
    P, R = m / max(1, len(pred)), m / max(1, len(gt))
    return 2 * P * R / (P + R) if P + R else 0.0


def test_mirrored_pair_indistinguishable_when_collapsed_but_separated_ce_distinguishes():
    coarse = sc([ev('building')], [ev('residential_building')])          # correct but coarse
    over = sc([ev('residential_building')], [ev('building')])            # unsupported specificity
    assert coarse['ce_precision'] == pytest.approx(1.0) and coarse['ce_recall'] < 1.0
    assert over['ce_recall'] == pytest.approx(1.0) and over['ce_precision'] < 1.0
    fc = collapsed_F([ev('building')], [ev('residential_building')])
    fo = collapsed_F([ev('residential_building')], [ev('building')])
    assert fc == pytest.approx(fo, abs=1e-12)


def test_same_reference_collapsed_ranking_prefers_unsupported_overspecific():
    ref = [ev('building')]
    coarse, over = [ev('built_environment')], [ev('residential_building')]
    assert collapsed_F(over, ref) > collapsed_F(coarse, ref)              # collapsed: unsupported detail wins
    sc_c, sc_o = sc(coarse, ref), sc(over, ref)
    assert sc_c['ce_precision'] == pytest.approx(1.0) and sc_c['ce_recall'] < 1.0
    assert sc_o['ce_precision'] < 1.0 and sc_o['ce_recall'] == pytest.approx(1.0)
