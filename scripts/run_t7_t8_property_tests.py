"""T7 (CE-P vs CE-R property test: omission should hurt recall, hallucination
should hurt precision) and T8 (contradiction-gate stress test on hand-written
synthetic pairs).

T7 reuses the already-computed results/E04/LEVIR_CC/summary.csv (P7_event_
omission and P8_hallucination rows) -- no new scoring needed, just an
extraction/framing pass, since the intended property test *is* exactly what
those two perturbation types measure.

T8 is a small, hand-written, deterministic set of caption pairs targeting
the ACTION_REVERSALS contradiction gate in ricce/metrics/event.py
(constructed<->demolished, appeared<->disappeared, expanded<->contracted)
plus a change<->no-change top-level gate test. These are test fixtures
written by the author, not experimental data or human ratings -- each
pair's expected gate behavior is stated by construction, and the script
verifies the real scorer's actual output against that expectation.
"""
import sys, json
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.utils import dump_json
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
from ricce.metrics.event import ricce_scores

OUT = ROOT / 'results' / 'ablations'
OUT.mkdir(parents=True, exist_ok=True)

# --- T7: extract from existing E04 summary ---
summary = pd.read_csv(ROOT / 'results' / 'E04' / 'LEVIR_CC' / 'summary.csv')
row_omit = summary[summary['type'] == 'P7_event_omission'].iloc[0]
row_hall = summary[summary['type'] == 'P8_hallucination'].iloc[0]
t7 = {
    'P7_event_omission': {'ce_precision': float(row_omit['ce_precision']), 'ce_recall': float(row_omit['ce_recall']),
                           'precision_minus_recall': float(row_omit['ce_precision'] - row_omit['ce_recall'])},
    'P8_hallucination': {'ce_precision': float(row_hall['ce_precision']), 'ce_recall': float(row_hall['ce_recall']),
                          'precision_minus_recall': float(row_hall['ce_precision'] - row_hall['ce_recall'])},
    'interpretation': 'Omission should selectively depress CE-R (recall) more than CE-P; hallucination should selectively depress CE-P more than CE-R. Source: results/E04/LEVIR_CC/summary.csv (real scored data, no new run needed).',
}
t7['property_holds'] = {
    'omission_hurts_recall_more': bool(row_omit['ce_precision'] - row_omit['ce_recall'] > 0.05),
    'hallucination_hurts_precision_more': bool(row_hall['ce_recall'] - row_hall['ce_precision'] > 0.05),
}
print('T7', t7)

# --- T8: hand-written contradiction-gate stress pairs ---
cfg = yaml.safe_load(open(ROOT / 'configs' / 'project.yaml'))
taxes = {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}
weights = cfg['metric']['weights']
p = cfg['parser']
parser = HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'],
                       use_embeddings=p.get('use_embeddings', True), embedding_model=p.get('embedding_model'),
                       similarity_threshold=float(p.get('similarity_threshold', 0.58)))

# (gt caption, prediction caption, expected: 'gated' if a hard contradiction
# should force ce_f1 to (near) 0 despite lexical overlap, else 'not_gated')
stress_pairs = [
    ("a building was constructed in the center of the image", "a building was demolished in the center of the image", 'gated'),
    ("several buildings were constructed near the road", "several buildings were demolished near the road", 'gated'),
    ("a new building appeared next to the river", "a building disappeared next to the river", 'gated'),
    ("some buildings appeared in the upper part of the image", "some buildings disappeared in the upper part of the image", 'gated'),
    ("a parking lot expanded along the highway", "a parking lot contracted along the highway", 'gated'),
    ("the residential area expanded to the north", "the residential area contracted to the north", 'gated'),
    ("a building was constructed on the bare land", "a building was constructed on the bare land", 'not_gated'),
    ("no changes occurred in this scene", "several buildings were constructed in this scene", 'gated'),
    ("several buildings were constructed in this scene", "no changes occurred in this scene", 'gated'),
    ("no changes occurred in this scene", "no changes occurred in this scene", 'not_gated'),
    ("a road was built through the field", "a road was demolished through the field", 'gated'),
    ("a large parking lot appeared beside the building", "a large parking lot appeared beside the building", 'not_gated'),
    ("the forest area expanded near the lake", "the forest area expanded near the lake", 'not_gated'),
    ("a house disappeared from the residential block", "a house appeared in the residential block", 'gated'),
    ("some buildings were demolished in the industrial zone", "some buildings were constructed in the industrial zone", 'gated'),
    ("the vegetation contracted along the riverbank", "the vegetation expanded along the riverbank", 'gated'),
    ("a small building was constructed near the parking area", "a small building was constructed near the parking area", 'not_gated'),
    ("the playground expanded to the east", "the playground expanded to the east", 'not_gated'),
    ("a warehouse was constructed in the open field", "a warehouse was demolished in the open field", 'gated'),
    ("a pond appeared in the green area", "a pond disappeared in the green area", 'gated'),
]

t8_rows = []
for gt_text, pred_text, expected in stress_pairs:
    ge = parser.parse(gt_text)
    pe = parser.parse(pred_text)
    sc = ricce_scores(pe, ge, taxes, weights)
    observed = 'gated' if sc['ce_f1'] < 0.15 else 'not_gated'
    t8_rows.append({'gt': gt_text, 'pred': pred_text, 'expected': expected, 'observed': observed,
                     'ce_f1': sc['ce_f1'], 'match': observed == expected})

t8_df = pd.DataFrame(t8_rows)
t8_df.to_csv(OUT / 'T8_contradiction_gate_stress.csv', index=False)
accuracy = float(t8_df['match'].mean())
print('T8 stress-test accuracy:', accuracy, '/', len(t8_df))
print(t8_df[['expected', 'observed', 'ce_f1', 'match']])

out = {'T7_property_test_omission_vs_hallucination': t7,
       'T8_contradiction_gate_stress_test': {'n_pairs': len(t8_df), 'accuracy_matches_expected': accuracy,
                                              'n_failures': int((~t8_df['match']).sum()),
                                              'failures': t8_df[~t8_df['match']][['gt', 'pred', 'expected', 'observed', 'ce_f1']].to_dict(orient='records')}}
dump_json(out, OUT / 'T7_T8_metrics.json')
print('wrote', OUT / 'T7_T8_metrics.json')
