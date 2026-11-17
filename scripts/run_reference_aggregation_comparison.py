"""P1-2 (v2 README): resolve the ambiguity between the paper's
described multi-reference consensus graph and the mean-over-references
scoring actually used in every headline table (ricce/experiments.py
e08(): `ces=[ricce_scores(pe, parser.parse(g), ...) for g in
r['references']]` then averaged -- this IS mean-over-references, the
consensus graph in ricce/metrics/consensus.py is implemented but never
called from the production scoring path).

Compares five aggregation protocols on the same frozen predictions:
  A. single fixed reference (references[0])
  B. mean-over-references (the actual current default)
  C. max-over-references (most charitable single reference)
  D. consensus graph, unweighted (cluster references into one merged event
     set via consensus_events(), score once against it, support ignored)
  E. consensus graph, support-weighted (recall contribution of each
     consensus event weighted by its support = fraction of references
     that mentioned it, i.e. genuinely using q(e) in scoring rather than
     only in the consensus construction)

Real frozen predictions, real references, nothing simulated.
"""
import json, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl, dump_json
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
from ricce.metrics.event import ricce_scores, event_similarity_pr, match_events
from ricce.metrics.consensus import consensus_events
from ricce.schema import Event

OUT = ROOT / 'results_r1' / 'reference_aggregation'
OUT.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(open(ROOT / 'configs' / 'project.yaml'))
taxes = {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}
weights = cfg['metric']['weights']
p = cfg['parser']
parser = HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'],
                       use_embeddings=p.get('use_embeddings', True), embedding_model=p.get('embedding_model'),
                       similarity_threshold=float(p.get('similarity_threshold', 0.58)))

manifest = {r['image_id']: r for r in read_jsonl(ROOT / 'artifacts' / 'manifests' / 'LEVIR_CC.jsonl')}


def support_weighted_score(pred_events, consensus_dicts, taxes, weights):
    """Same mechanics as ricce_scores, but recall mass for consensus
    event j is scaled by its support (in [1/n_refs, 1]) instead of 1, so
    an event only one of five references mentioned counts for less than
    one every reference agrees on. Precision (over predicted events) is
    left unweighted -- support is a property of *reference* agreement,
    it says nothing about whether a given predicted event is right."""
    gt_events = [Event(**{k: v for k, v in d.items() if k != 'support'}) for d in consensus_dicts]
    supports = [d.get('support', 1.0) for d in consensus_dicts]
    if not pred_events or not gt_events:
        return {'ce_precision': 0.0, 'ce_recall': 0.0, 'ce_f1': 0.0}
    pairs, _ = match_events(pred_events, gt_events, taxes, weights)
    p_mass = sum(x[2] for x in pairs)
    r_mass = sum(x[3] * supports[x[1]] for x in pairs)
    r_denom = sum(supports) if supports else 1.0
    prec = p_mass / max(1, len(pred_events))
    rec = r_mass / max(1e-9, r_denom)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {'ce_precision': prec, 'ce_recall': rec, 'ce_f1': f1}


def run_for_system(name, pred_path, max_n=None):
    preds = [r for r in read_jsonl(pred_path) if r['image_id'] in manifest]
    if max_n:
        preds = preds[:max_n]
    rows = {'A_single_ref': [], 'B_mean_over_refs': [], 'C_max_over_refs': [],
            'D_consensus_unweighted': [], 'E_consensus_support_weighted': []}
    timings = defaultdict(float)
    spread_vs_agreement = []  # (std of per-ref ce_f1, |B - D| gap) to test robustness to disagreement

    for r in preds:
        refs = manifest[r['image_id']]['references']
        if len(refs) < 2:
            continue
        pe = parser.parse(r['prediction'])
        ref_events = [parser.parse(g) for g in refs]

        t0 = time.time()
        a = ricce_scores(pe, ref_events[0], taxes, weights)
        timings['A_single_ref'] += time.time() - t0

        t0 = time.time()
        per_ref = [ricce_scores(pe, ge, taxes, weights) for ge in ref_events]
        b = {k: float(np.mean([x[k] for x in per_ref])) for k in ['ce_precision', 'ce_recall', 'ce_f1']}
        timings['B_mean_over_refs'] += time.time() - t0

        t0 = time.time()
        c = {k: float(np.max([x[k] for x in per_ref])) for k in ['ce_precision', 'ce_recall', 'ce_f1']}
        timings['C_max_over_refs'] += time.time() - t0

        t0 = time.time()
        cons = consensus_events(ref_events, taxes, weights)
        d = ricce_scores(pe, [Event(**{k: v for k, v in x.items() if k != 'support'}) for x in cons], taxes, weights)
        timings['D_consensus_unweighted'] += time.time() - t0

        t0 = time.time()
        e = support_weighted_score(pe, cons, taxes, weights)
        timings['E_consensus_support_weighted'] += time.time() - t0

        for key, val in [('A_single_ref', a), ('B_mean_over_refs', b), ('C_max_over_refs', c),
                          ('D_consensus_unweighted', d), ('E_consensus_support_weighted', e)]:
            rows[key].append(val)

        ref_f1_std = float(np.std([x['ce_f1'] for x in per_ref]))
        gap = abs(b['ce_f1'] - d['ce_f1'])
        spread_vs_agreement.append((ref_f1_std, gap))

    summary = {}
    for key, vals in rows.items():
        summary[key] = {
            'n': len(vals),
            'ce_precision': float(np.mean([v['ce_precision'] for v in vals])),
            'ce_recall': float(np.mean([v['ce_recall'] for v in vals])),
            'ce_f1': float(np.mean([v['ce_f1'] for v in vals])),
            'total_runtime_s': timings[key],
            'runtime_per_item_ms': 1000 * timings[key] / max(1, len(vals)),
        }

    stds = np.array([x[0] for x in spread_vs_agreement])
    gaps = np.array([x[1] for x in spread_vs_agreement])
    corr = float(np.corrcoef(stds, gaps)[0, 1]) if len(stds) > 2 and stds.std() > 0 else None

    return {
        'system': name, 'n_items': len(preds),
        'protocols': summary,
        'robustness_to_contradictory_references': {
            'metric': 'Pearson correlation between per-item reference disagreement (std of single-ref CE-F1 across the 5 references) and |mean-over-refs - consensus| CE-F1 gap',
            'pearson_r': corr,
            'interpretation': 'A positive correlation means consensus and mean-over-reference diverge more exactly when the references themselves disagree more -- i.e. the aggregation choice matters most precisely where it is hardest to get right, which is the scenario the method is meant to handle.',
        },
    }


results = {}
import os
MAX_N = int(os.environ.get('REF_AGG_MAX_N', '0')) or None
for name, path in [
    ('sft_seed42', ROOT / 'results' / 'baselines' / 'sft_seed42' / 'LEVIR_CC_test.jsonl'),
    ('chg2cap', ROOT / 'results' / 'baselines' / 'chg2cap' / 'LEVIR_CC_test.jsonl'),
]:
    if not path.exists():
        print(f'{name}: predictions not found at {path}, skipping')
        continue
    print(f'Scoring {name}...', flush=True)
    results[name] = run_for_system(name, path, max_n=MAX_N)
    print(json.dumps(results[name]['protocols'], indent=2))

dump_json(results, OUT / 'aggregation_comparison.json')
print('\nwrote', OUT / 'aggregation_comparison.json')
