"""P1-7 (v2 README): qualitative error taxonomy from real
frozen predictions. For each of several systems/datasets, selects
representative real examples across error categories using the already-
computed per-sample CE-P/CE-R/hallucination/omission scores (no new
scoring), and re-parses each to show the actual predicted/reference
events and the reason for the penalty -- everything traceable to a real
prediction, real reference, and real parsed event, nothing invented.
"""
import json, sys, csv
from pathlib import Path
import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl, dump_json
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
from ricce.metrics.event import ricce_scores

OUT = ROOT / 'results_r1' / 'error_analysis'
OUT.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(open(ROOT / 'configs' / 'project.yaml'))
taxes = {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}
weights = cfg['metric']['weights']
p = cfg['parser']
parser = HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'],
                       use_embeddings=p.get('use_embeddings', True), embedding_model=p.get('embedding_model'),
                       similarity_threshold=float(p.get('similarity_threshold', 0.58)))

CATEGORIES = {
    # hallucination_rate/omission_rate as defined in ricce_scores are
    # structurally 0 for almost every sample here since these datasets are
    # ~99% single-event captions (see the multi-event finding elsewhere in
    # here: only ~4-8% of captions are genuinely multi-event, and
    # even fewer of those appear on both sides of a single prediction/
    # reference pair) -- the field is only non-zero when the number of
    # *events* differs, not when individual event content is wrong. We use
    # the CE-P/CE-R imbalance instead as the practical proxy for
    # hallucination-like vs. omission-like content errors within a single
    # matched event, which is where almost all of the real error signal
    # actually lives on this data.
    'hallucination_like_low_precision': lambda r: float(r['ce_precision']) < 0.5 and float(r['ce_recall']) - float(r['ce_precision']) > 0.15,
    'omission_like_low_recall': lambda r: float(r['ce_recall']) < 0.5 and float(r['ce_precision']) - float(r['ce_recall']) > 0.15,
    'false_no_change': lambda r: 'no difference' in r['prediction'].lower() or 'nothing has changed' in r['prediction'].lower() or 'no change' in r['prediction'].lower(),
    'success_high_precision_recall': lambda r: float(r['ce_precision']) >= 0.95 and float(r['ce_recall']) >= 0.95,
    'partial_match_low_both': lambda r: 0.2 < float(r['ce_f1']) < 0.6,
    'total_miss': lambda r: float(r['ce_f1']) < 0.05,
}


def examples_for(system, dataset, n_per_cat=4):
    csv_path = ROOT / 'results' / 'E08' / dataset / system / 'per_sample_metrics.csv'
    if not csv_path.exists():
        return []
    manifest = {r['image_id']: r for r in read_jsonl(ROOT / 'artifacts' / 'manifests' / f'{dataset}.jsonl')}
    rows = list(csv.DictReader(open(csv_path)))
    out = []
    for cat, pred_fn in CATEGORIES.items():
        matches = [r for r in rows if r['image_id'] in manifest and pred_fn(r)]
        # deterministic selection: sort by image_id for reproducibility, take first n
        matches = sorted(matches, key=lambda r: r['image_id'])[:n_per_cat]
        for r in matches:
            m = manifest[r['image_id']]
            refs = m['references']
            pe = parser.parse(r['prediction'])
            ge0 = parser.parse(refs[0])
            ce = ricce_scores(pe, ge0, taxes, weights)
            out.append({
                'category': cat, 'system': system, 'dataset': dataset, 'image_id': r['image_id'],
                't1_path': m.get('t1_path'), 't2_path': m.get('t2_path'),
                'references': refs, 'prediction': r['prediction'],
                'parsed_prediction_events': [e.to_dict() for e in pe],
                'parsed_reference_events_ref0': [e.to_dict() for e in ge0],
                'ce_precision': float(r['ce_precision']), 'ce_recall': float(r['ce_recall']),
                'ce_f1': float(r['ce_f1']), 'hallucination_rate': float(r['hallucination_rate']),
                'omission_rate': float(r['omission_rate']),
            })
    return out


all_examples = []
for dataset in ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC']:
    for system in ['qwen', 'sft_seed42', 'chg2cap', 'swmcc']:
        ex = examples_for(system, dataset)
        print(f'{dataset}/{system}: {len(ex)} examples across {len(set(e["category"] for e in ex))} categories')
        all_examples.extend(ex)

dump_json(all_examples, OUT / 'error_examples.json')
by_cat = {}
for e in all_examples:
    by_cat.setdefault(e['category'], 0)
    by_cat[e['category']] += 1
print('\nTotal examples:', len(all_examples))
print('By category:', json.dumps(by_cat, indent=2))
dump_json({'total': len(all_examples), 'by_category': by_cat}, OUT / 'summary.json')
print('\nwrote', OUT / 'error_examples.json')
