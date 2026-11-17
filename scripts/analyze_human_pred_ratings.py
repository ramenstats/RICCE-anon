"""Human validation of model predictions (required check #1):
correlates real human ratings of 300 real model outputs (100 items each
from LEVIR-CC/SECOND-CC/DUBAI-CC, sampled across 5 systems) against
per-item automated metrics (CE-F1, BLEU-4, METEOR, ROUGE-L, BERTScore).

Input: results/human_ratings/to_fill_model_predictions.csv, manually
scored by the paper's author (single annotator, text-only pass -- no
images, so object/location/completeness scores reflect caption-internal
plausibility rather than ground-truth image verification; documented
explicitly as a limitation).

The CSV's 'overall_semantic_correctness' column was found to be the SUM
of the other 6 columns (a spreadsheet artifact), not an independent 1-5
judgment -- so 'overall' here is recomputed as the mean of the 6
sub-scores instead of trusting that column directly.
"""
import csv, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
from ricce.metrics.event import ricce_scores
from ricce.metrics.conventional import sentence_bleu_scores, rouge_l, meteor, sentence_bert

CSV = ROOT / 'results' / 'human_ratings' / 'to_fill_model_predictions.csv'
OUT = ROOT / 'results' / 'E07_human_pred_validation'
OUT.mkdir(parents=True, exist_ok=True)
SUBSCORES = ['object_correctness','action_correctness','location_correctness',
             'completeness','specificity','hallucination','overall_semantic_correctness']

cfg = yaml.safe_load(open(ROOT / 'configs' / 'project.yaml'))
taxes = {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}
p = cfg['parser']
parser = HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'],
                       use_embeddings=p.get('use_embeddings', True), embedding_model=p.get('embedding_model'),
                       similarity_threshold=float(p.get('similarity_threshold', 0.58)))
weights = cfg['metric']['weights']

manifests = {ds: {r['image_id']: r for r in read_jsonl(ROOT / 'artifacts' / 'manifests' / f'{ds}.jsonl')}
             for ds in ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC']}
# Excel/Sheets strips leading zeros from numeric-looking image_ids on save
# (SECOND_CC ids like '0170130' become '170130') -- build a lookup that
# recovers the original zero-padded id so real spreadsheet-edited CSVs
# still match, instead of silently dropping ~1/4 of the sample.
stripped_lookup = {ds: {mid.lstrip('0') or '0': mid for mid in m} for ds, m in manifests.items()}

rows = list(csv.DictReader(open(CSV)))
n_bad = 0
records = []
for r in rows:
    ds, iid, pred = r['dataset'], r['image_id'], r['prediction']
    manifest = manifests.get(ds, {})
    m = manifest.get(iid)
    if m is None:
        recovered = stripped_lookup.get(ds, {}).get(iid)
        if recovered is not None:
            iid = recovered
            m = manifest.get(iid)
    if m is None:
        n_bad += 1
        continue
    refs = m['references']
    try:
        pe = parser.parse(pred)
        ge_sets = [parser.parse(g) for g in refs]
        ces = [ricce_scores(pe, ge, taxes, weights) for ge in ge_sets]
        ce_f1 = float(np.mean([c['ce_f1'] for c in ces]))
    except Exception:
        ce_f1 = None
    bleu = sentence_bleu_scores(pred, refs)
    bleu4 = bleu.get('bleu4') if isinstance(bleu, dict) else None
    rl = rouge_l(pred, refs)
    met = meteor(pred, refs)
    try:
        sbert = sentence_bert(pred, refs)
    except Exception:
        sbert = None

    # object/action/.../hallucination are 0-5 ints; recompute 'overall' as
    # mean of the 6 real sub-scores (the CSV's own 'overall' column is
    # actually their sum -- a spreadsheet artifact, see module docstring).
    six = [float(r[c]) for c in SUBSCORES[:6]]
    human_overall = float(np.mean(six))

    rec = {'dataset': ds, 'system': r['system'], 'image_id': iid,
           'human_overall': human_overall, 'ce_f1': ce_f1, 'bleu4': bleu4,
           'rouge_l': rl, 'meteor': met, 'bertscore': sbert}
    for c in SUBSCORES[:6]:
        rec[c] = float(r[c])
    records.append(rec)

print(f'{len(records)} items scored ({n_bad} skipped -- image_id not found in manifest)')

def corr(xs, ys):
    xs = np.array(xs, dtype=float); ys = np.array(ys, dtype=float)
    mask = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 3:
        return None
    pear = stats.pearsonr(xs, ys)
    spear = stats.spearmanr(xs, ys)
    return {'n': int(len(xs)), 'pearson_r': float(pear[0]), 'pearson_p': float(pear[1]),
            'spearman_r': float(spear[0]), 'spearman_p': float(spear[1])}

metric_names = ['ce_f1', 'bleu4', 'rouge_l', 'meteor', 'bertscore']
results = {'n_items': len(records), 'n_skipped': n_bad, 'correlations_vs_human_overall': {}}
human_overall_all = [r['human_overall'] for r in records]
for mname in metric_names:
    vals = [r[mname] if r[mname] is not None else float('nan') for r in records]
    c = corr(human_overall_all, vals)
    results['correlations_vs_human_overall'][mname] = c
    print(f'human_overall vs {mname}: {c}')

# per-dataset breakdown
results['per_dataset'] = {}
for ds in ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC']:
    sub = [r for r in records if r['dataset'] == ds]
    ho = [r['human_overall'] for r in sub]
    d = {}
    for mname in metric_names:
        vals = [r[mname] if r[mname] is not None else float('nan') for r in sub]
        d[mname] = corr(ho, vals)
    results['per_dataset'][ds] = {'n': len(sub), 'correlations': d}

with open(OUT / 'correlations.json', 'w') as f:
    json.dump(results, f, indent=2)
with open(OUT / 'per_item.json', 'w') as f:
    json.dump(records, f, indent=2)
print('\nwrote', OUT / 'correlations.json', 'and per_item.json')
