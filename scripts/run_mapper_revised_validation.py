"""Mapper-coverage re-validation (README sec. 5):
re-runs the EXACT SAME 480-caption human-label validation harness used for
the frozen mapper (results_cvpr_final/mapper_v2), swapping in only
HybridParserRevised + configs/ontology_revised (see ricce/parsing/
hybrid_revised.py for exactly what changed and why) with n-gram containment
turned on. Writes results_metric_v2/mapper_revised/ alongside (not over) the
frozen results_cvpr_final/mapper_v2/, so the before/after comparison is a
diff between two artifact directories, not a claim without one."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import pandas as pd
from ricce.evaluation.mapper_v2 import (AXES, DATASETS, AXIS_TAX, HumanMapper, load_labels, label_records,
                                             classify_axis, relation, wilson, parser_field)
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid_revised import HybridParserRevised

OUT = ROOT / 'results_metric_v2' / 'mapper_revised'; OUT.mkdir(parents=True, exist_ok=True)

taxes = {k: Taxonomy.from_yaml(ROOT / 'configs' / 'ontology_revised' / f'{k}.yaml') for k in ['object', 'action', 'location', 'quantity']}
parser = HybridParserRevised(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'],
                              use_embeddings=True, embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                              similarity_threshold=0.58, use_ngram=True)
hm = HumanMapper(taxes)
lab = load_labels()
print(f'{len(lab)} labelled captions loaded', flush=True)

cells = []
for row in label_records(lab):
    m = hm.map_row(row)
    ev = parser.parse(row['caption'])[0]
    nochg = row['change_state'] == 'no_change'
    for ax in AXES:
        pn = parser_field(ev, ax)
        cat = classify_axis(ax, row[ax], m[ax]['node'], m[ax]['status'], pn, nochg)
        tax = taxes[AXIS_TAX[ax]] if ax in AXIS_TAX else None
        rel = None
        if cat in ('b_consistent', 'c_different') and tax is not None:
            rel = relation(tax, pn, m[ax]['node'])
        cells.append(dict(dataset=row['dataset'], axis=ax, cat=cat, present=row[ax] is not None,
                           mappable=m[ax]['status'] == 'mapped', rel=rel))

C = pd.DataFrame(cells)


def agg(g):
    n = len(g); pres = g[g.present]; np_ = len(pres)
    res = pres[pres.cat != 'd_unresolved']; mp = pres[pres.mappable]
    d = {'n_captions': n, 'n_present': np_}
    d['P_resolved_given_present'] = len(res) / np_ if np_ else None
    lo, hi = wilson(len(res), np_); d['P_resolved_CI95'] = [lo, hi]
    d['exact_acc_strict_over_mappable_present'] = float((mp.cat == 'b_consistent').mean()) if len(mp) else None
    return d


rows = []
for ax in AXES:
    for ds in DATASETS + ['ALL']:
        g = C[(C.axis == ax) & ((C.dataset == ds) | (ds == 'ALL'))]
        rows.append({'axis': ax, 'dataset': ds, **agg(g)})
pa = pd.DataFrame(rows)
pa.to_csv(OUT / 'per_axis.csv', index=False)

# side-by-side with the frozen numbers, if present
frozen_path = ROOT / 'results_cvpr_final' / 'mapper_v2' / 'per_axis.csv'
compare = None
if frozen_path.exists():
    fz = pd.read_csv(frozen_path)
    fz = fz[fz.dataset == 'ALL'][['axis', 'n_present', 'P_resolved_given_present']].rename(
        columns={'n_present': 'n_present_frozen', 'P_resolved_given_present': 'P_resolved_frozen'})
    rv = pa[pa.dataset == 'ALL'][['axis', 'n_present', 'P_resolved_given_present']].rename(
        columns={'n_present': 'n_present_revised', 'P_resolved_given_present': 'P_resolved_revised'})
    compare = fz.merge(rv, on='axis')
    compare['delta'] = compare['P_resolved_revised'] - compare['P_resolved_frozen']
    compare.to_csv(OUT / 'before_after_comparison.csv', index=False)
    print(compare.to_string(index=False), flush=True)

json.dump({'parser_stage_counts': parser.stage_counts,
           'parser_config': {'use_ngram': True, 'use_embeddings': True, 'similarity_threshold': 0.58,
                              'class': 'HybridParserRevised', 'ontology': 'configs/ontology_revised'},
           'note': 'Same 480-caption human-label harness as results_cvpr_final/mapper_v2 (single annotator, '
                   'proxy phrase->node mapping); only the parser+ontology changed.'},
          open(OUT / 'config.json', 'w'), indent=1)
print('DONE', flush=True)
