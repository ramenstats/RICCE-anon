"""v2-final: human-corrected parser-error analysis (real-prediction
block only), recomputed under the FINAL v2 evaluator. Mirrors
scripts/run_parser_error_propagation.py's real_prediction pairing exactly
(same 480 human-labelled captions, same 7 systems, same partial-mapping
convention -- an unmappable human field is left empty, mirroring how the
auto side treats an unresolved field), scoring both the ALL_mappable subset
(41 captions/287 pairs) and the broader SENSITIVITY_partial_mapping subset
(96 captions/672 pairs) with ricce_scores_v2 instead of ricce_scores."""
import sys, itertools
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
from ricce.evaluation.mapper_v2 import load_cfg, load_taxes, build_parser, HumanMapper, load_labels, label_records, DATASETS
from ricce.utils import read_jsonl
from ricce.metrics.event_v2 import ricce_scores_v2

OUT = ROOT / 'results_v2/parser_error_v2'; OUT.mkdir(parents=True, exist_ok=True)
cfg = load_cfg(); taxes = load_taxes(cfg); W = cfg['metric']['weights']; parser = build_parser(taxes, cfg); hm = HumanMapper(taxes)
lab = load_labels()
SYSTEMS = ['chg2cap', 'swmcc', 'qwen', 'sft_seed42', 'dpo_seed42', 'tcpo_seed42']
preds = {(s, ds): {r['image_id']: r['prediction'] for r in read_jsonl(ROOT / f'results/baselines/{s}/{ds}_test.jsonl')} for s in SYSTEMS for ds in DATASETS}
FINAL_V2 = dict(gate_mode='hard', exclude_root=True, dual_matching=True)

cap = {}
for row in label_records(lab):
    m = hm.map_row(row)
    cap[row['id']] = dict(row=row, mapped=m, ok=hm.fully_mappable(m), human=hm.to_event(m, row['caption']), auto=parser.parse(row['caption'])[0])

def cef(pe, ref):
    r = ricce_scores_v2([pe], [ref], taxes, W, **FINAL_V2)
    return np.nan if r['not_scorable'] else r['ce_f1']

rows = []
for cid, c in cap.items():
    ds, iid = c['row']['dataset'], c['row']['image_id']
    for s in SYSTEMS:
        if iid in preds[(s, ds)]:
            p = preds[(s, ds)][iid]
            pe = parser.parse(p)[0]
            auto, hum = c['auto'], c['human']
            f_auto, f_h = cef(pe, auto), cef(pe, hum)
            rows.append(dict(id=cid, dataset=ds, system=s, mappable=c['ok'], cef_auto_ref=f_auto, cef_human_ref=f_h, delta=f_auto - f_h))
D = pd.DataFrame(rows).dropna(subset=['cef_auto_ref', 'cef_human_ref'])
D.to_csv(OUT / 'per_pair.csv', index=False)

def corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return {'spearman': None, 'kendall': None}
    return {'spearman': float(spearmanr(a, b)[0]), 'kendall': float(kendalltau(a, b)[0])}

rng = np.random.default_rng(0)
def boot_mean(x, B=2000):
    x = np.asarray(x); n = len(x)
    if n == 0:
        return None
    m = [x[rng.integers(0, n, n)].mean() for _ in range(B)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

def block(g):
    return dict(n_pairs=len(g), n_captions=int(g['id'].nunique()), mean_cef_auto=float(g.cef_auto_ref.mean()),
                mean_cef_human=float(g.cef_human_ref.mean()), delta_cef_auto_minus_human=float(g.delta.mean()),
                delta_CI95=boot_mean(g.delta), item_level_corr=corr(g.cef_auto_ref, g.cef_human_ref))

summary = {'ALL_mappable': block(D[D.mappable]), 'SENSITIVITY_partial_mapping_all_paired': block(D)}
import json
json.dump(summary, open(OUT / 'summary.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
print('DONE')
