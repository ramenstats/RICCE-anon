"""Cross-family 'semantic validity' ordering: a correct statement (paraphrase or
correct-but-coarse ancestor) must outrank a contradictory one (action reversal,
change-state reversal, negation flip, source/target swap) for the SAME source
caption. Uses only pure RICCE-Bench items and the per-item score files in
results_cvpr_final/ordering/scores/. Strict accuracy (ties count as failures),
tie-half accuracy, cluster bootstrap over source captions (2000 resamples)."""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
items = {}
with open(ROOT / 'artifacts_cvpr_final/riccebench/items.jsonl') as f:
    for l in f:
        d = json.loads(l); items[d['sample_id']] = (d['source_id'], d['perturbation_type'], d['dataset'])
sc = pd.concat([pd.read_csv(p) for p in glob.glob(str(ROOT / 'results_cvpr_final/ordering/scores/*.csv'))])
sc = sc[sc.sample_id.isin(items)]
sc['source_id'] = sc.sample_id.map(lambda s: items[s][0]); sc['ptype'] = sc.sample_id.map(lambda s: items[s][1])
GOOD = ['P0_paraphrase', 'P1_near_ancestor']
BAD = ['P4_action_reversal', 'P9_change_state_reversal', 'negation_flip', 'source_target_swap']
rng = np.random.default_rng(0); out = []
for metric, g in sc.groupby('metric'):
    piv = g.pivot_table(index='source_id', columns='ptype', values='score')
    rows = []
    for gd in GOOD:
        for bd in BAD:
            if gd in piv and bd in piv:
                x = piv[[gd, bd]].dropna()
                if len(x): rows.append(pd.DataFrame({'source_id': x.index.values, 'good': gd, 'bad': bd, 'win': (x[gd] > x[bd] + 1e-9).astype(float),
                                                     'tie': ((x[gd] - x[bd]).abs() <= 1e-9).astype(float)}))
    if not rows: continue
    r = pd.concat(rows, ignore_index=True); r['half'] = r.win + 0.5 * r.tie
    def stat(d): return d.win.mean(), d.half.mean()
    per_src = r.groupby('source_id')[['win', 'half']].agg(['sum', 'count'])
    ids = per_src.index.values; idx = rng.integers(0, len(ids), (2000, len(ids)))
    ws, wc = per_src[('win', 'sum')].values, per_src[('win', 'count')].values
    boots = (ws[idx].sum(1) / wc[idx].sum(1))
    row = dict(metric=metric, n_constraints=len(r), strict=r.win.mean(), strict_lo=np.percentile(boots, 2.5), strict_hi=np.percentile(boots, 97.5), tiehalf=r.half.mean(), tie_rate=r.tie.mean())
    for bd in BAD:
        s = r[r.bad == bd]; row[f'strict_vs_{bd}'] = s.win.mean() if len(s) else np.nan
    out.append(row)
res = pd.DataFrame(out).sort_values('strict', ascending=False)
o = ROOT / 'results_cvpr_final/ordering'; res.to_csv(o / 'semantic_validity_ordering.csv', index=False)
(o / 'semantic_validity_README.md').write_text(
 'Semantic-validity suite: for each source caption, {paraphrase, near ancestor} must score STRICTLY higher than each of {action reversal, change-state reversal, negation flip, source/target swap}.\n'
 'Pure RICCE-Bench items only; ties count as failures (strict); CI = cluster bootstrap over source captions (2000). Circularity caveat: perturbations are generated with the RICCE ontology/parser.\n'
 'SPICE/LLM-judge cover the 300-source-per-dataset subset only.\n')
pd.set_option('display.width', 250); print(res.round(3).to_string(index=False))
