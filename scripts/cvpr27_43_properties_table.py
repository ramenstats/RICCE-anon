"""Table of the properties desired of a remote-sensing change-captioning metric
(CVPR2027_Schematic_and_Properties/tab_properties.tex).

Rows 1-3 are strict ordering accuracies on the benchmark, read from the saved
per-item scores with the semantic-validity rule of
scripts/run_semantic_validity_ordering.py (BERTScore: canonical tokenization,
scripts/cvpr27_09_bertscore_canonical.py). Row 4 is a design property. Row 5 is
the Spearman correlation with the mean overall human rating on the 896 scorable items
(03_HUMAN_REANALYSIS/human_main_metrics.csv). Every value is asserted against
the numbers already printed in the paper before the table is written.
"""
import glob, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'CVPR2027_Schematic_and_Properties'
S = ROOT / 'submission_cvpr27_new_version'
BC = S / '07_BASELINES' / 'bertscore_canonical'

GROUPS = {  # column -> bench metric ids, human-study metric ids
    'N-gram': (['bleu4', 'meteor', 'rougeL', 'cider_d_item'], ['bleu4', 'meteor', 'rougeL', 'cider_d', 'csm4']),
    'Emb.':   (['bertscore_canonical', 'sbert'], ['bertscore_f', 'sbert']),
    'SPICE':  (['spice'], ['spice_f']),
    'LLM':    (['llm_judge'], ['llm_judge']),
    'Hier.':  (['kiritchenko_macro', 'lca_f', 'hcm'], ['b_kiritchenko_macro', 'b_lca_f', 'b_hcm']),
    'TaxAw.': (['taxaware_single', 'taxaware_axisflat'], ['b_taxaware_single', 'b_taxaware_axisflat']),
    'RICCE':  (['CE-F'], ['ce_f1']),
}
SUBSET = {'spice', 'llm_judge'}  # scored on the 300-source-per-dataset subset only

# ---- per-item benchmark scores
items = {}
for l in open(ROOT / 'artifacts_cvpr_final/riccebench/items.jsonl'):
    d = json.loads(l); items[d['sample_id']] = (d['source_id'], d['perturbation_type'])
sc = pd.concat([pd.read_csv(p) for p in glob.glob(str(ROOT / 'results_cvpr_final/ordering/scores/*.csv'))])
bc = pd.read_csv(BC / 'bertscore.csv'); bc['metric'] = 'bertscore_canonical'
sc = pd.concat([sc[sc.metric != 'bertscore'], bc])
sc = sc[sc.sample_id.isin(items)].copy()
sc['source_id'] = sc.sample_id.map(lambda s: items[s][0]); sc['ptype'] = sc.sample_id.map(lambda s: items[s][1])

GOOD = ['P0_paraphrase', 'P1_near_ancestor']
BAD = ['P4_action_reversal', 'P9_change_state_reversal', 'negation_flip', 'source_target_swap']

def wins(piv, good, bad):
    out = []
    for gd in good:
        for bd in bad:
            if gd in piv and bd in piv:
                x = piv[[gd, bd]].dropna(); out += list(x[gd] > x[bd] + 1e-9)
    return out

bench = {}
for m, g in sc.groupby('metric'):
    piv = g.pivot_table(index='source_id', columns='ptype', values='score')
    v, st, ns = wins(piv, GOOD, BAD), wins(piv, GOOD, ['source_target_swap']), wins(piv, ['P1_near_ancestor'], ['P3_sibling_substitution'])
    bench[m] = dict(valid=sum(v) / len(v), n_valid=len(v), st=sum(st) / len(st), n_st=len(st), ns=sum(ns) / len(ns), n_ns=len(ns))

# ---- checks against the published numbers (Tab. ordering, Sec. 4 text, semantic_validity_ordering.csv)
sv = pd.read_csv(ROOT / 'results_cvpr_final/ordering/semantic_validity_ordering.csv').set_index('metric')
for m in sv.index:
    if m in bench:
        assert abs(bench[m]['valid'] - sv.loc[m, 'strict']) < 1e-9, m
        assert abs(bench[m]['st'] - sv.loc[m, 'strict_vs_source_target_swap']) < 1e-9, m
assert abs(bench['bertscore_canonical']['valid'] - json.load(open(BC / 'summary.json'))['semantic_validity']['strict']) < 1e-9
fam = pd.read_csv(ROOT / 'results_cvpr_final/ordering/ordering_accuracy_by_family.csv')
fam = pd.concat([fam[fam.metric != 'bertscore'], pd.read_csv(BC / 'ordering_accuracy_by_family.csv').assign(metric='bertscore_canonical')])
nsf = fam[fam.group == 'P1_near_ancestor>P3_sibling_substitution'].set_index('metric').accuracy_strict
for m in bench:
    if m in nsf.index:
        assert abs(bench[m]['ns'] - nsf[m]) < 1e-9, m
printed = {'bleu4': (.578, .018), 'meteor': (.509, .339), 'rougeL': (.519, .018), 'cider_d_item': (.587, .762),
           'bertscore_canonical': (.660, .485), 'sbert': (.780, .920), 'spice': (.407, .322), 'llm_judge': (.991, .828),
           'kiritchenko_macro': (.989, .969), 'lca_f': (.986, .954), 'hcm': (.979, .998),
           'taxaware_single': (.357, .873), 'taxaware_axisflat': (.602, .847), 'CE-F': (1.0, 1.0)}
for m, (v, n) in printed.items():
    assert round(bench[m]['valid'], 3) == v and round(bench[m]['ns'], 3) == n, (m, bench[m])

# CE-P ignores an omitted event and CE-R an invented one: every such pair ties
for metric, grp, n in [('CE-P', 'P0_identity>P7_event_omission', 908), ('CE-R', 'P0_identity>P8_event_hallucination', 2990)]:
    r = fam[(fam.metric == metric) & (fam.group == grp)].iloc[0]
    assert r.n_constraints == n and r.n_ties == n and r.accuracy_strict == 0, (metric, grp)

# ---- human correlations
hm = pd.read_csv(S / '03_HUMAN_REANALYSIS/human_main_metrics.csv')
hm = hm[hm.subset == 'all'].set_index('metric')
assert (hm.n == 896).all()
for m, v in {'bleu4': .39, 'meteor': .38, 'rougeL': .37, 'cider_d': .36, 'csm4': .32, 'bertscore_f': .36, 'sbert': .43,
             'spice_f': .33, 'llm_judge': .48, 'b_kiritchenko_macro': .60, 'b_lca_f': .60, 'b_hcm': .59,
             'b_taxaware_axisflat': .57, 'ce_f1': .61}.items():
    assert round(hm.loc[m, 'spearman'], 2) == v, m

# ---- table
def f2(x):
    s = f'{x:.2f}'; return s[1:] if s.startswith('0.') else s
def rng(vals):
    lo, hi = f2(min(vals)), f2(max(vals)); return lo if lo == hi else f'{lo}--{hi}'
def cell(ok, txt, wide):
    mark = r'$\checkmark$' if ok else r'$\times$'
    if wide:
        return mark + r'\,{\footnotesize ' + txt + '}'
    return r'\shortstack{' + mark + r'\\[-0.5pt]{\scriptsize ' + txt + '}}'

PROPS = [  # key, short name, requirement (wide table)
    ('valid', 'Polarity', r'a correct caption beats one stating the opposite change'),
    ('ns', 'Specificity', r'a slightly coarser correct caption beats a wrong sibling'),
    ('st', 'Binding', r'a correct caption beats one with source and target swapped'),
    ('pr', r'Invented/missed$^\dagger$', r'invented and missed change events are scored separately'),
    ('human', r'Human $\rho$', r'agrees with image-grounded human ratings (Spearman, 896 captions)'),
]
values, cells = {}, {}
for key, *_ in PROPS[:3]:
    for col, (bm, _) in GROUPS.items():
        v = [bench[m][key] for m in bm]
        values[f'{key}|{col}'] = {m: bench[m][key] for m in bm} | {f'n|{m}': bench[m]['n_' + key] for m in bm}
        txt = rng(v) + (r'$^{*}$' if col == 'RICCE' and key == 'valid' else '') + (r'$^\ddagger$' if set(bm) & SUBSET else '')
        cells[key, col] = (min(v) >= 0.95, txt)
for col, (_, hmm) in GROUPS.items():
    cells['pr', col] = (col == 'RICCE', None)
    v = [hm.loc[m, 'spearman'] for m in hmm]
    values[f'human|{col}'] = {m: hm.loc[m, 'spearman'] for m in hmm}
    cells['human', col] = (None, rng(v))

def render(wide):
    head = ['Property'] + (['Requirement'] if wide else []) + [r'\textbf{RICCE}' if c == 'RICCE' else c for c in GROUPS]
    spec = (r'l>{\raggedright\arraybackslash}p{4.6cm}' if wide else 'l') + 'c' * len(GROUPS)
    rows = []
    for key, short, req in PROPS:
        cs = []
        for col in GROUPS:
            ok, txt = cells[key, col]
            if txt is None: cs.append(r'$\checkmark$' if ok else r'$\times$')
            elif ok is None: cs.append(txt)
            else: cs.append(cell(ok, txt, wide))
        rows.append(' & '.join([short] + ([req] if wide else []) + cs) + r'\\')
    return ('\\begin{tabular}{@{}' + spec + '@{}}\\toprule\n' + ' & '.join(head) + '\\\\\\midrule\n'
            + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')

body = render(False)
OUT.mkdir(exist_ok=True)
(OUT / 'tab_properties.tex').write_text(body)
(OUT / 'tab_properties_wide.tex').write_text(render(True))
json.dump(dict(values=values, groups={k: dict(bench=v[0], human=v[1]) for k, v in GROUPS.items()},
               sources=dict(bench='results_cvpr_final/ordering/scores/*.csv + 07_BASELINES/bertscore_canonical/bertscore.csv',
                            human='submission_cvpr27_new_version/03_HUMAN_REANALYSIS/human_main_metrics.csv (subset=all)'),
               rule='checkmark if every metric of the group satisfies >= 0.95 of the constraints (strict, ties fail)'),
          open(OUT / 'table_values.json', 'w'), indent=1)
print(body); print(render(True))
