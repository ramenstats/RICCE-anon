"""CVPR'27 figures, from result files only.
  12_PAPER/figures/ordering_heatmap.pdf  -- v3 heatmap (scripts/v3_make_figures.py, same style and rows) with the
                                            BERTScore column recomputed with canonical tokenization
  12_PAPER/figures/human_forest.pdf      -- Spearman with the mean overall rating, 95% cluster-bootstrap CI, every
                                            metric, all items vs real outputs (03_HUMAN_REANALYSIS/human_main_metrics.csv)
  12_PAPER/figures/human_examples.pdf    -- real rated outputs chosen by a fixed rule (below), with T1/T2 thumbnails
Colours: reference palette slots 1-2 (blue #2a78d6, orange #eb6834); the two series also differ in marker shape.
Example rule (fixed before looking at the examples): among real outputs, the two items with the largest
CE-F-minus-human percentile gap (CE-F too high), the two with the largest human-minus-CE-F gap (CE-F too low),
and the two with the largest CIDEr-D error among items where CE-F is within 0.1 percentile of the annotators."""
import json, textwrap
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / 'submission_cvpr27_new_version'
FIG = S / '12_PAPER' / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
OUT2 = S / '11_FIGURES'; OUT2.mkdir(parents=True, exist_ok=True)
src = (ROOT / 'scripts' / 'v3_make_figures.py').read_text()
hdr = src.split('# ------------------------------------------------------------------ heatmap')[0]
ns = {'__file__': str(ROOT / 'scripts' / 'v3_make_figures.py')}
exec(compile(hdr, 'v3_make_figures_header', 'exec'), ns)                    # palette and rcParams of v3
INK, INK2, GRID, BLUE, ORANGE, SEQ = (ns[k] for k in ('INK', 'INK2', 'GRID', 'BLUE', 'ORANGE', 'SEQ'))

# ------------------------------------------------------------------ heatmap (v3 code path, canonical BERTScore)
BC = S / '07_BASELINES' / 'bertscore_canonical'
if (BC / 'ordering_accuracy_by_family.csv').exists():
    heat = src.split('# ------------------------------------------------------------------ heatmap')[1].split("print('saved heatmap')")[0]
    heat = heat.replace("f = pd.read_csv(R / 'ordering_accuracy_by_family.csv'); f = f[f.metric.notna()]",
                        "f = pd.read_csv(R / 'ordering_accuracy_by_family.csv'); f = f[f.metric.notna() & (f.metric != 'bertscore')]\n"
                        "f = pd.concat([f, pd.read_csv(BC / 'ordering_accuracy_by_family.csv')])")
    heat = heat.replace("sv = pd.read_csv(R / 'semantic_validity_ordering.csv')",
                        "sv = pd.read_csv(R / 'semantic_validity_ordering.csv'); sv = sv[sv.metric != 'bertscore']\n"
                        "sv = pd.concat([sv, pd.DataFrame([json.load(open(BC / 'summary.json'))['semantic_validity']])])")
    heat = heat.replace("fig.savefig(FIG / 'ordering_heatmap_v3.pdf')", "fig.savefig(OUTH)")
    assert heat.count('BC /') == 2 and 'OUTH' in heat
    exec(heat, {**ns, 'BC': BC, 'OUTH': FIG / 'ordering_heatmap.pdf', 'json': json, 'pd': pd, 'np': np, 'plt': plt})
    print('heatmap (canonical BERTScore)')

# ------------------------------------------------------------------ forest plot
M = pd.read_csv(S / '03_HUMAN_REANALYSIS' / 'human_main_metrics.csv')
NAME = {'bleu4': 'BLEU-4', 'meteor': 'METEOR', 'rougeL': 'ROUGE-L', 'cider_d': 'CIDEr-D', 'csm4': 'CSM-4', 'bertscore_f': 'BERTScore', 'sbert': 'SBERT',
        'spice_f': 'SPICE', 'llm_judge': 'LLM judge (text)', 'vlm_judge': 'VLM judge (images)', 'b_kiritchenko_macro': 'hF (Kiritchenko)',
        'b_lca_f': 'LCA-F', 'b_hcm': 'HCM', 'b_taxaware_axisflat': 'TaxAware axis-flat', 'ce_f1': 'CE-F (ours)'}
a = M[(M.subset == 'all') & M.metric.isin(NAME)].set_index('metric'); r = M[(M.subset == 'real') & M.metric.isin(NAME)].set_index('metric')
order = a.spearman.sort_values().index.tolist()
fig, ax = plt.subplots(figsize=(3.3, 3.4), dpi=300)
y = np.arange(len(order))
for d, off, col, mk, lab in ((a, -0.17, BLUE, 'o', f'all items ($n$={int(a.n.iloc[0])})'), (r, 0.17, ORANGE, 's', f'real outputs ($n$={int(r.n.iloc[0])})')):
    d = d.loc[order]
    ax.hlines(y + off, d.ci_lo, d.ci_hi, color=col, lw=1.0)
    ax.plot(d.spearman, y + off, mk, ms=3.6, color=col, mec='white', mew=0.5, label=lab)
ax.set_yticks(y); ax.set_yticklabels([NAME[k] for k in order], fontsize=6.5)
for t in ax.get_yticklabels():
    if t.get_text().startswith('CE-F'): t.set_fontweight('bold')
ax.axvline(0, color=GRID, lw=0.8, zorder=0); ax.grid(axis='x', color=GRID, lw=0.4); ax.set_axisbelow(True)
ax.set_xlabel('Spearman with mean overall rating (95% CI)', fontsize=6.5); ax.tick_params(length=0, labelsize=6.5)
for s_ in ('top', 'right', 'left'): ax.spines[s_].set_visible(False)
ax.legend(fontsize=6, frameon=False, loc='lower right')
plt.tight_layout(); fig.savefig(FIG / 'human_forest.pdf'); fig.savefig(OUT2 / 'metric_vs_human.pdf'); plt.close(fig)
print('forest')

# ------------------------------------------------------------------ rule-selected real examples
F = pd.read_csv(S / '03_HUMAN_REANALYSIS' / 'item_frame.csv')
R_ = F[F.subset == 'real'].dropna(subset=['ce_f1']).copy()
pct = lambda c: R_[c].rank(pct=True)
R_['g_ce'] = pct('ce_f1') - pct('h_overall'); R_['g_cd'] = pct('cider_d') - pct('h_overall')
hi = R_.sort_values('g_ce', ascending=False).head(2).assign(rule='CE-F too high')
lo = R_.sort_values('g_ce').head(2).assign(rule='CE-F too low')
ok = R_[R_.g_ce.abs() <= 0.1].assign(ag=lambda d: d.g_cd.abs()).sort_values('ag', ascending=False).head(2).assign(rule='CIDEr-D off, CE-F close')
ex = pd.concat([hi, lo, ok])
H = ROOT / 'artifacts_cvpr_final' / 'human_eval' / 'image_grounded'
items = {json.loads(l)['item_id']: json.loads(l) for l in open(H / 'items.jsonl')}
fig, axes = plt.subplots(len(ex), 3, figsize=(7.0, 1.3 * len(ex)), dpi=200, gridspec_kw=dict(width_ratios=[1, 1, 4.2]))
for i, (_, e) in enumerate(ex.iterrows()):
    it = items[e.item_id]
    for j, t in enumerate(('t1_thumb', 't2_thumb')):
        axes[i, j].imshow(Image.open(H / it[t]).convert('RGB'), interpolation='nearest'); axes[i, j].axis('off')
        axes[i, j].set_title('$T_1$ (before)' if j == 0 else '$T_2$ (after)', fontsize=5.5, loc='left')
    txt = (f"[{e.rule}]  {e.dataset} {e.image_id}, system: {e.system}\n" + textwrap.fill('candidate: ' + it['candidate'], 105) + '\n'
           + textwrap.fill('reference hint: ' + it['references'][0], 105) + '\n'
           + f"annotators' mean overall {e.h_overall:.2f} (ann1/2/3: {e.h_ann1:.0f}/{e.h_ann2:.0f}/{e.h_ann3:.0f})   "
           + f"CE-F {e.ce_f1:.2f}   CIDEr-D {e.cider_d:.2f}   BERTScore {e.bertscore_f:.2f}")
    axes[i, 2].axis('off'); axes[i, 2].text(0, 0.5, txt, fontsize=5.6, va='center', ha='left', family='serif')
plt.tight_layout(); fig.savefig(FIG / 'human_examples.pdf'); fig.savefig(OUT2 / 'real_disagreement_examples.pdf'); plt.close(fig)
ex[['item_id', 'rule', 'dataset', 'image_id', 'system', 'h_overall', 'ce_f1', 'cider_d', 'bertscore_f']].to_csv(OUT2 / 'real_disagreement_examples.csv', index=False)
print('examples', ex.item_id.tolist())
