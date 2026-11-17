"""v3 figures, from result files only.
  fig/ordering_heatmap_v3.pdf  -- strict ordering accuracy per constraint family; the RICCE
                                  column is the FINAL evaluator (results_v3/ordering), every
                                  other metric is unchanged (results_cvpr_final/ordering).
  fig/cross_dataset_v3.pdf     -- CE-F on changed vs. unchanged scenes and the share of changed
                                  scenes captioned as 'no change', per dataset and system
                                  (results_v3/systems/summary.json).
Colours: reference palette slots 1-2 (blue #2a78d6, orange #eb6834) and its blue sequential ramp;
unchanged-scene bars also carry a 45-degree hatch so the two series survive grayscale print."""
import json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'paper_tables' / 'fig'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
BLUE, ORANGE = '#2a78d6', '#eb6834'
SEQ = LinearSegmentedColormap.from_list('blue_seq', ['#f4f8fd', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'axes.linewidth': 0.6, 'pdf.fonttype': 42})

# ------------------------------------------------------------------ heatmap
R = ROOT / 'results_cvpr_final' / 'ordering'
f = pd.read_csv(R / 'ordering_accuracy_by_family.csv'); f = f[f.metric.notna()]
sv = pd.read_csv(R / 'semantic_validity_ordering.csv')
fin_fam = pd.read_csv(ROOT / 'results_v3' / 'ordering' / 'per_family_final_cef.csv').set_index('group').acc
fin_val = json.load(open(ROOT / 'results_v3' / 'ordering' / 'summary.json'))['F|validity']['acc']
mets = [('bleu4', 'BLEU-4'), ('rougeL', 'ROUGE-L'), ('meteor', 'METEOR'), ('cider_d_item', 'CIDEr-D'), ('bertscore', 'BERTScore'), ('sbert', 'SBERT'),
        ('spice', 'SPICE'), ('llm_judge', 'LLM judge'), ('kiritchenko_micro', 'Kiritchenko'), ('lca_f', 'LCA-F'), ('hcm', 'HCM'),
        ('taxaware_single', 'TA-single'), ('taxaware_axisflat', 'TA-flat'), ('FINAL', 'RICCE')]
rows = [('P0_paraphrase>P1_near_ancestor', 'paraphrase > near ancestor'), ('P1_near_ancestor>P2_distant_ancestor', 'near > distant ancestor'),
        ('P1_near_ancestor>P3_sibling_substitution', 'near ancestor > sibling'), ('P0_identity>multi_event_source_target_ambiguity', 'orig. > multi-event S/T ambiguity'),
        ('P0_identity>P7_event_omission', 'orig. > omission'), ('P0_identity>P8_event_hallucination', 'orig. > hallucination'),
        ('P0_identity>unsupported_descendant_specificity', 'orig. > unsupported detail'), ('P0_identity>P4_action_reversal', 'orig. > action reversal'),
        ('P0_identity>negation_flip', 'orig. > negation flip'), ('P0_identity>source_target_swap', 'orig. > S/T swap')]
M = np.full((len(rows) + 1, len(mets)), np.nan)
for j, (mk, _) in enumerate(mets):
    for i, (g, _) in enumerate(rows):
        if mk == 'FINAL':
            M[i, j] = fin_fam.get(g, np.nan)
        else:
            x = f[(f.metric == mk) & (f.group == g)]
            if len(x):
                M[i, j] = x.accuracy_strict.iloc[0]
    if mk == 'FINAL':
        M[len(rows), j] = fin_val
    else:
        x = sv[sv.metric == mk]
        if len(x):
            M[len(rows), j] = x.strict.iloc[0]
labels = [r[1] for r in rows] + ['correct > contradictory (validity suite)']
fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=300)
im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(len(mets))); ax.set_xticklabels([m[1] for m in mets], rotation=40, ha='right', fontsize=6.5)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6.5)
ax.tick_params(length=0)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        if np.isnan(v):
            ax.text(j, i, 'n/a', ha='center', va='center', fontsize=5, color=INK2)
        else:
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5.3, color='white' if v >= 0.62 else INK)
ax.axhline(len(rows) - 0.5, color='white', lw=2.0)
ax.axvline(len(mets) - 1.5, color='white', lw=2.0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_visible(False)
cb.set_label('strict ordering accuracy', fontsize=6.5)
plt.tight_layout(); fig.savefig(FIG / 'ordering_heatmap_v3.pdf'); plt.close(fig)
print('saved heatmap')

# ------------------------------------------------------------------ cross-dataset failure decomposition
X = pd.DataFrame(json.load(open(ROOT / 'results_v3' / 'systems' / 'summary.json'))['cross_dataset']).T
SYS = [('qwen', 'Qwen\n(0-shot)'), ('sft_seed42', 'SFT'), ('swmcc', 'SWMCC'), ('chg2cap', 'Chg2Cap')]
DS = [('LEVIR_CC', 'LEVIR-CC (in-domain)'), ('SECOND_CC', 'SECOND-CC (transfer)'), ('DUBAI_CC', 'Dubai-CC (transfer)')]
fig, axes = plt.subplots(2, 3, figsize=(7.0, 2.9), dpi=300, sharey='row', gridspec_kw=dict(height_ratios=[1.25, 1]))
x = np.arange(len(SYS)); w = 0.36
for c, (ds, dname) in enumerate(DS):
    ch = [float(X.loc[f'{ds}|{s}', 'ce_f_changed']) for s, _ in SYS]
    un = [float(X.loc[f'{ds}|{s}', 'ce_f_unchanged']) for s, _ in SYS]
    nc = [float(X.loc[f'{ds}|{s}', 'pred_nochange_share_on_changed']) for s, _ in SYS]
    a = axes[0, c]
    a.bar(x - w / 2 - 0.01, ch, w, color=BLUE, label='changed scenes', zorder=3)
    a.bar(x + w / 2 + 0.01, un, w, color='white', edgecolor=ORANGE, hatch='//////', linewidth=0.8, label='unchanged scenes', zorder=3)
    a.set_title(dname, fontsize=7, color=INK, pad=3)
    a.set_ylim(0, 1); a.set_xticks(x); a.set_xticklabels([])
    b = axes[1, c]
    b.bar(x, nc, 0.55, color=BLUE, zorder=3)
    b.set_ylim(0, 1); b.set_xticks(x); b.set_xticklabels([s[1] for s in SYS], fontsize=5.8)
    if ds == 'SECOND_CC':
        b.annotate('in-domain\ncheckpoint', xy=(4, nc[4]), xytext=(4, min(0.95, nc[4] + 0.42)), ha='center', fontsize=5.2, color=INK2,
                   arrowprops=dict(arrowstyle='-', color=INK2, lw=0.5))
    for ax_ in (a, b):
        ax_.yaxis.grid(True, color=GRID, lw=0.5, zorder=0); ax_.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax_.spines[sp].set_visible(False)
        ax_.tick_params(length=2)
axes[0, 0].set_ylabel('mean CE-F', fontsize=6.5)
axes[1, 0].set_ylabel('changed scenes\ncaptioned "no change"', fontsize=6.5)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, fontsize=6.5, frameon=False, loc='upper center', ncol=2, handlelength=1.6, bbox_to_anchor=(0.5, 1.0))
plt.tight_layout(h_pad=0.4, w_pad=0.6, rect=(0, 0, 1, 0.94)); fig.savefig(FIG / 'cross_dataset_v3.pdf'); plt.close(fig)
print('saved cross-dataset')
