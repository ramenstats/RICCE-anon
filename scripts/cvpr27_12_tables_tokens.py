"""CVPR'27: every NEW table and every NEW number of the manuscript, generated from
result files only (no hand-typed numbers).

Inputs : submission_cvpr27_new_version/{03_HUMAN_REANALYSIS, 07_BASELINES, 08_METHOD_ABLATIONS, 10_TEMPORAL}
         results_v3/ablations/ablation_summary.csv (benchmark columns of the ablation table)
         paper_tables/tables_v3/ordering.tex (all rows except BERTScore, unchanged)
Outputs: submission_cvpr27_new_version/12_PAPER/tables/*.tex
         submission_cvpr27_new_version/12_PAPER/tokens_c27.json   {token: {value, source}}
Tokens that REPLACE a v4 value (BERTScore after the tokenizer fix) are listed under "overrides"."""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
S = ROOT / 'submission_cvpr27_new_version'
O, B, A, T = S / '03_HUMAN_REANALYSIS', S / '07_BASELINES', S / '08_METHOD_ABLATIONS', S / '10_TEMPORAL'
TAB = S / '12_PAPER' / 'tables'; TAB.mkdir(parents=True, exist_ok=True)
f2 = lambda x: f'{x:.2f}'
f3 = lambda x: f'{x:.3f}'
sg = lambda x: f'{x:+.2f}'
sg3 = lambda x: f'{x:+.3f}'
V, SRC, OVR = {}, {}, {}


def put(k, v, src):
    assert k not in V, k
    V[k] = v; SRC[k] = src


def write(name, lines):
    (TAB / f'{name}.tex').write_text('\n'.join(lines) + '\n')


M = pd.read_csv(O / 'human_main_metrics.csv'); D = pd.read_csv(O / 'human_metric_differences.csv')
m = lambda s, k: M[(M.subset == s) & (M.metric == k)].iloc[0]
dd = lambda s, k: D[(D.subset == s) & (D.other == k)].iloc[0]
cell = lambda r: f"{f2(r.spearman)} \\scriptsize[{f2(r.ci_lo)}, {f2(r.ci_hi)}]"
dcell = lambda r: f"{sg(r['diff'])} \\scriptsize[{f2(r.ci_lo)}, {f2(r.ci_hi)}]"
N_ALL, N_REAL = int(m('all', 'ce_f1').n), int(m('real', 'ce_f1').n)
put('C27_N_ALL', str(N_ALL), 'human_main_metrics all n'); put('C27_N_REAL', str(N_REAL), 'human_main_metrics real n')

# ------------------------------------------------------------------ main human table
GROUPS = [('String and embedding metrics', [('bleu4', 'BLEU-4'), ('meteor', 'METEOR'), ('rougeL', 'ROUGE-L'), ('cider_d', 'CIDEr-D'), ('csm4', 'CSM-4'),
                                            ('bertscore_f', 'BERTScore'), ('sbert', 'SBERT'), ('spice_f', 'SPICE')]),
          ('Zero-shot judges', [('llm_judge', 'LLM judge (text, references)'), ('vlm_judge', 'VLM judge (images only)$^\\dagger$')]),
          ('Prior hierarchical metrics on our event parses', [('b_kiritchenko_macro', 'hF (Kiritchenko)'), ('b_lca_f', 'LCA-F'), ('b_hcm', 'HCM'),
                                                              ('b_taxaware_axisflat', 'TaxAware axis-flat')])]
L = ['\\begin{tabular}{@{}lcc@{}}\\toprule', f'Metric & All items ($n{{=}}{N_ALL}$) & Real outputs ($n{{=}}{N_REAL}$)\\\\\\midrule']
for g, rows in GROUPS:
    L.append(f'\\multicolumn{{3}}{{@{{}}l}}{{\\emph{{{g}}}}}\\\\')
    for k, lab in rows:
        L.append(f"{lab} & {cell(m('all', k))} & {cell(m('real', k))}\\\\")
L.append('\\midrule')
a, r = m('all', 'ce_f1'), m('real', 'ce_f1')
L.append(f"\\textbf{{\\cef}} & \\textbf{{{f2(a.spearman)}}} \\scriptsize[{f2(a.ci_lo)}, {f2(a.ci_hi)}] & \\textbf{{{f2(r.spearman)}}} \\scriptsize[{f2(r.ci_lo)}, {f2(r.ci_hi)}]\\\\\\midrule")
for k, lab in (('cider_d', 'CIDEr-D'), ('bertscore_f', 'BERTScore'), ('llm_judge', 'LLM judge'), ('b_lca_f', 'LCA-F')):
    L.append(f"\\cef\\ $-$ {lab} & {dcell(dd('all', k))} & {dcell(dd('real', k))}\\\\")
L.append('\\bottomrule\\end{tabular}')
write('human_main_c27', L)

KEYS = {'CE': 'ce_f1', 'CIDER': 'cider_d', 'BERT': 'bertscore_f', 'LLM': 'llm_judge', 'VLM': 'vlm_judge', 'SBERT': 'sbert', 'SPICE': 'spice_f',
        'CSM': 'csm4', 'BLEU': 'bleu4', 'HF': 'b_kiritchenko_macro', 'LCA': 'b_lca_f', 'HCM': 'b_hcm', 'TAF': 'b_taxaware_axisflat', 'CEP': 'ce_p', 'CER': 'ce_r'}
for t, k in KEYS.items():
    for s, ss in (('ALL', 'all'), ('REAL', 'real')):
        put(f'C27_{t}_{ss.upper() if ss == "all" else "REAL"}', f2(m(ss, k).spearman), f'human_main_metrics {ss} {k}')
for t in ('CE',):
    for s in ('all', 'real'):
        put(f'C27_{t}_{s.upper()}_LO', f2(m(s, 'ce_f1').ci_lo), 'ci'); put(f'C27_{t}_{s.upper()}_HI', f2(m(s, 'ce_f1').ci_hi), 'ci')
hier = lambda s: [m(s, k).spearman for k in ('b_kiritchenko_macro', 'b_lca_f', 'b_hcm')]
put('C27_HIER_ALL', f'{f2(min(hier("all")))}--{f2(max(hier("all")))}', 'hF/LCA-F/HCM all'); put('C27_HIER_REAL', f'{f2(min(hier("real")))}--{f2(max(hier("real")))}', 'hF/LCA-F/HCM real')
ng = lambda s: [m(s, k).spearman for k in ('bleu4', 'meteor', 'rougeL', 'cider_d', 'csm4')]
put('C27_NG_ALL', f'{f2(min(ng("all")))}--{f2(max(ng("all")))}', 'bleu/meteor/rouge/cider/csm all'); put('C27_NG_REAL', f'{f2(min(ng("real")))}--{f2(max(ng("real")))}', 'real')
for t, k in (('CIDER', 'cider_d'), ('BERT', 'bertscore_f'), ('LLM', 'llm_judge'), ('VLM', 'vlm_judge'), ('LCA', 'b_lca_f'), ('HF', 'b_kiritchenko_macro'),
             ('HCM', 'b_hcm'), ('TAF', 'b_taxaware_axisflat'), ('SBERT', 'sbert'), ('CSM', 'csm4'), ('SPICE', 'spice_f')):
    for s in ('all', 'real'):
        x = dd(s, k); tag = f'C27_D_{t}_{s.upper()}'
        put(tag, sg(x['diff']), f'human_metric_differences {s} {k}'); put(tag + '_LO', f2(x.ci_lo), 'ci'); put(tag + '_HI', f2(x.ci_hi), 'ci')
prim = D[(D.family == 'primary') & D.subset.isin(['all', 'real'])]
put('C27_P_PRIMARY', f'{math.ceil(prim.p_holm.max() * 1000) / 1000:.3f}', 'max Holm p, primary family, all and real subsets (rounded up)')
put('C27_D_PRIM_MIN', sg(prim['diff'].min()), 'smallest primary difference')
ex = D[(D.family == 'exploratory') & D.subset.isin(['all', 'real']) & D.other.isin(['b_kiritchenko_macro', 'b_lca_f', 'b_hcm'])]
put('C27_HIER_DIFF_RNG', f"{sg(ex['diff'].min())} to {sg(ex['diff'].max())}", 'CE-F minus hF/LCA-F/HCM, all and real')
K = pd.read_csv(O / 'human_kendall.csv').set_index(['subset', 'metric'])
for s in ('all', 'real'):
    for t, k in (('CE', 'ce_f1'), ('CIDER', 'cider_d'), ('BERT', 'bertscore_f'), ('LLM', 'llm_judge')):
        put(f'C27_TAU_{t}_{s.upper()}', f2(K.loc[(s, k), 'kendall_tau_b']), f'human_kendall {s} {k}')
vj = pd.read_csv(B / 'human_captions_vlm_judge.csv')
se = ['bleu4', 'meteor', 'rougeL', 'cider_d', 'csm4', 'bertscore_f', 'sbert', 'spice_f']
put('C27_SE_MAX_REAL', f2(max(m('real', k).spearman for k in se)), 'max over string/embedding metrics, real outputs')
put('C27_VLM_SD', f2(vj.vlm_judge.std()), 'sd of VLM judge expected score over all captions')

# ------------------------------------------------------------------ reliability, robustness
R = pd.read_csv(O / 'rating_agreement.csv').set_index('scale')
AXN = {'source_target': 'source/target', 'change_nochange': 'change/no change'}
L = ['\\begin{tabular}{@{}lcccc@{}}\\toprule', 'Scale & All three [95\\% CI] & ann1--ann2 & ann1--ann3 & ann2--ann3\\\\\\midrule']
for ax, r_ in R.iterrows():
    L.append(f"{AXN.get(ax, ax)} & {f2(r_.alpha_all)} \\scriptsize[{f2(r_.ci_lo)}, {f2(r_.ci_hi)}] & {f2(r_.alpha_ann1_ann2)} & {f2(r_.alpha_ann1_ann3)} & {f2(r_.alpha_ann2_ann3)}\\\\")
L.append('\\bottomrule\\end{tabular}'); write('human_reliability_c27', L)
o = R.loc['overall']
put('C27_ALPHA', f2(o.alpha_all), 'rating_agreement overall'); put('C27_ALPHA_LO', f2(o.ci_lo), 'ci'); put('C27_ALPHA_HI', f2(o.ci_hi), 'ci')
put('C27_ALPHA_23', f2(o.alpha_ann2_ann3), 'pair'); put('C27_ALPHA_1X', f"{f2(min(o.alpha_ann1_ann2, o.alpha_ann1_ann3))}--{f2(max(o.alpha_ann1_ann2, o.alpha_ann1_ann3))}", 'pairs with ann1')
put('C27_ALPHA_OBJ', f2(R.loc['object', 'alpha_all']), 'object'); put('C27_ALPHA_ACT', f2(R.loc['action', 'alpha_all']), 'action')

Rb = pd.read_csv(O / 'per_annotator_robustness.csv')
rb = lambda s, t, k: Rb[(Rb.subset == s) & (Rb.target == t) & (Rb.metric == k)].iloc[0]
TG = [('h_overall', 'mean of three (primary)'), ('h_overall_median', 'median of three'), ('h_ann1', 'ann1 alone'), ('h_ann2', 'ann2 alone'),
      ('h_ann3', 'ann3 alone'), ('h_loo_ann1', 'without ann1'), ('h_loo_ann2', 'without ann2'), ('h_loo_ann3', 'without ann3')]
L = ['\\begin{tabular}{@{}lcccccc@{}}\\toprule', ' & \\multicolumn{4}{c}{All items} & \\multicolumn{2}{c}{Real outputs}\\\\',
     '\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}Overall rating used & \\cef & CIDEr-D & BERTScore & LLM judge & \\cef & CIDEr-D\\\\\\midrule']
for t, lab in TG:
    L.append(f"{lab} & " + ' & '.join(f2(rb(s, t, k).spearman) for s, k in (('all', 'ce_f1'), ('all', 'cider_d'), ('all', 'bertscore_f'), ('all', 'llm_judge'),
                                                                             ('real', 'ce_f1'), ('real', 'cider_d'))) + '\\\\')
L.append('\\bottomrule\\end{tabular}'); write('human_robustness_c27', L)
put('C27_MED_CE', f2(rb('all', 'h_overall_median', 'ce_f1').spearman), 'median consensus all')
loo = [rb('all', f'h_loo_{a}', 'ce_f1').spearman for a in ('ann1', 'ann2', 'ann3')]
put('C27_LOO_CE', f'{f2(min(loo))}--{f2(max(loo))}', 'leave-one-out all')
put('C27_ANN_CE', ', '.join(f2(rb('all', f'h_{a}', 'ce_f1').spearman) for a in ('ann1', 'ann2', 'ann3')), 'per annotator all')
put('C27_ANN1_CIDER', f2(rb('all', 'h_ann1', 'cider_d').spearman), 'ann1 cider')
dif_pos = all(rb(s, t, 'cider_d').diff_vs_cef > 0 for s in ('all', 'real') for t, _ in TG)
put('C27_LOO_DIFF_CIDER', f"{f2(min(rb('all', f'h_loo_{a}', 'cider_d').diff_vs_cef for a in ('ann1', 'ann2', 'ann3')))}", 'min CE-F minus CIDEr-D over leave-one-out, all')

W = pd.read_csv(O / 'within_system.csv')
SYS = [('qwen', 'Qwen2.5-VL (zero-shot)'), ('sft_seed42', 'LoRA SFT'), ('chg2cap', 'Chg2Cap'), ('swmcc', 'SWMCC'), ('perturbed', 'constructed cases')]
WM = [('ce_f1', '\\cef'), ('cider_d', 'CIDEr-D'), ('bertscore_f', 'BERTScore'), ('llm_judge', 'LLM'), ('vlm_judge', 'VLM'), ('b_lca_f', 'LCA-F')]
L = ['\\begin{tabular}{@{}lc' + 'c' * len(WM) + '@{}}\\toprule', 'Captions & $n$ & ' + ' & '.join(l for _, l in WM) + '\\\\\\midrule']
for s_, lab in SYS:
    g = W[W.system == s_].set_index('metric')
    L.append(f"{lab} & {int(g.loc['ce_f1', 'n'])} & " + ' & '.join(f2(g.loc[k, 'spearman']) for k, _ in WM) + '\\\\')
L.append('\\bottomrule\\end{tabular}'); write('human_within_system_c27', L)
w = W.pivot(index='system', columns='metric', values='spearman')
put('C27_WS_CE_MIN', f2(w.loc[[s for s, _ in SYS], 'ce_f1'].min()), 'within-system min CE-F')
put('C27_WS_BEST', str(int(sum(w.loc[s, 'ce_f1'] >= w.loc[s, [k for k, _ in WM if k != 'ce_f1']].max() for s, _ in SYS))), 'systems where CE-F is highest')
put('C27_QWEN_CIDER', f2(w.loc['qwen', 'cider_d']), 'qwen cider'); put('C27_QWEN_CE', f2(w.loc['qwen', 'ce_f1']), 'qwen ce')

# ------------------------------------------------------------------ pairwise
P = pd.read_csv(O / 'pairwise_preferences.csv')
pw = lambda g, k: P[(P.group == g) & (P.source == 'all') & (P.other == k)].iloc[0]
PM = [('cider_d', 'CIDEr-D'), ('csm4', 'CSM-4'), ('bertscore_f', 'BERTScore'), ('sbert', 'SBERT'), ('spice_f', 'SPICE'), ('llm_judge', 'LLM judge'),
      ('vlm_judge', 'VLM judge'), ('bleu4', 'BLEU-4'), ('meteor', 'METEOR'), ('rougeL', 'ROUGE-L')]
L = ['\\begin{tabular}{@{}lcccccc@{}}\\toprule', ' & \\multicolumn{3}{c}{Disagreement-sampled pairs} & \\multicolumn{3}{c}{Agreement controls}\\\\',
     '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}Other metric & $n$ & \\cef & other & $n$ & \\cef & other\\\\\\midrule']
for k, lab in PM:
    a_, c_ = pw('disagree', k), pw('agree', k)
    L.append(f"{lab} & {int(a_.n_matched)} & {f2(a_.cef_acc)} & {f2(a_.other_acc)} & {int(c_.n_matched)} & {f2(c_.cef_acc)} & {f2(c_.other_acc)}\\\\")
L.append('\\bottomrule\\end{tabular}'); write('human_pairwise_c27', L)
x = pw('all', 'cider_d'); put('C27_PW_N', str(int(x.n_matched)), 'matched pairs ce vs cider'); put('C27_PW_CE', f2(x.cef_acc), 'acc'); put('C27_PW_CIDER', f2(x.other_acc), 'acc')
x = pw('all', 'sbert'); put('C27_PW_SBERT', f2(x.other_acc), 'acc sbert'); put('C27_PW_CE_SB', f2(x.cef_acc), 'acc ce on sbert-matched'); put('C27_PW_N_SB', str(int(x.n_matched)), 'n')
c = pw('agree', 'cider_d'); put('C27_PW_CTRL_N', str(int(c.n_matched)), 'controls'); put('C27_PW_CTRL_CE', f2(c.cef_acc), 'controls'); put('C27_PW_CTRL_CIDER', f2(c.other_acc), 'controls')
x = pw('all', 'llm_judge'); put('C27_PW_LLM', f2(x.other_acc), 'llm'); put('C27_PW_CE_LLM', f2(x.cef_acc), 'ce on llm-matched'); put('C27_PW_N_LLM', str(int(x.n_matched)), 'n')

# ------------------------------------------------------------------ variants on human data + ablation table
HV = pd.read_csv(A / 'human_variant_agreement.csv')
hv = lambda s, v: HV[(HV.subset == s) & (HV.variant == v)].iloc[0]
ab = pd.read_csv(ROOT / 'results_v3/ablations/ablation_summary.csv').set_index('variant')
ROWS = [('A0_full', 'Full \\method'), ('A1_no_hierarchy', 'A1 no hierarchy'), ('A2_no_decomposition', 'A2 no decomposition'),
        ('A3_no_binding', 'A3 no binding'), ('A4_no_gate', 'A4 no gate'), ('A5_unordered_source_target', 'A5 unordered S/T'),
        ('A6_greedy', 'A6 greedy alignment'), ('A7_no_nochange_state', 'A7 no no-change state'), ('A8_single_assignment', 'A8 one assignment'),
        ('A9_root_counted', 'A9 root counted'), ('A10_root_excluded', 'A10 root kept as node'), ('A11_gate_any_object', 'A11 gate across objects'),
        ('DEPTH_minus1', 'Depth $-1$'), ('DEPTH_minus2', 'Depth $-2$')]
L = ['\\begin{tabular}{@{}lcccccccc@{}}\\toprule', ' & \\multicolumn{6}{c}{\\bench and systems} & \\multicolumn{2}{c}{Human $\\rho$}\\\\',
     '\\cmidrule(lr){2-7}\\cmidrule(lr){8-9}Variant & Core & Valid. & N$>$S & Rev. & S/T & $\\tau$ & all & real\\\\\\midrule']
for v, lab in ROWS:
    r_ = ab.loc[v]
    h = [f2(hv(s, v).spearman) if ((HV.subset == s) & (HV.variant == v)).any() else '--' for s in ('all', 'real')]
    L.append(f'{lab} & {r_.core:.3f} & {r_.validity:.3f} & {r_.near_gt_sibling:.3f} & {r_.cef_action_reversal:.2f} & {r_.cef_source_target_swap:.2f} & {r_.tau_systems:.2f} & {h[0]} & {h[1]}\\\\')
L.append('\\bottomrule\\end{tabular}'); write('ablation_c27', L)
L = ['\\begin{tabular}{@{}lcccc@{}}\\toprule', 'Variant & $\\rho$ all & full$-$variant [95\\% CI] & $\\rho$ real & full$-$variant [95\\% CI]\\\\\\midrule']
for v in HV.variant.unique():
    a_, r_ = hv('all', v), hv('real', v)
    L.append(f"{v.replace('_', ' ')} & {f2(a_.spearman)} & {sg3(a_.full_minus_variant)} \\scriptsize[{f3(a_.diff_ci_lo)}, {f3(a_.diff_ci_hi)}] & "
             f"{f2(r_.spearman)} & {sg3(r_.full_minus_variant)} \\scriptsize[{f3(r_.diff_ci_lo)}, {f3(r_.diff_ci_hi)}]\\\\")
L.append('\\bottomrule\\end{tabular}'); write('human_variants_c27', L)
for t, v in (('NOGATE', 'A4_no_gate'), ('NOBIND', 'A3_no_binding'), ('NODEC', 'A2_no_decomposition'), ('ROOTC', 'A9_root_counted'), ('ONEASS', 'A8_single_assignment'),
             ('NOHIER', 'A1_no_hierarchy'), ('GANY', 'A11_gate_any_object')):
    for s in ('all', 'real'):
        r_ = hv(s, v); tag = f'C27_V_{t}_{s.upper()}'
        put(tag, sg3(r_.full_minus_variant), f'human_variant_agreement {s} {v} (full minus variant)')
        put(tag + '_LO', f3(r_.diff_ci_lo), 'ci'); put(tag + '_HI', f3(r_.diff_ci_hi), 'ci')
soft = HV[HV.variant.str.startswith('G_soft_') & ~HV.variant.isin(['G_soft_0', 'G_soft_1'])]
sa = soft[soft.subset == 'all']; sr = soft[soft.subset == 'real']
put('C27_SOFT_ALL', f"{sg3(-sa.full_minus_variant.max())} to {sg3(-sa.full_minus_variant.min())}", 'soft gate gamma 0.05-0.5 minus hard, all')
put('C27_SOFT_REAL', f"{sg3(-sr.full_minus_variant.max())} to {sg3(-sr.full_minus_variant.min())}", 'soft gate minus hard, real')
DJ = pd.read_csv(A / 'dual_vs_joint_on_rated_captions.csv').set_index('subset')
put('C27_V_N_ALL', str(int(hv('all', 'A0_full').n)), 'variant comparison n all'); put('C27_V_N_REAL', str(int(hv('real', 'A0_full').n)), 'variant comparison n real')
raise_ = [-hv('real', v).full_minus_variant for v in ('A2_no_decomposition', 'A3_no_binding', 'A11_gate_any_object')]
assert all(hv('real', v).diff_ci_hi < 0 for v in ('A3_no_binding', 'A11_gate_any_object')), 'A3/A11 raise claim not significant'
assert hv('real', 'A2_no_decomposition').diff_ci_lo < 0 < hv('real', 'A2_no_decomposition').diff_ci_hi, 'A2 difference became significant'
assert all(hv('real', v).diff_ci_lo > 0 for v in ('A9_root_counted', 'A10_root_excluded')), 'A9/A10 lower claim not significant'
assert hv('real', 'A4_no_gate').diff_ci_lo < 0 < hv('real', 'A4_no_gate').diff_ci_hi, 'A4 difference on real outputs became significant'
put('C27_V_RAISE_REAL', f3(max(raise_)), 'max raise of A2/A3/A11 on real outputs (variant minus full)')
put('C27_DJ_DIFFER', f"{int(DJ.loc['all', 'n_differ'])} of {int(DJ.loc['all', 'n']):,}".replace(',', '{,}'), 'dual vs joint differ')

G = pd.read_csv(A / 'gate_vs_human.csv')
gg = lambda s, a_: G[(G.subset == s) & (G.gate_active == a_)].iloc[0]
L = ['\\begin{tabular}{@{}llccccc@{}}\\toprule', 'Captions & gate & $n$ & overall & share $\\leq2$ & \\cef & \\cef\\ no gate\\\\\\midrule']
for s, lab in (('real', 'real outputs'), ('constructed', 'constructed')):
    for a_ in (True, False):
        r_ = gg(s, a_)
        L.append(f"{lab} & {'acts' if a_ else 'silent'} & {int(r_.n)} & {f2(r_.mean_overall)} & {f2(r_.share_overall_le_2)} & {f2(r_.mean_cef)} & {f2(r_.mean_cef_no_gate)}\\\\")
L.append('\\bottomrule\\end{tabular}'); write('gate_human_c27', L)
ra, rs = gg('real', True), gg('real', False)
put('C27_GATE_N', str(int(ra.n)), 'gate active real'); put('C27_GATE_NREAL', str(int(ra.n + rs.n)), 'real total')
put('C27_GATE_OV_ON', f2(ra.mean_overall), 'mean overall gate on'); put('C27_GATE_OV_OFF', f2(rs.mean_overall), 'gate off')
put('C27_GATE_LE2_ON', f'{100 * ra.share_overall_le_2:.0f}\\%', 'share <=2 on'); put('C27_GATE_LE2_OFF', f'{100 * rs.share_overall_le_2:.0f}\\%', 'off')

# ------------------------------------------------------------------ temporal fixed cohort
TF = pd.read_csv(T / 'fixed_denominator_results.csv'); TD = pd.read_csv(T / 'fixed_cohort_paired_differences.csv')
g_ = TF.groupby('group').agg(['mean', 'std'])
L = ['\\begin{tabular}{@{}lccccc@{}}\\toprule', 'Model & cond.\\ inv.\\ (model $n$) & E1 & E2 & E3 & E4\\\\\\midrule']
for gname, lab in (('sft', 'LoRA SFT'), ('sft2x', 'SFT, step-matched'), ('swapsft', 'SFT + swap')):
    r_ = g_.loc[gname]
    L.append(f"{lab} & {f3(r_[('E0_conditional_inversion', 'mean')])} ({r_[('E0_n', 'mean')]:.0f}) & " + ' & '.join(
        f"{f3(r_[(c, 'mean')])}$\\pm${f3(r_[(c, 'std')])}" for c in ('E1_fixed_gt_directional', 'E2_swapped_reverses_reference', 'E3_fixed_gt_nochange', 'E4_all_pairs')) + '\\\\')
L.append('\\midrule')
for comp, lab in (('swapsft - sft2x', 'swap $-$ step-matched'),):
    cells = []
    for e in ('E1', 'E2', 'E3', 'E4'):
        x = TD[(TD.comparison == comp) & (TD.endpoint == e)].iloc[0]; cells.append(f"{sg3(x['diff'])} \\scriptsize[{f3(x.ci_lo)}, {f3(x.ci_hi)}]")
    L.append(f'{lab} & & ' + ' & '.join(cells) + '\\\\')
L.append('\\bottomrule\\end{tabular}'); write('temporal_fixed_c27', L)
coh = pd.read_csv(T / 'frozen_300_image_pairs.csv').cohort.value_counts()
put('C27_T_N', str(int(coh['directional'])), 'gt-directional cohort'); put('C27_T_NOTHER', str(int(coh['other'])), 'other cohort')
put('C27_T_E0N_RNG', f"{int(TF.E0_n.min())}--{int(TF.E0_n.max())}", 'model-directional denominators over runs'); put('C27_T_NNC', str(int(coh['no_change'])), 'gt no-change cohort')
put('C27_T_SWAP', f3(g_.loc['swapsft', ('E1_fixed_gt_directional', 'mean')]), 'E1 swapsft'); put('C27_T_SFT2X', f3(g_.loc['sft2x', ('E1_fixed_gt_directional', 'mean')]), 'E1 sft2x')
x = TD[(TD.comparison == 'swapsft - sft2x') & (TD.endpoint == 'E1')].iloc[0]
put('C27_T_D', sg3(x['diff']), 'E1 diff'); put('C27_T_D_LO', f3(x.ci_lo), 'ci'); put('C27_T_D_HI', f3(x.ci_hi), 'ci')
x = TD[(TD.comparison == 'swapsft - sft2x') & (TD.endpoint == 'E3')].iloc[0]
put('C27_T_NC_D', sg3(x['diff']), 'E3 diff'); put('C27_T_NC_LO', f3(x.ci_lo), 'ci'); put('C27_T_NC_HI', f3(x.ci_hi), 'ci')
x = TD[(TD.comparison == 'swapsft - sft2x') & (TD.endpoint == 'E4')].iloc[0]
put('C27_T_E4_D', sg3(x['diff']), 'E4 diff'); put('C27_T_E4_LO', f3(x.ci_lo), 'ci'); put('C27_T_E4_HI', f3(x.ci_hi), 'ci')

# ------------------------------------------------------------------ input contracts of the human-study metrics
L = ['\\begin{tabular}{@{}llll@{}}\\toprule', 'Metric & Sees & Model / resource & Coverage\\\\\\midrule',
     'BLEU-4, METEOR, ROUGE-L & caption, references & -- & all\\\\',
     'CIDEr-D & caption, references & IDF from the test-split references & all\\\\',
     'CSM-4~\\cite{hcfnet2026} & caption, references & authors\' word-level rule files$^\\ast$ & all\\\\',
     'BERTScore & caption, references & RoBERTa-large, layer 17, no rescaling & all\\\\',
     'SBERT & caption, references & all-MiniLM-L6-v2, max cosine & all\\\\',
     'SPICE & caption, references & official Java implementation & all\\\\',
     'LLM judge & caption, references & Qwen2.5-3B-Instruct, greedy, 1--5 & all\\\\',
     'VLM judge & caption, $T_1$, $T_2$ & Qwen2.5-VL-3B-Instruct, expected 1--5 & all\\\\',
     'hF, LCA-F, HCM, TaxAware & caption, references & our parser and taxonomies & all\\\\',
     '\\cep, \\cer, \\cef & caption, references & our parser and taxonomies & scorable\\\\',
     'FMScore~\\cite{ricci2026fmscore} & caption, reference facts & LLM fact matching & not run$^\\S$\\\\',
     '\\bottomrule\\end{tabular}']
write('baseline_contracts_c27', L)

# ------------------------------------------------------------------ appendix-only numbers
HA = S / '02_HUMAN_AUDIT'
rc = pd.read_csv(HA / 'round_comparison.csv').set_index(['round', 'annotator'])
r1 = rc.loc['round1']; r2 = rc.loc['round2']
assert (r1.taskB_rows_with_ratings == 477).all() and (r1.taskB_invalid_or_empty_preference == 477).all() and (r1.na_cells_on_na_scales == 0).all()
assert (r2.invalid_rating_cells == 0).all() and (r2.taskB_rows_with_ratings == 0).all() and (r2.na_on_scales_without_na == 0).all()
put('C27_R1_DISTINCT', f"{int(r1.distinct_rating_rows.min())}--{int(r1.distinct_rating_rows.max())}", 'round_comparison round1 distinct rows')
put('C27_R2_NA', f"{int(r2.na_cells_on_na_scales.min()):,}--{int(r2.na_cells_on_na_scales.max()):,}".replace(',', '{,}'), 'round2 NA cells per file')
ci_ = pd.read_csv(HA / 'cross_file_identity.csv')
assert (ci_[ci_.file_1.str.startswith('round1') & ci_.file_2.str.startswith('round2')].identical_rating_rows == 0).all()
r22 = ci_[ci_.file_1.str.startswith('round2') & ci_.file_2.str.startswith('round2')].set_index(['file_1', 'file_2']).share_identical
put('C27_R2_IDENT_MAX', f"{100 * r22[('round2/ann2', 'round2/ann3')]:.1f}\\%", 'cross_file_identity ann2-ann3')
assert r22.idxmax() == ('round2/ann2', 'round2/ann3')
put('C27_R2_IDENT_MIN', f"{100 * max(r22[('round2/ann1', 'round2/ann2')], r22[('round2/ann1', 'round2/ann3')]):.1f}\\%", 'cross_file_identity pairs with ann1 (max of the two)')
put('C27_CIDER_MIX_ALL', f2(m('all', 'cider_mixed_v4').spearman), 'pooled proxy column, all items')
put('C27_LLMEV_ALL', f2(m('all', 'llm_judge_ev').spearman), 'text judge expected score, all')
put('C27_ANN1_CE', f2(rb('all', 'h_ann1', 'ce_f1').spearman), 'ann1 alone')
rg = pd.read_csv(O / 'regression.csv'); rg = rg[rg.model == 'fixed_effects_OLS_cluster_robust'].set_index('metric').coef
put('C27_REG_CE', f2(rg['ce_f1']), 'regression'); put('C27_REG_CIDER', f2(rg['cider_d']), 'regression'); put('C27_REG_BERT', f2(rg['bertscore_f']), 'regression')
CH = S / '04_CONFIRMATORY_HUMAN'
put('C27_H2_ITEMS', str(len(pd.read_csv(CH / 'random_sampling_manifest_KEY_DO_NOT_SHARE.csv'))), 'H2 items')
put('C27_H2_PAIRS', str(len(pd.read_csv(CH / 'pairs_manifest_KEY_DO_NOT_SHARE.csv'))), 'H2 pairs')
put('C27_P1_N', str(len(pd.read_csv(S / '05_PARSER_ORACLE' / 'gold_annotation_templates' / 'holdout_ids.csv'))), 'P1 captions')
put('C27_X1_N', str(len(pd.read_csv(S / '09_DOMAIN_TRANSFER' / 'independent_change_labels_sheet_ann1.csv'))), 'X1 pairs')
ga = pd.read_csv(A / 'gate_audit_KEY_DO_NOT_SHARE.csv')
put('C27_M1_ACT', str(int(ga.activated.sum())), 'gate audit activations'); put('C27_M1_NON', str(int((~ga.activated).sum())), 'non-activations')

# ------------------------------------------------------------------ canonical BERTScore (overrides v4 values)
BC = B / 'bertscore_canonical'
bc = json.load(open(BC / 'summary.json'))
ob = pd.read_csv(BC / 'ordering_accuracy_by_metric.csv'); ob = ob[(ob.dataset == 'ALL') & (ob.metric == 'bertscore')]
fam = pd.read_csv(BC / 'ordering_accuracy_by_family.csv'); fam = fam[fam.metric == 'bertscore']
LAD = ['P0_paraphrase>P1_near_ancestor', 'P1_near_ancestor>P2_distant_ancestor']
core = ob[ob.suites == 'core'].accuracy_strict.iloc[0]; ext = ob[ob.suites == 'extended'].accuracy_strict.iloc[0]
l = fam[fam.group.isin(LAD)]; anc = l.n_satisfied_strict.sum() / l.n_constraints.sum()
val = bc['semantic_validity']['strict']
old = (ROOT / 'paper_tables' / 'tables_v3' / 'ordering.tex').read_text().splitlines()
new = [f'BERTScore$^\\P$ & {f3(core)} & {f3(val)} & {f3(anc)} & {f3(ext)}\\\\' if x.startswith('BERTScore &') else x for x in old]
assert sum(x.startswith('BERTScore$^\\P$') for x in new) == 1
write('ordering_c27', new)
OVR['VAL_BERT'] = f2(val); OVR['ANC_BERT'] = f2(anc)
for k_, kind in (('C', 'correct, coarse'), ('R', 'reversed direction'), ('O', 'over-specific')):
    OVR[f'TZ_{k_}_BERT'] = f2(next(t['canonical'] for t in bc['teaser'] if t['kind'] == kind))
OVR['HP_BERT'] = f3(bc['pilot']['spearman_canonical'])
put('C27_BERT_CORE', f3(core), 'canonical bench core'); put('C27_BERT_VAL', f3(val), 'canonical validity')
mo = pd.read_csv(ROOT / 'results_cvpr_final/ordering/ordering_accuracy_by_metric.csv'); mo = mo[(mo.dataset == 'ALL') & (mo.suites == 'core')].set_index('metric').accuracy_strict
core_all = [mo[k] for k in ('bleu4', 'meteor', 'rougeL', 'cider_d_item', 'sbert', 'kiritchenko_micro', 'lca_f', 'hcm')] + [core]
OVR['CORE_RNG'] = f'{f2(min(core_all))}--{f2(max(core_all))}'
sv = pd.read_csv(ROOT / 'results_cvpr_final/ordering/semantic_validity_ordering.csv').set_index('metric').strict
fails = [1 - sv[k] for k in ('bleu4', 'meteor', 'rougeL', 'cider_d_item', 'sbert')] + [1 - val]
OVR['VAL_FAIL'] = f'{100 * min(fails):.0f}--{100 * max(fails):.0f}\\%'
json.dump({'tokens': {k: {'value': V[k], 'source': SRC[k]} for k in sorted(V)},
           'overrides': {k: {'value': v, 'source': 'canonical BERTScore (scripts/cvpr27_09_bertscore_canonical.py)'} for k, v in OVR.items()}},
          open(S / '12_PAPER' / 'tokens_c27.json', 'w'), indent=1)
print(f'{len(V)} tokens, {len(OVR)} overrides, tables: {sorted(p.name for p in TAB.glob("*.tex"))}')
print({k: V[k] for k in ('C27_CE_ALL', 'C27_CE_REAL', 'C27_HIER_ALL', 'C27_D_LCA_ALL', 'C27_V_NOGATE_ALL', 'C27_V_NOBIND_REAL', 'C27_P_PRIMARY')}, OVR)
