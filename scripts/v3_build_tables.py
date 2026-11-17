"""v3: every table of the v3 manuscript, generated from
result files only (no hand-typed numbers). Writes paper_tables/tables_v3/.
Also rescoring of the qualitative examples and a scoring-only timing under the
FINAL evaluator (results_v3/error_examples/, results_v3/runtime/)."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TV = ROOT / 'paper_tables' / 'tables_v3'; TV.mkdir(parents=True, exist_ok=True)
R3, RC, RM = ROOT / 'results_v3', ROOT / 'results_cvpr_final', ROOT / 'results_metric_v2'
f2, f3 = (lambda x: f'{x:.2f}'), (lambda x: f'{x:.3f}')


def w(name, body):
    (TV / f'{name}.tex').write_text(body)
    print('wrote', name)


def tab(spec, header, rows, extra=''):
    return '\\begin{tabular}{@{}' + spec + '@{}}\\toprule\n' + header + '\\\\\\midrule\n' + '\\\\\n'.join(rows) + '\\\\\n' + extra + '\\bottomrule\\end{tabular}\n'


# ------------------------------------------------------------------ ordering (main)
m = pd.read_csv(RC / 'ordering/ordering_accuracy_by_metric.csv'); m = m[m.dataset == 'ALL']
sv = pd.read_csv(RC / 'ordering/semantic_validity_ordering.csv').set_index('metric')
fam = pd.read_csv(RC / 'ordering/ordering_accuracy_by_family.csv')
fin = json.load(open(R3 / 'ordering/summary.json'))
LAD = ['P0_paraphrase>P1_near_ancestor', 'P1_near_ancestor>P2_distant_ancestor']
MET = [('bleu4', 'BLEU-4'), ('meteor', 'METEOR'), ('rougeL', 'ROUGE-L'), ('cider_d_item', 'CIDEr-D (per item)'), ('bertscore', 'BERTScore'),
       ('sbert', 'Sentence-BERT'), ('spice', 'SPICE$^\\ddagger$'), ('llm_judge', 'LLM judge$^\\ddagger$'), ('kiritchenko_micro', 'Kiritchenko hF (flat)'),
       ('lca_f', 'LCA-F (flat)'), ('hcm', 'HCM-F1 (flat)'), ('taxaware_single', 'TaxAware single-node'), ('taxaware_axisflat', 'TaxAware axis-flat')]
rows = []
for k, lab in MET:
    core = m[(m.metric == k) & (m.suites == 'core')].accuracy_strict.iloc[0]
    ext = m[(m.metric == k) & (m.suites == 'extended')].accuracy_strict.iloc[0]
    l = fam[(fam.metric == k) & (fam.group.isin(LAD))]
    anc = l.n_satisfied_strict.sum() / l.n_constraints.sum()
    rows.append(f'{lab} & {f3(core)} & {f3(sv.loc[k, "strict"])} & {f3(anc)} & {f3(ext)}')
rows.append(f'\\method (ours) & {f3(fin["F|core"]["acc"])} & {f3(fin["F|validity"]["acc"])}$^{{*}}$ & {f3(fin["F|ancestor_chain"]["acc"])} & {f3(fin["F|near_gt_sibling"]["acc"])}')
w('ordering', tab('lcccc', 'Metric & Core & Valid.$^\\S$ & Ancestor chain & near$>$sibling', rows))

# ------------------------------------------------------------------ flat vs event (main)
B = R3 / 'baselines'
if (B / 'diagnostics_summary_v2.csv').exists():
    dg = pd.read_csv(B / 'diagnostics_summary_v2.csv').set_index(['group', 'metric'])
    pt = pd.read_csv(B / 'perturbation_summary_v2.csv'); pt = pt[pt.dataset == 'ALL'].set_index(['type', 'metric'])
    MC = ['ricce', 'kiritchenko_micro', 'lca_f', 'hcm', 'taxaware_single', 'taxaware_axisflat']
    rows = ['\\multicolumn{7}{@{}l}{\\emph{Mean $F$ on real reference captions}}']
    for g, lab in [('st_swap', 'source/target swapped ($n{=}415$)'), ('bind_swap', 'event binding swapped ($n{=}547$)')]:
        rows.append(lab + ' & ' + ' & '.join(f2(dg.loc[(g, k), 'mean_F']) for k in MC))
    rows.append('\\multicolumn{7}{@{}l}{\\emph{Drop in $F$ vs.\\ the original caption (controlled perturbations)}}')
    for t, lab in [('P4_action_reversal', 'action reversal'), ('P9_change_state_reversal', 'change/no-change reversal'), ('P7_event_omission', 'event omission'),
                   ('P8_hallucination', 'hallucinated event'), ('P2_distant_ancestor', 'distant ancestor')]:
        rows.append(lab + ' & ' + ' & '.join(f2(pt.loc[(t, k), 'drop']) for k in MC))
    w('flat_vs_event', tab('lcccccc', 'Case & \\cef & Kiritchenko & LCA-F & HCM & TA-single & TA-flat', rows))

# ------------------------------------------------------------------ design decisions (main)
dj = json.load(open(R3 / 'dual_vs_joint' / 'dual_vs_joint_matching.json')) if (R3 / 'dual_vs_joint' / 'dual_vs_joint_matching.json').exists() else None
rm = json.load(open(R3 / 'root_mode' / 'summary_bench_properties.json'))
go = json.load(open(R3 / 'gate_object_match' / 'summary.json'))
gs = json.load(open(R3 / 'gate' / 'summary.json'))
abl = pd.read_csv(R3 / 'ablations' / 'ablation_summary.csv').set_index('variant')
if dj:
    joint = [v for k, v in dj.items() if 'joint' in k]
    jv = [v['recall_monotonicity']['violation_rate'] for v in joint]
    n_chk = dj['V2-A dual']['recall_monotonicity']['checked']
    oi_fail = sum(not v['order_invariant'] for v in joint)
    rows = [f'Alignment & one shared assignment ($S_F$, mean, min, geometric) & two assignments, $S_P$- and $S_R$-optimal & recall monotone under removal: 0/{n_chk} violations vs.\\ {100*min(jv):.1f}--{100*max(jv):.1f}\\%; order-invariant ({oi_fail} alternative fails)']
    ri, re_, ru = rm['include'], rm['exclude'], rm['unspecified']
    rows.append(f'Root-level labels & root counted; root excluded & root $=$ unspecified field & R1 root-only \\cef\\ $<0.5$: {f2(ri["R1_root_only"]["cef"])} / {f2(re_["R1_root_only"]["cef"])} / {f2(ru["R1_root_only"]["cef"])}; R2--R3 vague $=$ absent: only ``unspecified\'\' and ``counted\'\' pass; R4 near$>$sibling: {f3(ri["bench_near_gt_sibling"]["acc"])} / {f3(re_["bench_near_gt_sibling"]["acc"])} / {f3(ru["bench_near_gt_sibling"]["acc"])}')
    a, b = go['object_match=False'], go['object_match=True']
    rows.append(f'Gate scope & any aligned pair & pairs about compatible objects & stress recall {100*a["enlarged227"]["recall"]:.0f}\\%$\\to${100*b["enlarged227"]["recall"]:.0f}\\%; real reference pairs gated {100*a["real_refpair"]["rate"]:.1f}\\%$\\to${100*b["real_refpair"]["rate"]:.1f}\\%')
    st = gs['stress_enlarged227']['residual_cef_expected_gated']
    rows.append(f'Gate strength & soft ($\\gamma\\in[0.05,0.5]$); none & hard & no property excludes soft; residual \\cef\\ of a reversal {f2(st["hard"])} (hard), {f2(st["soft0.05"])}--{f2(st["soft0.5"])} (soft), {f2(st["none"])} (none); ranking changes only without a gate ($\\tau{{=}}{gs["system_tau_vs_hard"]["none"]:.2f}$)')
    w('design', '\\begin{tabular}{@{}p{1.55cm}p{2.9cm}p{2.5cm}p{8.6cm}@{}}\\toprule\nChoice & Alternatives compared & Adopted & Deciding evidence (property, stated before comparison)\\\\\\midrule\n'
      + '\\\\\n'.join(rows) + '\\\\\n\\bottomrule\\end{tabular}\n')

# ------------------------------------------------------------------ system benchmark (main)
S = pd.read_csv(R3 / 'systems/summary.csv')
NAME = {'qwen': 'Qwen2.5-VL (zero-shot)', 'sft_seed42': 'Qwen2.5-VL + LoRA SFT', 'swmcc': 'SWMCC', 'chg2cap': 'Chg2Cap',
        'rsiccformer': 'RSICCFormer', 'promptcc': 'PromptCC', 'rscama': 'RSCaMa'}
rows = []
for sy in ['qwen', 'sft_seed42', 'swmcc', 'chg2cap', 'rsiccformer', 'promptcc', 'rscama']:
    l = S[(S.dataset == 'LEVIR_CC') & (S.system == sy)].iloc[0]
    o = []
    for ds in ('SECOND_CC', 'DUBAI_CC'):
        x = S[(S.dataset == ds) & (S.system == sy)]
        o.append(f3(x.ce_f.iloc[0]) if len(x) else '--')
    rows.append(f'{NAME[sy]} & {f3(l.ce_p)} & {f3(l.ce_r)} & {f3(l.ce_f)} \\scriptsize[{f3(l.ce_f_lo)},{f3(l.ce_f_hi)}] & {f3(l.bleu4)} & {f3(l.cider_d)} & {o[0]} & {o[1]}')
w('system', '\\begin{tabular}{@{}lccccccc@{}}\\toprule\n & \\multicolumn{5}{c}{LEVIR-CC} & SECOND-CC & Dubai-CC\\\\\n\\cmidrule(lr){2-6}\\cmidrule(lr){7-7}\\cmidrule(lr){8-8}\nSystem & \\cep & \\cer & \\cef & BLEU-4 & CIDEr-D & \\cef & \\cef\\\\\\midrule\n'
  + '\\\\\n'.join(rows) + '\\\\\n\\bottomrule\\end{tabular}\n')

# ------------------------------------------------------------------ cross-dataset (supplement, full)
X = pd.DataFrame(json.load(open(R3 / 'systems/summary.json'))['cross_dataset']).T
DSN = {'LEVIR_CC': 'LEVIR-CC', 'SECOND_CC': 'SECOND-CC', 'DUBAI_CC': 'Dubai-CC'}
rows = []
for sy in ['qwen', 'sft_seed42', 'swmcc', 'chg2cap']:
    for i, ds in enumerate(['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC']):
        r = X.loc[f'{ds}|{sy}']
        lab = (f'\\multirow{{3}}{{*}}{{{NAME[sy].replace("Qwen2.5-VL ", "Qwen ")}}}' if i == 0 else '')
        rows.append(f'{lab} & {DSN[ds]} & {f3(float(r.ce_f))} & {f3(float(r.ce_f_changed))} & {f3(float(r.ce_f_unchanged))} & {100*float(r.pred_nochange_share_on_changed):.0f}\\% & {int(r.n_clean)} & {f3(float(r.ce_f_clean))} & {100*float(r.clean_subset_ref_changed_share):.0f}\\%')
    rows[-1] += '\\\\\\midrule' if sy != 'chg2cap' else ''
body = '\\begin{tabular}{@{}llccccccc@{}}\\toprule\nSystem & Dataset & \\cef\\ all & changed & unchanged & changed$\\to$``no change\'\' & $n$ parse-clean & \\cef\\ parse-clean & changed share in parse-clean\\\\\\midrule\n'
body += '\\\\\n'.join(rows).replace('\\\\\\midrule\\\\', '\\\\\\midrule') + '\\\\\n\\bottomrule\\end{tabular}\n'
w('cross_dataset', body)

# ------------------------------------------------------------------ natural multi-event (main)
NM = json.load(open(R3 / 'natural_multievent/summary.json'))
MM = [('cef', '\\cef'), ('kiritchenko_micro', 'Kiritchenko'), ('hcm', 'HCM'), ('taxaware_axisflat', 'TA-flat')]
rows = []
for sy in ['qwen', 'sft_seed42', 'swmcc', 'chg2cap']:
    rows.append(NAME[sy].replace('Qwen2.5-VL ', 'Qwen ').replace('(zero-shot)', 'zero-shot') + ' & ' + ' & '.join(f3(NM['metrics'][k]['per_system'][sy]['mean']) for k, _ in MM))
sz = lambda x: ('0.000' if abs(x) < 0.0005 else f'{x:+.3f}')
extra = '\\midrule\nSFT $-$ zero-shot & ' + ' & '.join(sz(NM["metrics"][k]["sft_minus_qwen"]["diff"]) for k, _ in MM) + '\\\\\n'
extra += '\\quad 95\\% CI & ' + ' & '.join(f'\\scriptsize[{NM["metrics"][k]["sft_minus_qwen"]["lo"]:+.3f},{NM["metrics"][k]["sft_minus_qwen"]["hi"]:+.3f}]' for k, _ in MM) + '\\\\\n'
extra += '$\\tau$ vs.\\ full-test order & ' + ' & '.join(f'{NM["metrics"][k]["tau_vs_full_test_order"]:.1f}' for k, _ in MM) + '\\\\\n'
w('natural_multievent', tab('lcccc', 'System & ' + ' & '.join(l for _, l in MM), rows, extra))

# ------------------------------------------------------------------ temporal (main)
ts = json.load(open(RC / 'temporal_equivariance/summary.json'))['per_metric']
e10 = json.load(open(ROOT / 'results/E10/LEVIR_CC/metrics.json'))
LG = pd.read_csv(R3 / 'systems/learning_groups.csv').set_index('group')
q = S[(S.dataset == 'LEVIR_CC') & (S.system == 'qwen')].iloc[0]
pm = lambda g, k: f'{ts[g][k]["mean"]:.3f} $\\pm$ {ts[g][k]["std"]:.3f}'
rows = [f'Qwen2.5-VL zero-shot (1) & {f3(q.ce_f)} & {f3(q.cider_d)} & {f3(e10["directional_inversion_accuracy"])} & {f3(e10["no_change_consistency_under_swap"])}']
for g, lab in [('sft', 'LoRA SFT (3)'), ('sft2x', 'SFT, $1.9\\times$ steps (3)'), ('swapsft', 'SFT + swap augmentation (3)')]:
    rows.append(f'{lab} & {LG.loc[g, "mean"]:.3f} $\\pm$ {LG.loc[g, "std"]:.3f} & {pm(g, "cider_d")} & {pm(g, "inversion_acc")} & {pm(g, "nochange_consistency")}')
w('temporal', tab('lcccc', 'Model (runs) & \\cef & CIDEr-D & Inversion acc. & No-change cons.', rows))

# ------------------------------------------------------------------ ablations (main + supplement)
COMP = [('A0_full', 'Full \\method'), ('A1_no_hierarchy', 'A1 no hierarchy (exact node)'), ('A2_no_decomposition', 'A2 no event decomposition'),
        ('A3_no_binding', 'A3 no event binding'), ('A4_no_gate', 'A4 no contradiction gate'), ('A5_unordered_source_target', 'A5 unordered source/target'),
        ('A6_greedy', 'A6 greedy alignment'), ('A7_no_nochange_state', 'A7 no explicit no-change'), ('A8_single_assignment', 'A8 one shared assignment'),
        ('A9_root_counted', 'A9 root counted as ancestor'), ('A10_root_excluded', 'A10 root label kept as a node'), ('A11_gate_any_object', 'A11 gate across objects'),
        ('DEPTH_minus1', 'Ontology depth $-1$'), ('DEPTH_minus2', 'Ontology depth $-2$')]
rows = []
for v, lab in COMP:
    r = abl.loc[v]
    rows.append(f'{lab} & {f3(r.core)} & {f3(r.validity)} & {f3(r.near_gt_sibling)} & {f2(r.cef_action_reversal)} & {f2(r.cef_source_target_swap)} & {r.tau_systems:.2f} & {f3(r.natural_multievent_mean)}')
w('ablation', tab('lccccccc', 'Variant & Core & Valid. & near$>$sib. & \\cef\\ act.\\ rev. & \\cef\\ S/T swap & $\\tau$ & Nat.\\ multi', rows))
wr = abl[abl.group.isin(['weights', 'axis_loo'])]
rows = [f'{v.replace("_", " ")} & {f3(r.core)} & {f3(r.validity)} & {f3(r.near_gt_sibling)} & {f3(r.mean_cef_bench)} & {r.tau_systems:.2f}' for v, r in wr.iterrows()]
w('ablation_weights', tab('lccccc', 'Variant & Core & Valid. & near$>$sib. & mean \\cef & $\\tau$', rows))

# ------------------------------------------------------------------ gate (main)
G = gs
rows = []
for k, lab in [('hard', 'hard (used)'), ('soft0.05', 'soft, $\\gamma{=}0.05$'), ('soft0.1', 'soft, $\\gamma{=}0.1$'), ('soft0.2', 'soft, $\\gamma{=}0.2$'),
               ('soft0.3', 'soft, $\\gamma{=}0.3$'), ('soft0.5', 'soft, $\\gamma{=}0.5$'), ('none', 'none')]:
    b_ = G['bench'][k]
    rows.append(f'{lab} & {f3(G["stress_enlarged227"]["residual_cef_expected_gated"][k])} & {f3(G["stress_enlarged227"]["cef_expected_not_gated"][k])} & {f3(b_["validity_acc"])} & {f3(b_["residual_cef_contradictory"])} & {f3(b_["mean_margin"])} & {G["system_tau_vs_hard"][k]:.2f}')
w('gate', tab('lcccccc', 'Gate & \\multicolumn{2}{c}{Stress set: \\cef} & \\multicolumn{3}{c}{\\bench\\ (validity suite)} & $\\tau$\\\\\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\n & reversed & controls & acc. & \\cef\\ contr. & margin & ', rows))

# ------------------------------------------------------------------ mapper (main, frozen + revised merged)
ma = pd.read_csv(RC / 'mapper_v2/per_axis.csv'); ma = ma[ma.dataset == 'ALL'].set_index('axis')
mr = pd.read_csv(RM / 'mapper_revised/before_after_comparison.csv').set_index('axis')
rows = []
for ax in ['object', 'action', 'source', 'target', 'location', 'quantity', 'change_state']:
    r = ma.loc[ax]; lo, hi = json.loads(r.P_resolved_CI95)
    ex = f'{r.exact_rate:.2f} ($n{{=}}{int(r.n_mappable_resolved)}$)' if r.n_mappable_resolved > 0 else '--'
    rows.append(f'{ax.replace("_", " ")} & {int(r.n_present)} & {r.P_resolved_given_present:.2f} \\scriptsize[{lo:.2f},{hi:.2f}] & {mr.loc[ax, "P_resolved_revised"]:.2f} & {ex}')
w('mapper', tab('lrccc', 'Axis & $n_{\\text{present}}$ & $P$(resolved$\\mid$present) [95\\% CI] & revised mapper & exact node$^\\dagger$', rows))

# ------------------------------------------------------------------ supplement: dual vs joint, root modes, learning, refagg, cascade
if dj:
    rows = []
    for k, v in dj.items():
        rmono = v['recall_monotonicity']
        rows.append(f'{k.replace("V2-A ", "").replace("V2-B ", "")} & {rmono["violated"]}/{rmono["checked"]} & {"yes" if v["order_invariant"] else "no"} & {f3(v["root_only_cef"])} & {f3(v["bench_ordering"]["valid_acc"])} & {v["tau_vs_V2A"]:.2f}')
    w('dual_joint', tab('lccccc', 'Alignment & recall-monotonicity violations & order-invariant & root-only \\cef & validity acc. & $\\tau$ vs.\\ dual', rows))
rows = []
for k, lab in [('include', 'root counted'), ('legacy', 'root excluded (5 axes)'), ('exclude', 'root excluded (all axes)'), ('unspecified', 'root $=$ unspecified (used)')]:
    v = rm[k]
    rows.append(f'{lab} & {f2(v["R1_root_only"]["cef"])} & {"pass" if v["R2_pass"] else "fail"} & {"pass" if v["R3_pass"] else "fail"} & {f3(v["bench_near_gt_sibling"]["acc"])} & {f3(v["bench_core"]["acc"])} & {f3(v["bench_validity"]["acc"])} & {f3(v["mean_cef_all"])}')
w('root_mode', tab('lccccccc', 'Root treatment & R1 root-only \\cef & R2 & R3 & R4 near$>$sib. & core & validity & mean \\cef', rows))
LGN = [('sft', 'LoRA SFT (reference)'), ('dpo_original', 'DPO, original setup'), ('tcpo_original', 'TCPO, original setup'), ('dpo_stab', 'DPO, stabilised'),
       ('dpo_stab_kl', 'DPO, stabilised + KL'), ('tcpo_true', 'TCPO true severity'), ('tcpo_clipped', 'TCPO clipped severity'),
       ('tcpo_normalized', 'TCPO normalised severity'), ('tcpo_random', 'TCPO random severity'), ('tcpo_uniform', 'TCPO uniform (no severity)'),
       ('tcpo_inverse', 'TCPO inverse severity'), ('tcpo_curriculum', 'TCPO hard-negative curriculum')]
rows = [f'{lab} & {int(LG.loc[g, "count"])} & {LG.loc[g, "mean"]:.3f} $\\pm$ {(0.0 if np.isnan(LG.loc[g, "std"]) else LG.loc[g, "std"]):.3f}' for g, lab in LGN]
w('learning', tab('lcc', 'Variant (from the SFT initialisation) & seeds & \\cef\\ mean $\\pm$ std', rows))
if (R3 / 'refagg_cascade/reference_aggregation.csv').exists():
    ra = pd.read_csv(R3 / 'refagg_cascade/reference_aggregation.csv')
    rows = []
    for p, lab in [('single', 'single reference'), ('mean', 'mean over references (used)'), ('max', 'max over references'), ('consensus', 'consensus graph'), ('consensus_weighted', 'consensus, $q(e)$-weighted')]:
        rows.append(lab + ' & ' + ' & '.join(f3(ra[(ra.system == s) & (ra.protocol == p)].ce_f.iloc[0]) for s in ('sft_seed42', 'chg2cap', 'rscama')))
    w('refagg', tab('lccc', 'Protocol & SFT & Chg2Cap & RSCaMa', rows))
if (R3 / 'refagg_cascade/mapper_cascade.csv').exists():
    mc = pd.read_csv(R3 / 'refagg_cascade/mapper_cascade.csv')
    rows = []
    for c in mc.config.unique():
        x = mc[mc.config == c].set_index('system')
        rows.append(c + ' & ' + ' & '.join(f3(x.loc[s, 'ce_f']) for s in ('qwen', 'sft_seed42', 'rscama')) + f' & {x.unresolved_share.mean():.2f}')
    w('cascade', tab('lcccc', 'Mapper stages & Qwen ZS & SFT & RSCaMa & unresolved', rows))

# ------------------------------------------------------------------ supplement: activated reference pairs
ex = pd.read_csv(R3 / 'gate/activated_reference_pair_examples.csv')
esc = lambda s: str(s).strip().replace('_', '\\_').replace('&', '\\&').replace('%', '\\%')
rows = [f'{esc(r.ref_a)} & {esc(r.ref_b)} & {r.cef_none:.2f}$\\to${r.cef_hard:.2f}' for r in ex.itertuples()]
w('gate_examples', '\\begin{tabular}{@{}p{6.6cm}p{6.6cm}c@{}}\\toprule\nReference A & Reference B (same image) & \\cef\\ no gate$\\to$gate\\\\\\midrule\n' + '\\\\\n'.join(rows) + '\\\\\n\\bottomrule\\end{tabular}\n')

# ------------------------------------------------------------------ error examples under the final evaluator + runtime
from ricce.benchmarks.env import load_taxes, load_weights, make_parser
from ricce.metrics.final import final_scores, FINAL_CONFIG
from ricce.schema import Event
T, W = load_taxes(), load_weights()
P = make_parser()
EE = pd.read_csv(RC / 'error_examples/examples.csv')
out = []
for r in EE.itertuples():
    pe, ge = P.parse(r.prediction, allow_multi_event=True), P.parse(r.reference, allow_multi_event=True)
    s = final_scores(pe, ge, T, W)
    fmt = lambda es: ' + '.join(f'({e.object or "-"}, {e.action or "-"}, {e.location or "-"})' for e in es) if es else '(empty)'
    out.append(dict(system=r.system, image_id=r.image_id, category=r.category, reference=r.reference, prediction=r.prediction,
                    ref_events=fmt(ge), pred_events=fmt(pe), ce_p=s['ce_precision'], ce_r=s['ce_recall'], ce_f=s['ce_f1'],
                    ce_p_previous=r.ce_p, ce_r_previous=r.ce_r))
EO = R3 / 'error_examples'; EO.mkdir(exist_ok=True)
pd.DataFrame(out).to_csv(EO / 'examples_final.csv', index=False)
SN = {'sft_seed42': 'SFT', 'rscama': 'RSCaMa'}
short = lambda s, n=150: esc(s if len(s) <= n else s[:n].rsplit(' ', 1)[0] + ' \\ldots')
rows = [f'{r["category"]} ({SN.get(r["system"], r["system"])}) & {short(r["reference"])} & {short(r["prediction"])} & {esc(r["ref_events"])} $\\to$ {esc(r["pred_events"])} & {r["ce_p"]:.2f}/{r["ce_r"]:.2f}' for r in out]
w('error_examples', '\\begin{tabular}{@{}p{1.9cm}p{3.6cm}p{4.0cm}p{5.0cm}c@{}}\\toprule\nCase (system) & Reference (first) & Prediction & Parsed events (ref $\\to$ pred) & \\cep/\\cer\\\\\\midrule\n'
  + '\\\\\n'.join(rows) + '\\\\\n\\bottomrule\\end{tabular}\n')
items = [json.loads(l) for l in open(ROOT / 'artifacts_cvpr_final/riccebench/items.jsonl')][:5000]
E = lambda L: [Event(**{k: v for k, v in d.items() if k in Event.__dataclass_fields__}) for d in L]
pairs = [(E(x['candidate_event']), E(x['source_event'])) for x in items]
t0 = time.perf_counter()
for _ in range(3):
    for a, b in pairs:
        final_scores(a, b, T, W)
dt = (time.perf_counter() - t0) / 3
RT = R3 / 'runtime'; RT.mkdir(exist_ok=True)
json.dump(dict(n_pairs=len(pairs), seconds_total=dt, seconds_per_1000=1000 * dt / len(pairs), config=FINAL_CONFIG,
               note='scoring only (pre-parsed events), single process, shared server'), open(RT / 'scoring_only.json', 'w'), indent=1)
print('scoring-only s/1000 pairs', 1000 * dt / len(pairs))
