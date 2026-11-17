"""Robustness to simulated acquisition interference, step 4: tables, figure, numbers.

Reads results_cvpr27_robustness/ (scripts/cvpr27_25_robustness_score.py) and
writes CVPR2027_Robustness/:
  tab_interference_main.tex   main-paper table (CE-P / CE-R / CE-F, clean vs. the
                              25-scenario average, all three datasets)
  tab_supp_*.tex              supplement tables (per-scenario CE-F, diagnostics,
                              per-type marginals, severity sweep, string metrics)
  fig_interference_examples.pdf  the five interference types on one LEVIR-CC pair
  numbers.json                every number quoted in the text, with its source
Nothing here computes a score; it only formats saved results.
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results_cvpr27_robustness'
OUT = ROOT / 'CVPR2027_Robustness'; OUT.mkdir(exist_ok=True)
S = pd.read_csv(R / 'per_system.csv')
C = pd.read_csv(R / 'per_condition.csv')
SW = pd.read_csv(R / 'severity_sweep.csv')
PW = pd.read_csv(R / 'pairwise_corrupted.csv')
J = json.load(open(R / 'robustness.json'))
DS = [('LEVIR_CC', 'LEVIR-CC'), ('SECOND_CC', 'SECOND-CC'), ('DUBAI_CC', 'Dubai-CC')]
ORDER = ['swmcc', 'chg2cap', 'rsiccformer', 'promptcc', 'rscama']   # Tab. 5 order
NAME = {'swmcc': 'SWMCC', 'chg2cap': 'Chg2Cap', 'rsiccformer': 'RSICCFormer', 'promptcc': 'PromptCC', 'rscama': 'RSCaMa'}
TYPES = {'F': 'fog', 'G': 'Gaussian noise', 'I': 'impulse noise', 'M': 'motion blur', 'S': 'snow'}
NUM = {}


def f3(x):
    return f'{x:.3f}'


def sgn(x, d=3):
    return ('+' if x >= 0 else '$-$') + f'{abs(x):.{d}f}'


def rows(ds):
    s = S[S.dataset == ds].set_index('system')
    return [(sy, s.loc[sy]) for sy in ORDER if sy in s.index]


def nm(sy, ds):
    return NAME[sy]


# ------------------------------------------------------------------ main table
L = [r'\begin{table}[t]', r'\centering\small\resizebox{\columnwidth}{!}{\begin{tabular}{@{}llccc@{\hspace{10pt}}cccc@{}}\toprule',
     r' & & \multicolumn{3}{c}{Clean} & \multicolumn{4}{c}{Interference (mean of 25 scenarios)}\\',
     r'\cmidrule(lr){3-5}\cmidrule(l){6-9}', r'Data & System & \cep & \cer & \cef & \cep & \cer & \cef & $\Delta$\cef\\\midrule']
for k, (ds, lab) in enumerate(DS):
    rr = rows(ds)
    if not rr:
        continue
    if k:
        L.append(r'\midrule')
    for j, (sy, r) in enumerate(rr):
        first = rf'\multirow{{{len(rr)}}}{{*}}{{{lab}}}' if j == 0 else ''
        L.append(f'{first} & {nm(sy, ds)} & {f3(r.clean_ce_p)} & {f3(r.clean_ce_r)} & {f3(r.clean_ce_f)} & {f3(r.corr_ce_p)} & {f3(r.corr_ce_r)} & {f3(r.corr_ce_f)} & {sgn(r.delta_ce_f)}\\\\')
excl = ((S.delta_ce_f_hi < 0) | (S.delta_ce_f_lo > 0))
NUM['n_cells'] = int(len(S)); NUM['n_cells_ci_excludes_0'] = int(excl.sum())
ci_note = ('all 95\\% CIs exclude 0' if excl.all() else
           f'95\\% CI excludes 0 for {int(excl.sum())} of {len(S)} rows')
L += [r'\bottomrule\end{tabular}}',
      r'\caption{\textbf{Simulated acquisition interference} (Real-CD protocol~\cite{fang2026realcd}; ImageNet-C functions~\cite{hendrycks2019benchmarking,michaelis2019dragon}, severity 3), '
      r'mean over the 25 combinations of five types on $I^{t_1}$ and $I^{t_2}$. Clean: our re-run (\cef\ within '
      + f"{max(abs(v['clean_minus_tab5_ce_f']) for v in J['checks']['clean_vs_tab5_all_cells'].values()):.4f}"
      + r' of \cref{tab:system}). $\Delta$: interference $-$ clean '
      rf'({ci_note}; App.~\xref{{app:interference}}{{Q}}). $^\dagger$In-domain; other SECOND-CC/Dubai-CC rows: zero-shot LEVIR-CC checkpoints.}}',
      r'\label{tab:interference}', r'\end{table}']
(OUT / 'tab_interference_main.tex').write_text('\n'.join(L) + '\n')

# ------------------------------------------------------------------ supplement: per-scenario CE-F (Real-CD Table II layout)
def scen_table(ds_list, label, caption):
    cols = [(ds, sy) for ds in ds_list for sy, _ in rows(ds)]
    hdr = ' & '.join(NAME[sy] for ds, sy in cols)
    L = [r'\begin{table*}[t]', r'\centering\scriptsize\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}l' + 'c' * len(cols) + '@{}}\\toprule']
    if len(ds_list) > 1:
        spans, x = [], 2
        for ds in ds_list:
            n = len(rows(ds)); spans.append((x, x + n - 1, dict(DS)[ds])); x += n
        L.append(' & ' + ' & '.join(rf'\multicolumn{{{b - a + 1}}}{{c}}{{{t}}}' for a, b, t in spans) + r'\\')
        L.append(''.join(rf'\cmidrule(lr){{{a}-{b}}}' for a, b, _ in spans))
    L.append(r'$T_1$--$T_2$ & ' + hdr + r'\\\midrule')
    cc = C.set_index(['dataset', 'system', 'condition'])
    for cond in ['clean'] + [f's3_{a}-{b}' for a in 'FGIMS' for b in 'FGIMS']:
        vals = [cc.loc[(ds, sy, cond), 'ce_f'] for ds, sy in cols]
        L.append(('Clean' if cond == 'clean' else cond[3:]) + ' & ' + ' & '.join(f3(v) for v in vals) + r'\\')
        if cond == 'clean':
            L.append(r'\midrule')
    L.append(r'\midrule')
    s = S.set_index(['dataset', 'system'])
    L.append(r'Mean of 25 & ' + ' & '.join(f3(s.loc[(ds, sy), 'corr_ce_f']) for ds, sy in cols) + r'\\')
    L.append(r'Same type (5) & ' + ' & '.join(f3(s.loc[(ds, sy), 'corr_ce_f_same_type']) for ds, sy in cols) + r'\\')
    L.append(r'Mixed types (20) & ' + ' & '.join(f3(s.loc[(ds, sy), 'corr_ce_f_mixed_type']) for ds, sy in cols) + r'\\')
    L += [r'\bottomrule\end{tabular}', rf'\caption{{{caption}}}', rf'\label{{{label}}}', r'\end{table*}']
    return '\n'.join(L) + '\n'


(OUT / 'tab_supp_scenarios_levir.tex').write_text(scen_table(['LEVIR_CC'], 'tab:int-levir',
    r'\cef\ on LEVIR-CC for each interference scenario (row label: interference on $I^{t_1}$--on $I^{t_2}$; '
    r'F fog, G Gaussian noise, I impulse noise, M motion blur, S snow; severity 3). Mean over five references per image, mean over 1{,}929 images.'))
(OUT / 'tab_supp_scenarios_second_dubai.tex').write_text(scen_table(['SECOND_CC', 'DUBAI_CC'], 'tab:int-second-dubai',
    r'\cef\ on SECOND-CC (1{,}227 pairs) and Dubai-CC (150 pairs) for each interference scenario (labels as in Table~\ref{tab:int-levir}). '
    r'$^\dagger$In-domain checkpoint; the other columns are LEVIR-CC checkpoints applied zero-shot.'))

# ------------------------------------------------------------------ supplement: diagnostics with CIs
L = [r'\begin{table*}[t]', r'\centering\scriptsize\setlength{\tabcolsep}{3.5pt}\begin{tabular}{@{}llccccccccc@{}}\toprule',
     r' & & \multicolumn{3}{c}{Change, interference $-$ clean [95\% CI]} & \multicolumn{2}{c}{Missed change} & \multicolumn{2}{c}{Hallucinated change} & \multicolumn{2}{c}{Gated}\\',
     r'\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(l){10-11}',
     r'Dataset & System & \cep & \cer & \cef & clean & interf. & clean & interf. & clean & interf.\\\midrule']
for ds, lab in DS:
    for sy, r in rows(ds):
        ci = lambda m: f'{sgn(r[f"delta_{m}"])} [{sgn(r[f"delta_{m}_lo"])}, {sgn(r[f"delta_{m}_hi"])}]'
        pc = lambda x: f'{100 * x:.0f}\\%'
        L.append(f'{lab} & {nm(sy, ds)} & {ci("ce_p")} & {ci("ce_r")} & {ci("ce_f")} & {pc(r.clean_missed_change)} & {pc(r.corr_missed_change)} & '
                 f'{pc(r.clean_hallucinated_change)} & {pc(r.corr_hallucinated_change)} & {pc(r.clean_gated)} & {pc(r.corr_gated)}\\\\')
L += [r'\bottomrule\end{tabular}',
      r'\caption{Where the loss under interference comes from. Changes: paired bootstrap over images (10{,}000 resamples) of the per-image mean over the 25 scenarios minus the clean value. '
      r'Missed change: share of changed scenes (fewer than half of the parsed references say no change) captioned as no change. Hallucinated change: share of unchanged scenes captioned with a change event. '
      r'Gated: share of captions on which the contradiction gate acts against at least one reference. Interference columns are means over the 25 scenarios.}',
      r'\label{tab:int-diag}', r'\end{table*}']
(OUT / 'tab_supp_diagnostics.tex').write_text('\n'.join(L) + '\n')

# ------------------------------------------------------------------ supplement: per-type marginals and string metrics
t12_ = pd.read_csv(R / 't1_vs_t2.csv').set_index(['dataset', 'system', 'type'])   # scripts/cvpr27_25b
L = [r'\begin{table*}[t]', r'\centering\scriptsize\setlength{\tabcolsep}{3.5pt}\begin{tabular}{@{}llccccccccccccccc@{}}\toprule',
     r' & & \multicolumn{5}{c}{Type on $I^{t_1}$ (any on $I^{t_2}$)} & \multicolumn{5}{c}{Type on $I^{t_2}$ (any on $I^{t_1}$)} & \multicolumn{5}{c}{Difference, $I^{t_1}$ $-$ $I^{t_2}$}\\',
     r'\cmidrule(lr){3-7}\cmidrule(lr){8-12}\cmidrule(l){13-17}', r'Data & System & F & G & I & M & S & F & G & I & M & S & F & G & I & M & S\\\midrule']
for ds, lab in DS:
    for sy, r in rows(ds):
        dd = [t12_.loc[(ds, sy, k)] for k in 'FGIMS']
        L.append(f'{lab} & {nm(sy, ds)} & ' + ' & '.join(f3(r[f'ce_f_t1_{k}']) for k in 'FGIMS') + ' & ' + ' & '.join(f3(r[f'ce_f_t2_{k}']) for k in 'FGIMS')
                 + ' & ' + ' & '.join(sgn(x['mean']) + ('$^*$' if x.ci_excludes_0 else '') for x in dd) + r'\\')
L += [r'\bottomrule\end{tabular}', r'\caption{\cef\ by interference type, averaged over the five scenarios in which that type affects $I^{t_1}$ (left) or $I^{t_2}$ (middle), '
      r'and their difference (right; the scenario with the type on both images appears in both means and cancels). $^*$95\% CI of the difference excludes 0 (paired bootstrap over images, 10{,}000 resamples).}',
      r'\label{tab:int-types}', r'\end{table*}']
(OUT / 'tab_supp_types.tex').write_text('\n'.join(L) + '\n')

L = [r'\begin{table}[t]', r'\centering\scriptsize\setlength{\tabcolsep}{3pt}\resizebox{\columnwidth}{!}{\begin{tabular}{@{}llcccccc@{}}\toprule',
     r' & & \multicolumn{2}{c}{BLEU-4} & \multicolumn{2}{c}{CIDEr-D} & \multicolumn{2}{c}{\cef}\\',
     r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}', r'Data & System & clean & interf. & clean & interf. & clean & interf.\\\midrule']
for ds, lab in DS:
    for sy, r in rows(ds):
        L.append(f'{lab} & {nm(sy, ds)} & {f3(r.clean_bleu4)} & {f3(r.corr_bleu4)} & {f3(r.clean_cider_d)} & {f3(r.corr_cider_d)} & {f3(r.clean_ce_f)} & {f3(r.corr_ce_f)}\\\\')
L += [r'\bottomrule\end{tabular}}', r'\caption{String metrics on the same captions (clean and mean of the 25 interference scenarios); BLEU-4 and CIDEr-D as in Table~\xref{tab:system}{5}.}',
      r'\label{tab:int-string}', r'\end{table}']
(OUT / 'tab_supp_string.tex').write_text('\n'.join(L) + '\n')

# ------------------------------------------------------------------ supplement: severity sweep
sw = SW.set_index(['system', 'severity'])
L = [r'\begin{table}[t]', r'\centering\scriptsize\setlength{\tabcolsep}{3pt}\resizebox{\columnwidth}{!}{\begin{tabular}{@{}lccccccc@{}}\toprule',
     r' & & \multicolumn{3}{c}{\cef\ at severity} & \multicolumn{3}{c}{Missed change at severity}\\',
     r'\cmidrule(lr){3-5}\cmidrule(l){6-8}', r'System & Clean & 1 & 3 & 5 & 1 & 3 & 5\\\midrule']
cl = S[S.dataset == 'LEVIR_CC'].set_index('system')
for sy in ORDER:
    if (sy, 1) not in sw.index:
        continue
    L.append(f'{NAME[sy]} & {f3(cl.loc[sy, "clean_ce_f"])} & ' + ' & '.join(f3(sw.loc[(sy, v), 'ce_f']) for v in (1, 3, 5)) + ' & '
             + ' & '.join(f'{100 * sw.loc[(sy, v), "missed_change"]:.0f}\\%' for v in (1, 3, 5)) + r'\\')
L += [r'\bottomrule\end{tabular}}', r'\caption{Severity sweep on LEVIR-CC: the same interference type on both images (F--F, G--G, I--I, M--M, S--S), mean over the five types; '
      r'severity 3 is the setting of Table~\xref{tab:interference}{6}. Clean missed-change rates are in Table~\ref{tab:int-diag}.}', r'\label{tab:int-sweep}', r'\end{table}']
(OUT / 'tab_supp_sweep.tex').write_text('\n'.join(L) + '\n')
print('tables written', flush=True)


# ------------------------------------------------------------------ example figure (supplement)
# Selection rule, fixed before any caption of these scenarios was looked at: the first LEVIR-CC
# test pair, in image-ID order, whose parsed references describe a change (majority) and on which
# RSCaMa's clean caption reaches CE-F >= 0.9; the figure shows whatever RSCaMa outputs for the
# five same-type scenarios (F-F, G-G, I-I, M-M, S-S) of that pair.
import shutil, subprocess
D = pd.read_csv(R / 'per_image_scores.csv.gz')
rs = D[(D.dataset == 'LEVIR_CC') & (D.system == 'rscama')].set_index(['condition', 'image_id'])
cl = rs.loc['clean']
cand = sorted(cl[(cl.ref_changed) & (cl.ce_f >= 0.9)].index)
iid = cand[0]
man = {json.loads(l)['image_id']: json.loads(l) for l in open(ROOT / 'artifacts_cvpr27_robustness' / 'manifests' / 'LEVIR_CC' / 'clean.jsonl')}
row = man[iid]
FD = OUT / 'src' / 'fig_interference_examples'; (FD / 'img').mkdir(parents=True, exist_ok=True)
shutil.copy(row['t1_path'], FD / 'img' / 'clean_t1.png'); shutil.copy(row['t2_path'], FD / 'img' / 'clean_t2.png')
for k in 'FGIMS':
    for t in ('t1', 't2'):
        shutil.copy(ROOT / 'artifacts_cvpr27_robustness' / 'images' / 'LEVIR_CC' / 's3' / k / t / f'{iid}.png', FD / 'img' / f'{k}_{t}.png')
esc = lambda s: s.replace('\\', '').replace('&', '\\&').replace('%', '\\%').replace('$', '\\$').replace('#', '\\#').replace('_', '\\_')
cols = [('clean', 'Clean', 'clean')] + [(f's3_{k}-{k}', TYPES[k].capitalize(), k) for k in 'FGIMS']
ex = dict(image_id=iid, rule='first LEVIR-CC test pair (ID order) with changed references and RSCaMa clean CE-F >= 0.9',
          n_candidates=len(cand), reference=row['references'][0].strip(), cells={})
W_ = 2.75
tz = [r'\documentclass[border=1pt]{standalone}', r'\usepackage[T1]{fontenc}', r'\usepackage{times}', r'\usepackage{tikz}', r'\usepackage{graphicx}',
      r'\begin{document}\begin{tikzpicture}[x=1cm,y=1cm,font=\fontsize{7}{8.4}\selectfont]']
for j, (cond, title, k) in enumerate(cols):
    x = j * (W_ + 0.12)
    r_ = rs.loc[(cond, iid)]
    ex['cells'][cond] = dict(caption=r_.prediction, ce_p=float(r_.ce_p), ce_r=float(r_.ce_r), ce_f=float(r_.ce_f), bleu4=float(r_.bleu4))
    tz.append(rf'\node[anchor=south] at ({x + W_ / 2:.3f},{2 * W_ + 0.18:.3f}) {{\textbf{{{title}}}{" on both" if k != "clean" else ""}}};')
    for i, t in enumerate(('t1', 't2')):
        tz.append(rf'\node[anchor=south west,inner sep=0] at ({x:.3f},{(1 - i) * (W_ + 0.08) + 0.0:.3f}) {{\includegraphics[width={W_}cm,height={W_}cm]{{img/{k}_{t}.png}}}};')
    tz.append(rf'\node[anchor=north west,text width={W_ - 0.05}cm,align=left,inner sep=1pt] at ({x:.3f},-0.06) '
              rf"{{\itshape ``{esc(r_.prediction)}''\\[1pt]\upshape \cef\,{r_.ce_f:.2f}\quad BLEU-4\,{r_.bleu4:.2f}}};")
tz.append(rf'\node[anchor=east,rotate=90] at (-0.18,{2 * W_ + 0.08:.3f}) {{$I^{{t_1}}$}};')
tz.append(rf'\node[anchor=east,rotate=90] at (-0.18,{W_:.3f}) {{$I^{{t_2}}$}};')
tz += [r'\end{tikzpicture}\end{document}']
src = '\n'.join(tz).replace(r'\cef', r'CE-F')
(FD / 'fig_interference_examples.tex').write_text(src + '\n')
r_ = subprocess.run([os.environ.get('TECTONIC') or shutil.which('tectonic') or os.path.expanduser('~/tectonic_env/bin/tectonic'), 'fig_interference_examples.tex'], cwd=FD, capture_output=True, text=True)
assert r_.returncode == 0, r_.stderr[-2000:]
shutil.copy(FD / 'fig_interference_examples.pdf', OUT / 'fig_interference_examples.pdf')
NUM['example'] = ex
print('example figure:', iid, 'from', len(cand), 'candidates', flush=True)
json.dump(NUM, open(OUT / 'numbers.json', 'w'), indent=1, default=float)


# ------------------------------------------------------------------ text: CVPR2027_Robustness/templates/*.in -> filled files
# Every number in the text is a @@TOKEN@@ computed here from the saved results; a
# template with an unfilled token fails the run.
import re
TPL = OUT / 'templates'
num = lambda n: f'{int(n):,}'.replace(',', '{,}')
TOK = {}
CLOG = json.load(open(ROOT / 'artifacts_cvpr27_robustness' / 'corruption_log.json'))
for ds, key in (('LEVIR_CC', 'N_LEVIR'), ('SECOND_CC', 'N_SECOND'), ('DUBAI_CC', 'N_DUBAI')):
    TOK[key] = num(CLOG['datasets'][ds]['s3']['n_pairs'])
TOK['N_IMG_S3'] = num(sum(v['s3']['n_images'] for v in CLOG['datasets'].values()))
idn, t5 = J['checks']['clean_identical_to_frozen'], J['checks']['clean_vs_tab5_all_cells']
key_order = lambda k: ([d for d, _ in DS].index(k.split('|')[0]), ORDER.index(k.split('|')[1]))
same = [k for k in sorted(idn, key=key_order) if idn[k]['identical'] == idn[k]['n']]
diff = [k for k in sorted(idn, key=key_order) if k not in same]
lab = lambda k: f"{NAME[k.split('|')[1]]} on {dict(DS)[k.split('|')[0]]}"
s_ = (f'For {len(same)} of the {len(idn)} system--dataset cells the clean captions are identical to the frozen predictions '
      r'and every score equals Table~\xref{tab:system}{5}.')
if diff:
    s_ += (f' In the other {len(diff)}, a few clean captions differ, with code, checkpoint and decoding unchanged ('
           + '; '.join(f"{lab(k)}: {num(idn[k]['n'] - idn[k]['identical'])} of {num(idn[k]['n'])}" for k in diff)
           + r'); their clean \cef\ differs from Table~\xref{tab:system}{5} by at most '
           + f"{max(abs(t5[k]['clean_minus_tab5_ce_f']) for k in diff):.4f}.")
TOK['IDENTITY_SENTENCE'] = s_
NUM['identity'] = dict(same=same, diff={k: idn[k] for k in diff}, tab5_diff={k: t5[k] for k in diff})

# ------------------------------------------------------------------ supplement: same vs. mixed interference (scripts/cvpr27_25b)
AV = pd.read_csv(R / 'same_vs_mixed.csv'); T12 = pd.read_csv(R / 't1_vs_t2.csv')
av = AV.set_index(['dataset', 'system', 'contrast'])
pc = lambda x: f'{100 * x:.0f}\\%'
L = [r'\begin{table*}[t]', r'\centering\scriptsize\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}llcccccc@{}}\toprule',
     r' & & \multicolumn{3}{c}{Hallucinated change, unchanged scenes} & \multicolumn{3}{c}{Missed change, changed scenes}\\',
     r'\cmidrule(lr){3-5}\cmidrule(l){6-8}',
     r'Dataset & System & same type & mixed types & mixed $-$ same [95\% CI] & same type & mixed types & same $-$ mixed [95\% CI]\\\midrule']
for ds, lab_ in DS:
    for sy, _ in rows(ds):
        h, m = av.loc[(ds, sy, 'hallucinated_change_unchanged')], av.loc[(ds, sy, 'missed_change_changed')]
        L.append(f'{lab_} & {nm(sy, ds)} & {pc(h.same)} & {pc(h.mixed)} & {sgn(h["mean"], 2)} [{sgn(h.lo, 2)}, {sgn(h.hi, 2)}] & '
                 f'{pc(m.same)} & {pc(m.mixed)} & {sgn(m["mean"], 2)} [{sgn(m.lo, 2)}, {sgn(m.hi, 2)}]\\\\')
L += [r'\bottomrule\end{tabular}',
      r'\caption{Same versus different interference on the two images (severity 3). Same type: mean over the five scenarios F--F, \dots, S--S; '
      r'mixed types: mean over the other 20. Per image, the rate over the scenarios of each group; differences and 95\% CIs from a paired bootstrap over images '
      r'(10{,}000 resamples), on the unchanged (left) or changed (right) scenes of each dataset. Definitions as in Table~\ref{tab:int-diag}.}',
      r'\label{tab:int-samemixed}', r'\end{table*}']
(OUT / 'tab_supp_same_mixed.tex').write_text('\n'.join(L) + '\n')

# ------------------------------------------------------------------ result tokens; every qualitative word of the templates is guarded by an assert
pc0 = lambda x: f'{100 * x:.0f}'
s = S.set_index(['dataset', 'system'])
cell = lambda ds, sy: f'{nm(sy, ds)} on {dict(DS)[ds]}'
rng_ = lambda xs: f'{pc0(min(xs))}--{pc0(max(xs))}\\%'
assert (S.delta_ce_f < 0).all()                                      # "every row loses CE-F"
TOK['N_ROWS'] = str(len(S)); TOK['N_CI_EXCL'] = str(int(excl.sum()))
TOK['CI_INCL_ROWS'] = ', '.join(cell(r.dataset, r.system) for r in S[~excl].itertuples()) or 'none'
lv = S[S.dataset == 'LEVIR_CC'].sort_values('retained_ce_f')
TOK.update(LEVIR_RET_BEST_SYS=NAME[lv.iloc[-1].system], LEVIR_RET_BEST=pc0(lv.iloc[-1].retained_ce_f),
           LEVIR_RET_WORST_SYS=NAME[lv.iloc[0].system], LEVIR_RET_WORST=pc0(lv.iloc[0].retained_ce_f),
           LEVIR_D_BEST=sgn(lv.iloc[-1].delta_ce_f), LEVIR_D_WORST=sgn(lv.iloc[0].delta_ce_f))
assert lv.delta_ce_f.idxmax() == lv.index[-1] and lv.delta_ce_f.idxmin() == lv.index[0]   # same extremes in Delta and retention
pl = PW[PW.dataset == 'LEVIR_CC']
assert (pl.p_holm < 0.05).all()                                      # "all LEVIR-CC pairs differ"
TOK.update(LEVIR_PW_N=str(len(pl)), LEVIR_PW_PMAX=f'{pl.p_holm.max():.4f}')
TOK.update(LEVIR_RANK_RHO=f"{J['LEVIR_CC_rank_clean_vs_corrupted_spearman']:.2f}")
assert abs(J['LEVIR_CC_corrupted_rank_cef_vs_bleu4_spearman'] - 1) < 1e-9 and abs(J['LEVIR_CC_corrupted_rank_cef_vs_cider_d_spearman'] - 1) < 1e-9  # "identically"
h = AV[AV.contrast == 'hallucinated_change_unchanged']; m = AV[AV.contrast == 'missed_change_changed']
hx, hn, mx_ = h.loc[h['mean'].idxmax()], h.loc[h['mean'].idxmin()], m.loc[m['mean'].idxmax()]
TOK.update(HALL_N=str(int(h.consistent.sum())), HALL_ROWS=str(len(h)), HALL_MAX=f'{hx["mean"]:.2f}', HALL_MAX_CELL=cell(hx.dataset, hx.system),
           HALL_MIN=f'{hn["mean"]:.2f}', HALL_MIN_CELL=cell(hn.dataset, hn.system),
           MISS_N=str(int(m.consistent.sum())), MISS_ROWS=str(len(m)), MISS_MAX=f'{mx_["mean"]:.2f}', MISS_MAX_CELL=cell(mx_.dataset, mx_.system),
           MISS_NOT=', '.join(cell(r.dataset, r.system) for r in m[~m.consistent].itertuples()) or 'none')
assert (h['mean'] > 0).all() and (m['mean'] > 0).all()              # direction holds in every row
c_ = av.loc[('LEVIR_CC', 'chg2cap', 'hallucinated_change_unchanged')]
TOK.update(HALL_CHG_SAME=pc0(c_.same), HALL_CHG_MIXED=pc0(c_.mixed))
d = s.loc[('DUBAI_CC', 'swmcc')]
assert d.delta_ce_f_lo < 0 < d.delta_ce_f_hi                         # "CI includes 0"
TOK.update(DSW_MISS_CLEAN=pc0(d.clean_missed_change), DSW_MISS_CORR=pc0(d.corr_missed_change),
           DSW_HALL_CLEAN=pc0(d.clean_hallucinated_change), DSW_HALL_CORR=pc0(d.corr_hallucinated_change),
           DSW_DCTF=sgn(d.delta_ce_f), DSW_DCTF_LO=sgn(d.delta_ce_f_lo), DSW_DCTF_HI=sgn(d.delta_ce_f_hi),
           DSW_B4_CLEAN=f3(d.clean_bleu4), DSW_B4_CORR=f3(d.corr_bleu4), DSW_B4_D=sgn(d.corr_bleu4 - d.clean_bleu4),
           DSW_CH_CLEAN=f3(d.clean_ce_f_changed), DSW_CH_CORR=f3(d.corr_ce_f_changed),
           DSW_UN_CLEAN=f3(d.clean_ce_f_unchanged), DSW_UN_CORR=f3(d.corr_ce_f_unchanged))
tr = S[S.dataset != 'LEVIR_CC']
assert (tr.corr_missed_change < tr.clean_missed_change).all() and (tr.corr_hallucinated_change > tr.clean_hallucinated_change).all()
TOK.update(TR_N=str(len(tr)), TR_MISS_CLEAN=rng_(tr.clean_missed_change), TR_MISS_CORR=rng_(tr.corr_missed_change),
           TR_HALL_CLEAN=rng_(tr.clean_hallucinated_change), TR_HALL_CORR=rng_(tr.corr_hallucinated_change),
           TR_D=f'{sgn(tr.delta_ce_f.max())} to {sgn(tr.delta_ce_f.min())}')
TOK.update(T12_N=str(int(T12.ci_excludes_0.sum())), T12_TOTAL=str(len(T12)))
tx = T12.loc[T12['mean'].abs().idxmax()]
TOK.update(T12_MAX_CELL=cell(tx.dataset, tx.system), T12_MAX_TYPE=TYPES[tx.type], T12_MAX_T1=f3(tx.ce_f_on_t1), T12_MAX_T2=f3(tx.ce_f_on_t2))
t12 = T12.set_index(['dataset', 'system', 'type'])
fa, fb = t12.loc[('LEVIR_CC', 'chg2cap', 'F')], t12.loc[('LEVIR_CC', 'swmcc', 'F')]
assert fa['mean'] < 0 < fb['mean'] and fa.ci_excludes_0 and fb.ci_excludes_0   # "fog hurts Chg2Cap more on T1, SWMCC more on T2"
TOK.update(T12_FA_T1=f3(fa.ce_f_on_t1), T12_FA_T2=f3(fa.ce_f_on_t2), T12_FB_T1=f3(fb.ce_f_on_t1), T12_FB_T2=f3(fb.ce_f_on_t2))
swp = SW.pivot(index='system', columns='severity')
mono_f = [sy for sy in swp.index if swp.loc[sy, ('ce_f', 1)] > swp.loc[sy, ('ce_f', 3)] > swp.loc[sy, ('ce_f', 5)]]
mono_m = [sy for sy in swp.index if swp.loc[sy, ('missed_change', 1)] < swp.loc[sy, ('missed_change', 3)] < swp.loc[sy, ('missed_change', 5)]]
assert len(mono_f) == len(swp) and len(swp) - len(mono_m) == 1       # "all systems" / "all but one"
nm_ = [sy for sy in swp.index if sy not in mono_m][0]
TOK.update(SW_N=str(len(swp)), SW_MISS_EXC=NAME[nm_], SW_MISS_EXC_1=pc0(swp.loc[nm_, ('missed_change', 1)]),
           SW_MISS_EXC_CLEAN=pc0(s.loc[('LEVIR_CC', nm_)].clean_missed_change))
TOK.update(EX_ID=str(ex['image_id']).replace('_', '\\_'), EX_NCAND=num(ex['n_candidates']), EX_REF=esc(ex['reference']))
assert TOK['HALL_N'] == TOK['HALL_ROWS']                            # main text: "in all rows"


def fill(name, extra=None):
    s = (TPL / f'{name}.in').read_text()
    s = re.sub(r'@@TABLE:([a-z_]+)@@', lambda m: (OUT / f'{m.group(1)}.tex').read_text().strip(), s)
    for k, v in {**TOK, **(extra or {})}.items():
        s = s.replace(f'@@{k}@@', v)
    left = re.findall(r'@@[A-Z0-9_:a-z]+@@', s)
    assert not left, (name, left)
    (OUT / name).write_text(s)
    return s


if (TPL / 'app_interference_results.tex.in').exists():
    TOK['RESULTS'] = fill('app_interference_results.tex').strip()
    for name in ('app_interference_supp.tex', 'sec_interference_main.tex', 'main_text_edits.txt'):
        fill(name)
    print('text filled', flush=True)
else:
    print('results template missing: text not filled', flush=True)
NUM['tokens'] = TOK
json.dump(NUM, open(OUT / 'numbers.json', 'w'), indent=1, default=float)
