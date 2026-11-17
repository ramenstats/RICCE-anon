"""Motivation (teaser) and method-overview figures for the CVPR 2027 paper.

Every number drawn in either figure is read from a saved result file or recomputed here with the
final evaluator and checked against the saved value; figure_values.json lists each number with
its source. Nothing is typed in by hand.

  Fig. A (motivation, full width):
    (a) human-study pair P0078 (LEVIR-CC train_000700): two captions of the same image pair,
        judged side by side by all three annotators; per-caption scores of BLEU-4, CIDEr-D, CSM-4,
        BERTScore, the LLM judge and CE-P/CE-R/CE-F against the reference the study used.
        Selection: among the 380 study pairs, those where the annotators' majority, CE-F and all
        of BLEU-4/CIDEr-D/BERTScore express a preference and the three string/embedding metrics
        disagree with the majority (119 pairs), restricted to pairs with an action-reversal caption
        that all three annotators rejected: 4 pairs (P0104, P0078, P0214, P0110). The choice among
        them is ours, made after seeing them: P0078 is the one whose reference and both captions
        each parse to a single event with the same fields except the edited one (P0104: the parser
        keeps "a road" as a second event in caption B but not in the reference; P0214: it maps the
        reference's object to "bare surface"; P0110 has two events and is the method figure's
        example). In all three P0078 captions the parser also reads "at right angles" as location
        "right", identically, so no score difference comes from it. The example is an
        illustration; panels (b) and (c) give the aggregate evidence.
    (b) RICCE-Bench semantic-validity suite (n = 16,538): share of correct-vs-contradictory
        comparisons each string/embedding metric ranks wrongly (ties count as wrong) = 1 - Tab. 1.
    (c) Spearman correlation with the annotators' mean overall rating (95% cluster-bootstrap CI),
        all items and real system outputs (Tab. 2).
  Fig. B (method, full width): the pipeline on human-study pair P0110 (LEVIR-CC train_000811),
    reference and the caption the annotators preferred (3/3): parsed events, per-axis hP/hR,
    the S_P and S_R matrices and their two optimal assignments; the gate panel uses P0078's
    reversed caption from Fig. A.

Outputs (CVPR2027_Figures/): fig_motivation.{pdf,png}, fig_method.{pdf,png}, src/ (standalone
LaTeX sources + images, compile on Overleaf or locally), figure_values.json, README.md.
"""
import glob, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.benchmarks.env import load_taxes, load_weights
from ricce.evaluation.parse_store import ParseStore
from ricce.evaluation.human_analysis import in_study
from ricce.metrics.final import final_scores, gate_activations
from ricce.metrics.event_v2 import event_field_scores_v2, aggregate_pr, event_similarity_pr_v2, _score_matrix
from scipy.optimize import linear_sum_assignment

S = ROOT / 'submission_cvpr27_new_version'
H = ROOT / 'artifacts_cvpr_final' / 'human_eval' / 'image_grounded'
OUT = ROOT / 'CVPR2027_Figures'
SRC = OUT / 'src'
(SRC / 'img').mkdir(parents=True, exist_ok=True)
TECTONIC = os.environ.get('TECTONIC') or shutil.which('tectonic') or os.path.expanduser('~/tectonic_env/bin/tectonic')
T, W = load_taxes(), load_weights()
VAL = {}


def put(k, v, src):
    VAL[k] = dict(value=v, source=src)
    return v


f2 = lambda x: f'{x:.2f}'
f3 = lambda x: f'{x:.3f}'

# ------------------------------------------------------------------ data
TM = pd.read_csv(S / '07_BASELINES' / 'human_captions_text_metrics.csv').set_index('job')
CS = pd.read_csv(S / '07_BASELINES' / 'human_captions_csm.csv').set_index('job')
KEY = {k['item_id']: k for k in map(json.loads, open(H / 'hidden' / 'answer_key_final_evaluator.jsonl')) if in_study(k)}
PAIRS = {json.loads(l)['pair_id']: json.loads(l) for l in open(H / 'pairwise_items.jsonl')}
ANN = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(H / 'collected' / '*.csv')))])
VOTES = ANN[ANN.task == 'B'].set_index(['item_id', 'annotator_id']).preference
PF = pd.read_csv(S / '03_HUMAN_REANALYSIS' / 'pairwise_frame.csv').set_index('pair_id')
store = ParseStore(ROOT / 'results_v3' / 'human_eval' / 'key_rescoring' / 'parse_cache_human_eval.json')
ev = lambda t: store.events(t, 'multi')

# selection rule of the motivating example (documented in the module docstring)
q = PF[PF.human.isin(['A', 'B']) & (PF.ce_f1 == PF.human)]
for m in ('bleu4', 'cider_d', 'bertscore_f'):
    q = q[q[m].isin(['A', 'B']) & (q[m] != q.human)]
put('SEL_N_DISAGREE', len(q), 'pairwise_frame.csv: majority, CE-F and BLEU-4/CIDEr-D/BERTScore all decided; the three disagree with the majority')
cand = []
for pid, r in q.iterrows():
    k = KEY[pid]; v = [VOTES.get((pid, a)) for a in ('ann1', 'ann2', 'ann3')]
    rev = 'a' if k['perturbation_type_a'] == 'P4_action_reversal' else 'b' if k['perturbation_type_b'] == 'P4_action_reversal' else None
    if rev is None or len(set(v)) != 1 or v[0] == rev.upper():
        continue
    other = 'b' if rev == 'a' else 'a'
    cand.append((k[f'metrics_{rev}']['bleu4'] - k[f'metrics_{other}']['bleu4'], pid))
cand.sort(reverse=True)
put('SEL_CANDIDATES', [p for _, p in cand], 'pairs passing the rule, by BLEU-4 margin of the reversal')
assert sorted(p for _, p in cand) == ['P0078', 'P0104', 'P0110', 'P0214'], cand
put('SEL_CHOICE', 'P0078 (motivation) and P0110 (method), chosen by the authors among the four; reasons in the module docstring', 'author choice')


def pair_values(pid):
    """Per-side values of one study pair, with the CE scores recomputed and checked."""
    k, it = KEY[pid], PAIRS[pid]
    refs = json.loads(TM.loc[pid + 'a', 'references'])
    assert refs == json.loads(TM.loc[pid + 'b', 'references']) and len(refs) == 1
    out = dict(pid=pid, dataset=it['dataset'], image_id=it['image_id'], ref=refs[0].strip(), ref_raw=refs[0], t1=it['t1_path'], t2=it['t2_path'],
               votes={a: VOTES.get((pid, a)) for a in ('ann1', 'ann2', 'ann3')})
    for s in 'ab':
        c = it[f'candidate_{s}']
        assert TM.loc[pid + s, 'candidate'] == c
        sc = final_scores(ev(c), ev(refs[0]), T, W)
        for m, mk in (('ce_p', 'ce_precision'), ('ce_r', 'ce_recall'), ('ce_f1', 'ce_f1')):
            assert abs(sc[mk] - k[f'metrics_{s}'][m]) < 1e-9, (pid, s, m)
        out[s] = dict(caption=c.strip(), bleu4=k[f'metrics_{s}']['bleu4'], cider_d=TM.loc[pid + s, 'cider_d'],
                      bertscore=TM.loc[pid + s, 'bertscore_f'], llm=TM.loc[pid + s, 'llm_judge'], csm4=CS.loc[pid + s, 'csm4'],
                      ce_p=sc['ce_precision'], ce_r=sc['ce_recall'], ce_f=sc['ce_f1'], ptype=k[f'perturbation_type_{s}'])
    return out


SRCP = ('pairwise_items.jsonl (captions, images); human_captions_text_metrics.csv (CIDEr-D, BERTScore, LLM judge, reference); '
        'human_captions_csm.csv (CSM-4); answer_key_final_evaluator.jsonl (BLEU-4); CE recomputed with final_scores and equal to the key; '
        'collected/*.csv (votes)')
M = pair_values('P0078')
assert M['a']['ptype'] == 'P4_action_reversal' and all(v == 'B' for v in M['votes'].values())
for s, lab in (('a', 'REV'), ('b', 'OK')):
    for m in ('bleu4', 'cider_d', 'bertscore', 'llm', 'csm4', 'ce_p', 'ce_r', 'ce_f'):
        put(f'M_{lab}_{m}', M[s][m], SRCP)
    put(f'M_{lab}_caption', M[s]['caption'], SRCP)
put('M_REF', M['ref'], SRCP); put('M_VOTES', M['votes'], 'collected/annotations_ann{1,2,3}_2026-10-29.csv, task B, P0078')
# the gate panel of Fig. B: ungated similarity of the reversed caption's event vs the reference event
pR, gR = ev(M['a']['caption']), ev(M['ref_raw'])
assert len(pR) == len(gR) == len(ev(M['b']['caption'])) == 1
assert pR[0].location == gR[0].location == ev(M['b']['caption'])[0].location == 'right'   # "at right angles", see docstring
put('M_EVENTS', {k: [evd_ for evd_ in [{f: getattr(e, f) for f in ('object', 'action', 'source', 'target', 'location', 'quantity', 'exact_count', 'change_state')}
                                        for e in ev(c)]] for k, c in (('ref', M['ref_raw']), ('A', M['a']['caption']), ('B', M['b']['caption']))},
    'parse_cache_human_eval.json (multi-event parse)')
ung = event_similarity_pr_v2(pR[0], gR[0], T, W, gate_mode='none', root_mode='unspecified')
gat = event_similarity_pr_v2(pR[0], gR[0], T, W, gate_mode='hard', root_mode='unspecified', gate_object_match=True)
assert gat == (0.0, 0.0) and gate_activations(pR, gR, T, W) == [(0, 0, 'action_family')]
assert pR[0].object == gR[0].object == 'building' and (pR[0].action, gR[0].action) == ('demolished', 'constructed')
put('G_UNGATED', list(ung), 'event_similarity_pr_v2(gate_mode=none) on the P0078 reversed caption vs its reference')

# ------------------------------------------------------------------ method example P0110
X = pair_values('P0110')
assert all(v == 'A' for v in X['votes'].values()) and X['a']['ptype'] == 'P2_distant_ancestor'
P_, G_ = ev(X['a']['caption']), ev(X['ref_raw'])
assert len(P_) == len(G_) == 2
SP, SR = _score_matrix(P_, G_, T, W, 'hard', 0.1, True, 'unspecified', True)
rP, cP = linear_sum_assignment(-SP); rR, cR = linear_sum_assignment(-SR)
assert list(cP) == [0, 1] and list(cR) == [0, 1]                       # both optimal assignments are the diagonal
cep, cer = SP[rP, cP].sum() / 2, SR[rR, cR].sum() / 2
assert abs(cep - X['a']['ce_p']) < 1e-12 and abs(cer - X['a']['ce_r']) < 1e-12
fs = event_field_scores_v2(P_[0], G_[0], T, True, 'unspecified')
sp0, sr0 = aggregate_pr(fs, W)
assert abs(sp0 - SP[0, 0]) < 1e-12 and abs(sr0 - SR[0, 0]) < 1e-12
anc = lambda ax, n: list(T[ax].ancestors(n))[1:]                     # root excluded
assert T['object'].children['built_environment'] == ['building', 'transportation'] and 'parking_area' in T['object'].children['transportation']
assert T['action'].children['change'][:2] == ['appearance', 'disappearance']
assert anc('object', 'parking_area') == ['built_environment', 'transportation', 'parking_area']
assert anc('object', 'built_environment') == ['built_environment'] and anc('object', 'building') == ['built_environment', 'building']
assert anc('action', 'constructed')[-2:] == ['appearance', 'constructed'] and anc('action', 'demolished')[-2:] == ['disappearance', 'demolished']
SRCX = SRCP.replace('P0078', 'P0110') + '; events from parse_cache_human_eval.json; matrices from event_v2._score_matrix (final configuration)'
put('X_REF', X['ref'], SRCX); put('X_CAND', X['a']['caption'], SRCX); put('X_VOTES', X['votes'], 'collected sheets, task B, P0110')
put('X_SP', SP.tolist(), SRCX); put('X_SR', SR.tolist(), SRCX)
put('X_FIELDS_p1g1', {k: v for k, v in fs.items()}, 'event_field_scores_v2(p1, g1)'); put('X_SPSR_p1g1', [sp0, sr0], 'aggregate_pr')
put('X_CE', [X['a']['ce_p'], X['a']['ce_r'], X['a']['ce_f']], SRCX)
put('X_weights', W, 'load_weights()')
evd = lambda e: {k: getattr(e, k) for k in ('object', 'action', 'source', 'target', 'location', 'quantity', 'exact_count', 'change_state')}
put('X_EVENTS', dict(g=[evd(e) for e in G_], p=[evd(e) for e in P_]), 'parse_cache_human_eval.json (multi-event parse)')
for e in G_ + P_:
    assert e.source is None and e.location is None                    # the figure lists o, a, t, q, (c), z only

# ------------------------------------------------------------------ aggregate panels
sv = pd.read_csv(ROOT / 'results_cvpr_final' / 'ordering' / 'semantic_validity_ordering.csv').set_index('metric')
bc = json.load(open(S / '07_BASELINES' / 'bertscore_canonical' / 'summary.json'))['semantic_validity']
assert int(sv.loc['bleu4', 'n_constraints']) == bc['n_constraints'] == 16538
FAIL = [('METEOR', 1 - sv.loc['meteor', 'strict']), ('ROUGE-L', 1 - sv.loc['rougeL', 'strict']), ('BLEU-4', 1 - sv.loc['bleu4', 'strict']),
        ('CIDEr-D', 1 - sv.loc['cider_d_item', 'strict']), ('BERTScore', 1 - bc['strict']), ('Sentence-BERT', 1 - sv.loc['sbert', 'strict'])]
FAIL.sort(key=lambda x: -x[1])
put('B_FAIL', FAIL, 'results_cvpr_final/ordering/semantic_validity_ordering.csv (strict); canonical BERTScore: 07_BASELINES/bertscore_canonical/summary.json')
HM = pd.read_csv(S / '03_HUMAN_REANALYSIS' / 'human_main_metrics.csv')
HUM_METRICS = [('ce_f1', r'\textbf{CE-F (ours)}'), ('llm_judge', 'LLM judge'), ('sbert', 'Sentence-BERT'), ('bleu4', 'BLEU-4'),
               ('bertscore_f', 'BERTScore'), ('cider_d', 'CIDEr-D'), ('csm4', 'CSM-4'), ('spice_f', 'SPICE')]
hv = {}
for m, _ in HUM_METRICS:
    for sub in ('all', 'real'):
        r = HM[(HM.subset == sub) & (HM.metric == m)].iloc[0]
        hv[m, sub] = (r.spearman, r.ci_lo, r.ci_hi, int(r.n))
assert {hv[m, 'all'][3] for m, _ in HUM_METRICS} == {896} and {hv[m, 'real'][3] for m, _ in HUM_METRICS} == {286}
order = sorted(HUM_METRICS[1:], key=lambda x: -hv[x[0], 'all'][0])
HUM_METRICS = [HUM_METRICS[0]] + order
put('C_SPEARMAN', {m: {s: hv[m, s][:3] for s in ('all', 'real')} for m, _ in HUM_METRICS}, '03_HUMAN_REANALYSIS/human_main_metrics.csv')

# ------------------------------------------------------------------ images
for tag, d in (('m', M), ('x', X)):
    for t in ('t1', 't2'):
        Image.open(d[t]).convert('RGB').save(SRC / 'img' / f'{tag}_{t}.png')
put('IMAGES', {'motivation': [M['t1'], M['t2']], 'method': [X['t1'], X['t2']]}, 'pairwise_items.jsonl t1_path/t2_path (original resolution)')

# ------------------------------------------------------------------ LaTeX
# The sources compile with pdfLaTeX (Overleaf) and XeLaTeX/tectonic; T1 + times gives the paper's Times in both.
PRE = r"""\documentclass[border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{amsmath,amssymb,pifont,booktabs,array,graphicx,colortbl}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc}
% colours: validated categorical slots 1-2, neutral ink ramp, status colours (always paired with an icon)
\definecolor{ink}{HTML}{0B0B0B}\definecolor{ink2}{HTML}{52514E}\definecolor{muted}{HTML}{898781}
\definecolor{grid}{HTML}{E1E0D9}\definecolor{axisc}{HTML}{C3C2B7}\definecolor{panel}{HTML}{F6F6F3}
\definecolor{cOurs}{HTML}{2A78D6}\definecolor{cOursL}{HTML}{86B6EF}
\definecolor{cBase}{HTML}{6F6E69}\definecolor{cBaseL}{HTML}{B4B3AB}
\definecolor{cRef}{HTML}{2A78D6}\definecolor{cCand}{HTML}{EB6834}\definecolor{cCandT}{HTML}{B5481C}
\definecolor{cBad}{HTML}{D03B3B}\definecolor{cGood}{HTML}{0CA30C}\definecolor{cGoodText}{HTML}{006300}
\newcommand{\cmark}{{\color{cGood}\ding{51}}}
\newcommand{\xmark}{{\color{cBad}\ding{55}}}
\newcommand{\fs}[1]{\fontsize{#1}{#1}\selectfont\setlength{\baselineskip}{1.17\dimexpr#1pt\relax}}
\newcommand{\method}{\textsc{RICCE}}
\newcommand{\bench}{\textsc{RICCE-Bench}}
\begin{document}
"""


def tex_escape(s):
    s = s.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')
    return s[:-2] + '~.' if s.endswith(' .') else s


def mark_words(cap, words, colour):
    for w in words:
        assert w in cap, (w, cap)
        cap = cap.replace(w, r'\textbf{\color{%s}%s}' % (colour, w), 1)
    return cap


def pref(a, b):
    """Which caption a metric ranks first (A is the reversal the annotators rejected), with its icon."""
    if abs(a - b) <= 1e-6: return 'tie'
    return r'A\,\xmark' if a > b else r'B\,\cmark'


# ---------------- Fig. A: motivation
WTOT, HA = 17.4, 5.9
A = M['a']; B = M['b']
nA = sum(v == 'A' for v in M['votes'].values()); nB = sum(v == 'B' for v in M['votes'].values())
capA = mark_words(tex_escape(A['caption']), ['demolished'], 'cBad')
capB = mark_words(tex_escape(B['caption']), ['built environment'], 'cGoodText')
cols = [('BLEU-4', 'bleu4', f2), ('CIDEr-D', 'cider_d', f2), ('CSM-4', 'csm4', f2), ('BERTScore', 'bertscore', f3), ('LLM judge', 'llm', lambda x: f'{x:.0f}'),
        ('CE-P', 'ce_p', f2), ('CE-R', 'ce_r', f2), (r'\textbf{CE-F}', 'ce_f', f2)]
cell = lambda s, c: (r'\textbf{%s}' % c[2](s[c[1]])) if c[1] == 'ce_f' else c[2](s[c[1]])
TABLE = (r'\setlength{\tabcolsep}{3.2pt}\renewcommand{\arraystretch}{1.1}'
         r'\begin{tabular}{@{}l ccccc!{\color{axisc}\vrule width 0.5pt}ccc@{}}\toprule' '\n'
         r' & \multicolumn{5}{c}{\color{ink2}string, embedding and judge scores} & \multicolumn{3}{c}{\color{cOurs}\method}\\\cmidrule(lr){2-6}\cmidrule(l){7-9}' '\n'
         r'Caption & ' + ' & '.join(c[0] for c in cols) + r'\\\midrule' '\n'
         r'A\,\xmark\ reversed & ' + ' & '.join(cell(A, c) for c in cols) + r'\\' '\n'
         r'B\,\cmark\ coarse & ' + ' & '.join(cell(B, c) for c in cols) + r'\\\midrule' '\n'
         r'\color{ink2}ranks first & ' + ' & '.join(pref(A[c[1]], B[c[1]]) for c in cols) + r'\\\bottomrule\end{tabular}')

lines = [PRE, r'\fs{7}\color{ink}', r'\begin{tikzpicture}[x=1cm,y=1cm]', r'\path[use as bounding box] (0,0) rectangle (%.2f,%.2f);' % (WTOT, HA)]
L = lines.append
# panel (a)
PA = 9.0
L(r'\node[anchor=north west,inner sep=0] at (0,%.2f) {\fs{7.6}\textbf{(a) Same image pair, two captions} \color{ink2}(LEVIR-CC, human study)};' % HA)
IMW, YI = 2.3, HA - 0.42
L(r'\node[anchor=north west,inner sep=0] (t1) at (0,%.2f) {\includegraphics[width=%.2fcm]{img/m_t1.png}};' % (YI, IMW))
L(r'\node[anchor=north west,inner sep=0] (t2) at (%.2f,%.2f) {\includegraphics[width=%.2fcm]{img/m_t2.png}};' % (IMW + 0.08, YI, IMW))
for t, lab in (('t1', r'$T_1$ before'), ('t2', r'$T_2$ after')):
    L(r'\node[anchor=north west,fill=black,fill opacity=0.6,text opacity=1,text=white,inner sep=1.3pt,font=\fs{6.3}] at (%s.north west) {%s};' % (t, lab))
XT = 2 * IMW + 0.24; TW = PA - XT - 0.1
box = r'anchor=north west,inner sep=2.2pt,text width=%.2fcm,align=left,rounded corners=1.5pt,font=\fs{6.8}' % TW
L(r'\node[%s,draw=axisc,fill=panel] (ref) at (%.2f,%.2f) {{\color{ink2}Reference:} \emph{%s}};' % (box, XT, YI, tex_escape(M['ref'])))
L(r'\node[%s,draw=cBad!75,line width=0.5pt,below=1.1mm of ref.south west,anchor=north west] (ca) {\textbf{A}\,\xmark\ \emph{%s}\\[0.5pt]{\color{ink2}annotators choosing A: %d of 3}};' % (box, capA, nA))
L(r'\node[%s,draw=cGood!80!black,line width=0.5pt,below=1.1mm of ca.south west,anchor=north west] (cb) {\textbf{B}\,\cmark\ \emph{%s}\\[0.5pt]{\color{ink2}annotators choosing B: \textbf{%d of 3}}};' % (box, capB, nB))
L(r'\node[anchor=south west,inner sep=0] at (0,0) {\resizebox{%.2fcm}{!}{\fs{6.8}%s}};' % (PA, TABLE))

# panel (b): failure share on the semantic-validity suite
X0, X1 = 9.38, 13.02
L(r'\node[anchor=north west,inner sep=0,text width=%.2fcm,align=left] at (%.2f,%.2f) {\fs{7.6}\textbf{(b) Contradiction ranked first}\\[1pt]'
  r'\fs{6.4}\color{ink2}\bench: share of %s pairs (a correct and a contradictory caption) each metric ranks wrongly};' % (X1 - X0, X0, HA, f'{bc["n_constraints"]:,}'))
YT, BH, GAP = HA - 1.3, 0.3, 0.17
xb0 = X0 + 1.4; xb1 = X1 - 0.38; sc = (xb1 - xb0) / 0.6
ylo = YT - len(FAIL) * (BH + GAP) + GAP
for t in (0, 0.2, 0.4, 0.6):
    x = xb0 + t * sc
    L(r'\draw[grid,line width=0.3pt] (%.3f,%.3f) -- (%.3f,%.3f);' % (x, YT + 0.08, x, ylo - 0.06))
    L(r'\node[anchor=north,inner sep=1pt,text=muted,font=\fs{6}] at (%.3f,%.3f) {%d\%%};' % (x, ylo - 0.06, round(100 * t)))
for i, (name, v) in enumerate(FAIL):
    yt = YT - i * (BH + GAP); yb = yt - BH
    L(r'\fill[cBase,rounded corners=0.6pt] (%.3f,%.3f) rectangle (%.3f,%.3f);' % (xb0, yb, xb0 + v * sc, yt))
    L(r'\node[anchor=east,inner sep=1.5pt,font=\fs{6.6}] at (%.3f,%.3f) {%s};' % (xb0, (yt + yb) / 2, name))
    L(r'\node[anchor=west,inner sep=1.5pt,font=\fs{6.6}] at (%.3f,%.3f) {%.0f\%%};' % (xb0 + v * sc, (yt + yb) / 2, 100 * v))
L(r'\draw[axisc,line width=0.5pt] (%.3f,%.3f) -- (%.3f,%.3f);' % (xb0, YT + 0.08, xb0, ylo - 0.06))
L(r'\node[anchor=south west,inner sep=0,text width=%.2fcm,align=left,font=\fs{6.1},text=ink2] at (%.2f,0) {Ties count as wrong. \method\ ranks none wrongly, by construction of its gate (not shown).};' % (X1 - X0, X0))

# panel (c): agreement with the annotators
C0, C1 = 13.33, WTOT
L(r'\node[anchor=north west,inner sep=0,text width=%.2fcm,align=left] at (%.2f,%.2f) {\fs{7.6}\textbf{(c) Agreement with human ratings}\\[1pt]'
  r'\fs{6.4}\color{ink2}Spearman $\rho$ with the annotators\textquoteright\ mean overall rating (95\%% CI)};' % (C1 - C0, C0, HA))
xc0 = C0 + 1.42; xc1 = C1 - 0.42; scc = (xc1 - xc0) / 1.0
YT = HA - 1.3; bh, gin, gout = 0.155, 0.02, 0.115; rowh = 2 * bh + gin + gout
ylo = YT - len(HUM_METRICS) * rowh + gout
for t in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
    x = xc0 + t * scc
    L(r'\draw[grid,line width=0.3pt] (%.3f,%.3f) -- (%.3f,%.3f);' % (x, YT + 0.06, x, ylo - 0.05))
    L(r'\node[anchor=north,inner sep=1pt,text=muted,font=\fs{6}] at (%.3f,%.3f) {%s};' % (x, ylo - 0.05, {0: '0', 1.0: '1'}.get(t, f'{t:.1f}')))
for i, (m, name) in enumerate(HUM_METRICS):
    yt = YT - i * rowh
    ours = m == 'ce_f1'
    for j, (sub, col) in enumerate((('all', 'cOurs' if ours else 'cBase'), ('real', 'cOursL' if ours else 'cBaseL'))):
        rho, lo, hi, _ = hv[m, sub]
        y1 = yt - j * (bh + gin); y0 = y1 - bh
        L(r'\fill[%s,rounded corners=0.5pt] (%.3f,%.3f) rectangle (%.3f,%.3f);' % (col, xc0, y0, xc0 + max(rho, 0) * scc, y1))
        L(r'\draw[ink,line width=0.35pt] (%.3f,%.3f) -- (%.3f,%.3f);' % (xc0 + lo * scc, (y0 + y1) / 2, xc0 + hi * scc, (y0 + y1) / 2))
        if ours or m == 'llm_judge':
            L(r'\node[anchor=west,inner sep=1pt,font=\fs{6.1}%s] at (%.3f,%.3f) {%s};' % (r'\bfseries' if ours else '', xc0 + hi * scc, (y0 + y1) / 2, f2(rho)))
    L(r'\node[anchor=east,inner sep=1.5pt,font=\fs{6.6}] at (%.3f,%.3f) {%s};' % (xc0, yt - bh - gin / 2, name))
L(r'\draw[axisc,line width=0.5pt] (%.3f,%.3f) -- (%.3f,%.3f);' % (xc0, YT + 0.06, xc0, ylo - 0.05))
L(r'\fill[cBase] (%.3f,0.03) rectangle ++(0.2,0.13); \fill[cOurs] (%.3f,0.03) rectangle ++(0.2,0.13);' % (C0, C0 + 0.22))
L(r'\node[anchor=west,inner sep=1pt,font=\fs{6.1}] at (%.3f,0.095) {all %d items};' % (C0 + 0.44, hv['ce_f1', 'all'][3]))
L(r'\fill[cBaseL] (%.3f,0.03) rectangle ++(0.2,0.13); \fill[cOursL] (%.3f,0.03) rectangle ++(0.2,0.13);' % (C0 + 1.75, C0 + 1.97))
L(r'\node[anchor=west,inner sep=1pt,font=\fs{6.1}] at (%.3f,0.095) {%d real outputs};' % (C0 + 2.19, hv['ce_f1', 'real'][3]))
for xs in (9.19, 13.15):
    L(r'\draw[grid,line width=0.5pt] (%.2f,0.05) -- (%.2f,%.2f);' % (xs, xs, HA - 0.05))
L(r'\end{tikzpicture}'); L(r'\end{document}')
(SRC / 'fig_motivation.tex').write_text('\n'.join(lines) + '\n')

# ---------------- Fig. B: method overview
HB = 6.0
NODE = {'parking_area': 'parking area', 'built_environment': 'built environment'}
gA, pA = G_, P_
for e in gA + pA:
    assert e.target == e.object and e.change_state == 'change'          # the legend states t = o and z = change for all four
Q = lambda e: e.quantity + (f' ({e.exact_count})' if e.exact_count else '')
fo = fs['object']; ft = fs['target']
assert fo == ft and fo[0] == 1.0 and abs(fo[1] - 1 / 3) < 1e-12 and fs['source'] == fs['location'] == (None, None)
assert all(fs[k] == (1.0, 1.0) for k in ('action', 'quantity', 'change_state'))

mlines = [PRE, r'\fs{7}\color{ink}', r'\begin{tikzpicture}[x=1cm,y=1cm,>={Latex[length=1.6mm,width=1.3mm]}]', r'\path[use as bounding box] (0,0) rectangle (%.2f,%.2f);' % (WTOT, HB)]
L = mlines.append
COLS = [(0.0, 3.0, None, 'Inputs'), (3.2, 6.25, 1, 'Event parsing'), (6.45, 9.95, 2, 'Hierarchical scoring'), (10.15, 12.95, 3, 'Contradiction gate'),
        (13.15, 15.65, 4, 'Dual alignment'), (15.85, WTOT, 5, 'Scores')]
TOP, HB0 = HB - 0.62, HB - 0.52
for x0, x1, k, title in COLS:
    L(r'\fill[%s,rounded corners=2pt] (%.2f,%.2f) rectangle (%.2f,%.2f);' % ('cOurs!14' if k else 'panel', x0, HB0, x1, HB))
    if k:
        L(r'\node[circle,fill=cOurs,text=white,inner sep=0pt,minimum size=3.1mm,font=\fs{6.4}\bfseries] at (%.3f,%.3f) {%d};' % (x0 + 0.23, (HB0 + HB) / 2, k))
    L(r'\node[anchor=west,inner sep=0,font=\fs{7.4}\bfseries] at (%.3f,%.3f) {%s};' % (x0 + (0.44 if k else 0.12), (HB0 + HB) / 2, title))
    L(r'\draw[grid,rounded corners=2pt,line width=0.5pt] (%.2f,0) rectangle (%.2f,%.2f);' % (x0, x1, TOP + 0.02))
for (a0, a1, *_), (b0, *__) in zip(COLS[:-1], COLS[1:]):
    L(r'\draw[->,ink2,line width=0.7pt] (%.3f,%.3f) -- (%.3f,%.3f);' % (a1 - 0.04, (HB0 + HB) / 2, b0 + 0.04, (HB0 + HB) / 2))
T0 = TOP - 0.1                                                        # top of column content

# column 0: inputs
L(r'\node[anchor=north west,inner sep=0] (x1) at (0.1,%.2f) {\includegraphics[width=1.36cm]{img/x_t1.png}};' % T0)
L(r'\node[anchor=north west,inner sep=0] (x2) at (1.54,%.2f) {\includegraphics[width=1.36cm]{img/x_t2.png}};' % T0)
for t, lab in (('x1', r'$T_1$'), ('x2', r'$T_2$')):
    L(r'\node[anchor=north west,fill=black,fill opacity=0.6,text opacity=1,text=white,inner sep=1pt,font=\fs{5.9}] at (%s.north west) {%s};' % (t, lab))
L(r'\node[anchor=north west,inner sep=0,text width=2.8cm,align=left,font=\fs{6},text=ink2] (ctx) at (0.1,%.2f) {image pair for context; the metric reads only the captions};' % (T0 - 1.42))
ibox = r'inner sep=2pt,text width=2.62cm,align=left,rounded corners=1.5pt,font=\fs{6.5}'
L(r'\node[%s,draw=cRef,fill=cRef!6,below=1.4mm of ctx.south west,anchor=north west] (gin) {{\color{cRef}\textbf{reference} $g^{(k)}$, one of $K$}\\ \emph{%s}};' % (ibox, tex_escape(X['ref'])))
L(r'\node[%s,draw=cCand,fill=cCand!6,below=1.4mm of gin.south west,anchor=north west] (yin) {{\color{cCandT}\textbf{candidate} $y$}\\ \emph{%s}};'
  % (ibox, mark_words(tex_escape(X['a']['caption']), ['built environment'], 'cCandT')))

# column 1: event parsing
L(r'\node[anchor=north west,inner sep=0,text width=2.85cm,align=left,font=\fs{6.2}] (pp) at (3.3,%.2f) {\hyphenpenalty=10000 one event per clause with its own directional action; each field mapped to a taxonomy node '
  r'(alias $\rightarrow$ $n$-gram $\rightarrow$ embedding); unresolved $=$ absent};' % T0)
L(r'\node[anchor=north west,inner sep=0,font=\fs{6.2},text=ink2,below=1.6mm of pp.south west,anchor=north west] (evh) {fields shown: $(o,\ a,\ q)$};')
prev = 'below=1.1mm of evh.south west,anchor=north west'
for tag, e, col, tc in (('g_1', gA[0], 'cRef', 'cRef'), ('g_2', gA[1], 'cRef', 'cRef'), ('p_1', pA[0], 'cCand', 'cCandT'), ('p_2', pA[1], 'cCand', 'cCandT')):
    nm = tag.replace('_', '')
    L(r'\node[anchor=north west,inner sep=1.7pt,text width=2.7cm,align=left,draw=%s,fill=%s!6,rounded corners=1.2pt,font=\fs{6.3},%s] (%s) '
      r'{{\color{%s}$%s$}\ \ %s\\ \hspace*{3.6mm}%s $\cdot$ %s};' % (col, col, prev, nm, tc, tag, NODE.get(e.object, e.object), e.action, Q(e)))
    prev = 'below=%s of %s.south west,anchor=north west' % ('1.6mm' if tag == 'g_2' else '0.8mm', nm)
L(r'\node[anchor=north west,inner sep=0,text width=2.85cm,align=left,font=\fs{6},text=ink2,below=1.6mm of p2.south west,anchor=north west] {all four: $t{=}o$, $z{=}$change; $s$, $\ell$ absent};')

# column 2: hierarchical scoring
TX = 6.57
L(r'\node[anchor=north west,inner sep=0,font=\fs{6.2},text=ink2] at (%.2f,%.2f) {object taxonomy (part)};' % (TX, T0))
nd = r'draw=%s,line width=%s,fill=%s,rounded corners=1pt,inner sep=1.3pt,font=\fs{6.3},anchor=west'
ys = [T0 - 0.4 - 0.335 * i for i in range(5)]
L(r'\node[%s,dashed] (n0) at (%.2f,%.3f) {entity (root: unspecified)};' % (nd % ('axisc', '0.4pt', 'white'), TX + 0.02, ys[0]))
L(r'\node[%s] (n1) at (%.2f,%.3f) {built environment};' % (nd % ('cCand', '1.1pt', 'cRef!16'), TX + 0.3, ys[1]))
L(r'\node[%s] (n4) at (%.2f,%.3f) {building};' % (nd % ('axisc', '0.4pt', 'white'), TX + 0.58, ys[2]))
L(r'\node[%s] (n2) at (%.2f,%.3f) {transportation};' % (nd % ('axisc', '0.4pt', 'cRef!16'), TX + 0.58, ys[3]))
L(r'\node[%s] (n3) at (%.2f,%.3f) {parking area};' % (nd % ('axisc', '0.4pt', 'cRef!16'), TX + 0.86, ys[4]))
L(r'\node[anchor=west,inner sep=1.5pt,font=\fs{6.1},text=cCandT] at (n1.east) {$\leftarrow o$ of $p_1$};')
L(r'\node[anchor=west,inner sep=1.5pt,font=\fs{6.1},text=cRef] at (n3.east) {$\leftarrow o$ of $g_1$};')
for a_, b_ in (('n0', 'n1'), ('n1', 'n4'), ('n1', 'n2'), ('n2', 'n3')):
    L(r'\draw[axisc,line width=0.5pt] ([xshift=1.6mm]%s.south west) |- (%s.west);' % (a_, b_))
yl = ys[4] - 0.42
L(r'\fill[cRef!16] (%.2f,%.3f) rectangle ++(0.2,0.13); \node[anchor=west,inner sep=1pt,font=\fs{6}] at (%.2f,%.3f) {$A(g)$}; \node[anchor=west,inner sep=1pt,font=\fs{6},text=ink2] at (%.2f,%.3f) {(root excluded)};' % (TX + 0.02, yl, TX + 0.24, yl + 0.065, TX + 1.75, yl + 0.065))
L(r'\draw[cCand,line width=1.1pt] (%.2f,%.3f) rectangle ++(0.2,0.13); \node[anchor=west,inner sep=1pt,font=\fs{6}] at (%.2f,%.3f) {$A(p)$};' % (TX + 0.95, yl, TX + 1.17, yl + 0.065))
L(r'\node[anchor=north west,inner sep=0,font=\fs{6.6},align=left] (hf) at (%.2f,%.3f) {$hP=\frac{|A(p)\cap A(g)|}{|A(p)|}=\frac{1}{1}$\\[2pt] $hR=\frac{|A(p)\cap A(g)|}{|A(g)|}=\frac{1}{3}$};' % (TX + 0.02, yl - 0.12))
L(r'\node[anchor=north west,inner sep=0,font=\fs{6.2},text=ink2,below=1.2mm of hf.south west,anchor=north west] (hn) {coarse: full precision, partial recall};')
assert f2(fo[1]) == '0.33'
L(r'\node[anchor=north west,inner sep=0,text width=3.3cm,align=left,font=\fs{6.1},below=1.6mm of hn.south west,anchor=north west] (axn) {per axis of $(p_1,g_1)$, $(hP,hR)$:\\'
  r'\ \ $o$, $t$: $(1, %s)$;\ \ $a$, $q$, $z$: $(1, 1)$\\[1pt] weighted by $w$: $z$ %.1f, $a$ %.1f, $o,t$ %.1f, $q$ %.1f};'
  % (f2(fo[1]), W['change_state'], W['action'], W['object'], W['quantity']))
assert W['object'] == W['target']
L(r'\node[anchor=north west,inner sep=0,font=\fs{6.6},below=1.2mm of axn.south west,anchor=north west] {$\Rightarrow\ S_P=\mathbf{%s},\ \ S_R=\mathbf{%s}$};' % (f2(sp0), f2(sr0)))

# column 3: contradiction gate (Fig. A, caption A)
GX = 10.25
L(r'\node[anchor=north west,inner sep=0,text width=2.6cm,align=left,font=\fs{6.2}] (gt) at (%.2f,%.2f) {sets $S_P$ and $S_R$ of an event pair to 0 if the events assert opposite changes about compatible objects:\\[1pt]'
  r'$\cdot$ change vs.\ no change\\$\cdot$ opposed action families\\$\cdot$ source and target swapped};' % (GX, T0))
tn = r'anchor=west,inner sep=1.3pt,rounded corners=1pt,font=\fs{6.3}'
ya = [T0 - 2.02 - 0.315 * k for k in range(5)]
L(r'\node[%s,draw=axisc] (r0) at (%.2f,%.3f) {change};' % (tn, GX + 0.02, ya[0]))
L(r'\node[%s,draw=axisc,fill=cRef!16] (r1) at (%.2f,%.3f) {appearance};' % (tn, GX + 0.32, ya[1]))
L(r'\node[%s,draw=cRef,fill=cRef!16] (r3) at (%.2f,%.3f) {constructed};' % (tn, GX + 0.62, ya[2]))
L(r'\node[%s,draw=axisc,fill=cCand!14] (r2) at (%.2f,%.3f) {disappearance};' % (tn, GX + 0.32, ya[3]))
L(r'\node[%s,draw=cCand,fill=cCand!14] (r4) at (%.2f,%.3f) {demolished};' % (tn, GX + 0.62, ya[4]))
for a_, b_ in (('r0', 'r1'), ('r1', 'r3'), ('r0', 'r2'), ('r2', 'r4')):
    L(r'\draw[axisc,line width=0.5pt] ([xshift=1.4mm]%s.south west) |- (%s.west);' % (a_, b_))
L(r'\draw[<->,cBad,line width=0.8pt] ([xshift=0.8mm]r1.east) to[out=0,in=0,looseness=1.3] node[right=0.5pt,font=\fs{5.9},text=cBad,align=left] {opposed\\families} ([xshift=0.8mm]r2.east);')
L(r'\node[anchor=north west,inner sep=0,text width=2.6cm,align=left,font=\fs{6.2}] (gx) at (%.2f,%.3f) {Fig.~1(a): \textcolor{cRef}{built} (reference) vs.\ \textcolor{cCandT}{demolished} (caption A), same object};' % (GX, ya[4] - 0.22))
L(r'\node[anchor=north west,inner sep=1.8pt,draw=cBad,line width=0.6pt,rounded corners=1.2pt,font=\fs{6.3},below=1.1mm of gx.south west,anchor=north west] (gb) {\xmark\ \ $S_P=S_R$: $%s\rightarrow 0$};' % f2(ung[0]))
assert ung[0] == ung[1]
L(r'\node[anchor=north west,inner sep=0,font=\fs{6.2},text=ink2,below=0.9mm of gb.south west,anchor=north west] {CE-F of caption A $=%s$};' % f2(A['ce_f']))

# column 4: dual alignment
AX = 13.27


def matrix(x, y, Mx, name, sel):
    cw, chh = 0.6, 0.34
    out = [r'\node[anchor=west,inner sep=0,font=\fs{6.8}] at (%.3f,%.3f) {$%s$};' % (x, y + 0.17, name)]
    for j in range(2):
        out.append(r'\node[font=\fs{6.3},text=cRef,inner sep=0] at (%.3f,%.3f) {$g_%d$};' % (x + 0.62 + (j + 0.5) * cw, y + 0.17, j + 1))
    for i in range(2):
        out.append(r'\node[font=\fs{6.3},text=cCandT,inner sep=0] at (%.3f,%.3f) {$p_%d$};' % (x + 0.42, y - (i + 0.5) * chh, i + 1))
        for j in range(2):
            x0_, y0_ = x + 0.62 + j * cw, y - (i + 1) * chh
            on = (i, j) in sel
            out.append(r'\draw[%s] (%.3f,%.3f) rectangle ++(%.3f,%.3f);' % ('fill=cOurs!20,draw=cOurs,line width=0.8pt' if on else 'draw=grid,line width=0.4pt', x0_, y0_, cw, chh))
            out.append(r'\node[font=\fs{6.5}%s,inner sep=0] at (%.3f,%.3f) {%s};' % (r'\bfseries' if on else '', x0_ + cw / 2, y0_ + chh / 2, f2(Mx[i, j])))
    return out


selP = set(zip(rP.tolist(), cP.tolist())); selR = set(zip(rR.tolist(), cR.tolist()))
for l_ in matrix(AX, T0 - 0.3, SP, 'S_P', selP): L(l_)
L(r'\node[anchor=north west,inner sep=0,align=left,font=\fs{6.3}] at (%.2f,%.2f) {CE-P $=\tfrac12(%s+%s)$\\ \hspace*{5.2mm}$=\mathbf{%s}$};' % (AX, T0 - 1.06, f2(SP[0, 0]), f2(SP[1, 1]), f2(cep)))
for l_ in matrix(AX, T0 - 2.02, SR, 'S_R', selR): L(l_)
L(r'\node[anchor=north west,inner sep=0,align=left,font=\fs{6.3}] at (%.2f,%.2f) {CE-R $=\tfrac12(%s+%s)$\\ \hspace*{5.2mm}$=\mathbf{%s}$};' % (AX, T0 - 2.73, f2(SR[0, 0]), f2(SR[1, 1]), f2(cer)))
L(r'\node[anchor=north west,inner sep=0,text width=2.3cm,align=left,font=\fs{6.2}] at (%.2f,%.2f) {one Hungarian assignment per matrix (boxed): precision and recall stay separate; '
  r'unmatched candidate events lower CE-P, unmatched reference events lower CE-R};' % (AX, T0 - 3.4))

# column 5: scores
OX = 15.97
for i, (lab, v) in enumerate((('CE-P', cep), ('CE-R', cer), ('CE-F', X['a']['ce_f']))):
    y = T0 - 0.05 - i * 0.6
    L(r'\node[anchor=west,inner sep=0,font=\fs{7}] at (%.2f,%.3f) {%s};' % (OX, y - 0.2, lab))
    L(r'\node[anchor=east,inner sep=0,font=\fs{10.5}\bfseries,text=cOurs] at (%.2f,%.3f) {%s};' % (WTOT - 0.12, y - 0.2, f2(v)))
L(r'\node[anchor=north west,inner sep=0,text width=1.42cm,align=left,font=\fs{6},text=ink2] at (%.2f,%.2f) {CE-F: harmonic mean; all three averaged over the $K$ references};' % (OX, T0 - 1.92))
for i, (t1, t2) in enumerate(((r'$P{=}1,\ R{<}1$', 'coarse'), (r'$P{<}1,\ R{=}1$', 'over-specific'), (r'gated pair', 'reversed: 0'))):
    y = T0 - 3.3 - i * 0.6
    L(r'\node[anchor=north west,inner sep=1.5pt,text width=1.3cm,align=left,draw=grid,fill=panel,rounded corners=1.2pt,font=\fs{6.2}] at (%.2f,%.2f) {%s\\ \color{ink2}%s};' % (OX, y, t1, t2))
L(r'\end{tikzpicture}'); L(r'\end{document}')
(SRC / 'fig_method.tex').write_text('\n'.join(mlines) + '\n')

# ------------------------------------------------------------------ LaTeX for the paper (numbers filled from the values above)
hier = [HM[(HM.subset == 'all') & (HM.metric == m)].spearman.iloc[0] for m in ('b_kiritchenko_macro', 'b_lca_f', 'b_hcm')]
fails = [v for _, v in FAIL]
assert all(hv['ce_f1', s][0] > hv[m, s][0] for m, _ in HUM_METRICS[1:] for s in ('all', 'real'))   # "more closely than the metrics shown"
assert all(pref(A[c[1]], B[c[1]]) == r'A\,\xmark' for c in cols if c[1] in ('bleu4', 'cider_d', 'csm4', 'bertscore'))
assert all(pref(A[c[1]], B[c[1]]) == r'B\,\cmark' for c in cols if c[1] in ('llm', 'ce_p', 'ce_r', 'ce_f'))
rng = lambda xs, k=2: f'{min(xs):.{k}f}--{max(xs):.{k}f}'
pct = f'{100 * min(fails):.0f}--{100 * max(fails):.0f}'
SNIP = r"""%% ===== Generated by scripts/cvpr27_22_make_overview_figures.py; every number below is in figure_values.json =====
%% (1) main.tex: replace the whole \begin{center} ... \end{center} block inside \twocolumn[{...}] (the old Fig. 1:
%%     TikZ pipeline strip + illustrative table) with:
\begin{center}
\vspace{-1.5em}
\includegraphics[width=\textwidth]{figures/fig_motivation.pdf}
\captionof{figure}{\textbf{Why change captions need a different metric.} (a) Two captions of one LEVIR-CC image pair from our human study, scored against the reference the study used. Caption A reverses the change with one word; caption B is correct but coarser than the reference. All three annotators chose B, yet BLEU-4, CIDEr-D, CSM-4 and BERTScore rank A first; the LLM judge (1--5) and \method\ rank B first, and \cep/\cer\ show that B is correct but under-specified. We selected this pair among the @@NSEL@@ study pairs in which all annotators rejected an action reversal that BLEU-4, CIDEr-D and BERTScore preferred. (b) On \bench, these metrics rank a contradictory caption above a correct one in @@PCT@@\% of @@NCON@@ comparisons. (c) On @@NALL@@ rated captions (@@NREAL@@ real system outputs), \cef\ agrees with the annotators more closely than the string, embedding and judge metrics shown; prior hierarchical metrics computed on \method's event parses come close (@@HIER@@; \cref{tab:human}).}
\label{fig:teaser}
\end{center}

%% (2) main.tex, Introduction: replace the paragraph that starts "A change caption states which object changed"
%%     (it quotes the numbers of the old illustrative table) with:
A change caption states which object changed, in which direction, from which state to which, where, how much, and whether anything changed at all. \cref{fig:teaser}(a) shows two captions of one image pair from our human study. Against the reference \emph{``@@REF@@''}, caption A differs by one word, \emph{demolished}, and asserts the opposite change; caption B replaces \emph{houses} by the coarser \emph{built environment}. All three annotators chose B, yet BLEU-4 (@@B_A@@ against @@B_B@@), CIDEr-D (@@C_A@@ against @@C_B@@), CSM-4 (@@S_A@@ against @@S_B@@) and BERTScore (@@E_A@@ against @@E_B@@) rank the reversal first, because one differing token barely moves surface or embedding overlap; on \bench\ such metrics rank a contradictory caption above a correct one in @@PCT@@\% of comparisons (\cref{fig:teaser}(b)). A useful metric should rank a coarse or an over-specific caption above a reversal, and should say which of the two lacks detail and which adds unsupported detail.

%% (3) main.tex, Section 3: insert after its first paragraph (the one ending "...against human ratings (\cref{sec:human-results})."):
\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{figures/fig_method.pdf}
\caption{\textbf{Overview of \method} on a two-event pair from the human study (LEVIR-CC; all three annotators preferred this candidate to the same caption with its first event reversed). (1) Each caption is parsed into structured events; unresolved fields stay absent. (2) Each axis is scored with hierarchical precision and recall over its taxonomy, root excluded: the candidate's \emph{built environment} is a correct ancestor of the reference's \emph{parking area}, so it keeps full precision and loses recall; fixed axis weights combine them into $S_P$ and $S_R$. (3) The object-scoped gate zeroes event pairs that assert opposite changes about compatible objects (here caption A of \cref{fig:teaser}). (4) \cep\ and \cer\ are read off two separately optimal one-to-one assignments; (5) \cef\ is their harmonic mean, averaged over the $K$ references. Every value is computed by the evaluator.}
\label{fig:method}
\end{figure*}
"""
for k_, v_ in {'NSEL': {4: 'four'}[len(cand)], 'PCT': pct, 'NCON': f'{bc["n_constraints"]:,}'.replace(',', '{,}'), 'NALL': str(hv['ce_f1', 'all'][3]),
               'NREAL': str(hv['ce_f1', 'real'][3]), 'HIER': rng(hier), 'REF': tex_escape(M['ref']).replace('~.', '').rstrip(' .'),
               'B_A': f2(A['bleu4']), 'B_B': f2(B['bleu4']), 'C_A': f2(A['cider_d']), 'C_B': f2(B['cider_d']), 'S_A': f2(A['csm4']), 'S_B': f2(B['csm4']),
               'E_A': f3(A['bertscore']), 'E_B': f3(B['bertscore'])}.items():
    SNIP = SNIP.replace(f'@@{k_}@@', v_)
assert '@@' not in SNIP and len(cand) == 4 and nB == 3 and X['votes'] == {'ann1': 'A', 'ann2': 'A', 'ann3': 'A'}
(OUT / 'snippets_for_main.tex').write_text(SNIP)
put('SNIPPET_NUMBERS', dict(pct=pct, hier=rng(hier)), 'snippets_for_main.tex')

# ------------------------------------------------------------------ compile + previews
for name in ('fig_motivation', 'fig_method'):
    r = subprocess.run([TECTONIC, '-X', 'compile', f'{name}.tex', '--outdir', str(OUT)], cwd=SRC, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:]); raise SystemExit(f'{name}: compile failed')
    for w in ('Overfull', 'Underfull \\hbox (badness 10000)', 'undefined'):
        if w in r.stderr: print(f'{name}: warning {w}')
    subprocess.run(['pdftoppm', '-png', '-r', '600', '-singlefile', str(OUT / f'{name}.pdf'), str(OUT / name)], check=True)
    info = subprocess.run(['pdfinfo', str(OUT / f'{name}.pdf')], capture_output=True, text=True).stdout
    size = [l for l in info.splitlines() if l.startswith('Page size')][0]
    put(f'{name}_page_size', size.split(':', 1)[1].strip(), 'pdfinfo')
json.dump(VAL, open(OUT / 'figure_values.json', 'w'), indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())
print({k: VAL[k]['value'] for k in ('fig_motivation_page_size', 'fig_method_page_size', 'SEL_CANDIDATES')})
