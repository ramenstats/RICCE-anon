"""Framework figure and taxonomy figure of RICCE, in the style of Fig. 2 of Snaebjarnarson et al. (2025),
"Taxonomy-aware evaluation of vision-language models": node-link taxonomies on which the candidate's and the
reference's nodes are marked and their ancestor sets A(p), A(g) (root excluded) are shaded.

  fig_framework (full width): the whole evaluator on one real example,
    LEVIR-CC test_001589, the caption of RSICCformer (results/baselines/rsiccformer/LEVIR_CC_test.jsonl) scored against
    reference 3 of the image's five (artifacts/manifests/LEVIR_CC.jsonl).
    Selection: we searched the seven systems' LEVIR-CC test captions of at most 16 words for an event pair whose
    object, action, location and quantity are all present and whose fields differ at different taxonomy levels, and
    chose this one (a choice made after seeing the candidates). Its words map by alias (house -> building, built ->
    constructed, appears -> appearance, disappears -> disappearance, top/bottom right -> upper/lower right, a ->
    single), so the figure shows no mapper quirk. It has a second candidate event, which shows the gate and an
    unmatched event.
    (1) captions and parsed events; (2) each field on its taxonomy (pruned trees), A(p) and A(g) shaded;
    (3) per-field hP/hR and the weighted S_P, S_R; (4) gate, the two optimal assignments, CE-P/CE-R/CE-F and the
    mean over the K references (equal to the saved per-image value of results_v3/systems/per_image).
  fig_taxonomies (full width): the four taxonomies printed from configs/ontology/*.yaml as node-link trees, one ancestor
    set per tree shaded, and the three cases of hP/hR computed by Taxonomy.hierarchical_scores.

Every number drawn is computed here with the evaluator (ricce/metrics/final.py configuration) and checked;
figure_values.json lists each with its source. Outputs -> CVPR2027_Framework_Figures/.
"""
import json, os, shutil, subprocess, sys
from fractions import Fraction
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.benchmarks.env import load_taxes, load_weights
from ricce.evaluation.parse_store import ParseStore
from ricce.metrics.final import final_scores
from ricce.metrics.event_v2 import event_field_scores_v2, aggregate_pr, event_similarity_pr_v2, _score_matrix
from ricce.utils import read_jsonl

OUT = ROOT / 'CVPR2027_Framework_Figures'
SRC = OUT / 'src'
(SRC / 'img').mkdir(parents=True, exist_ok=True)
TECTONIC = os.environ.get('TECTONIC') or shutil.which('tectonic') or os.path.expanduser('~/tectonic_env/bin/tectonic')
T, W = load_taxes(), load_weights()
VAL = {}


def put(k, v, src):
    VAL[k] = dict(value=v, source=src)
    return v


f2 = lambda x: f'{x:.2f}'


def frac(x):
    f = Fraction(x).limit_denominator(12)
    assert abs(float(f) - x) < 1e-9
    return str(f.numerator) if f.denominator == 1 else f'{f.numerator}/{f.denominator}'


# ------------------------------------------------------------------ the example
SYS, IMG, K_REF = 'rsiccformer', 'test_001589', 2
store = ParseStore(ROOT / 'results_v3' / 'parse_cache.json')
ev = lambda t: store.events(t, 'multi')
man = {r['image_id']: r for r in read_jsonl(ROOT / 'artifacts' / 'manifests' / 'LEVIR_CC.jsonl') if r.get('split') == 'test'}[IMG]
pred = [r for r in read_jsonl(ROOT / 'results' / 'baselines' / SYS / 'LEVIR_CC_test.jsonl') if r['image_id'] == IMG][0]['prediction'].strip()
refs = man['references']; ref = refs[K_REF].strip()
P, G = ev(pred), ev(refs[K_REF])
assert pred == 'a building disappears and a house appears at the bottom right corner of the scene', pred
assert ref == 'a house is built at the top right corner of the scene .', ref
assert len(P) == 2 and len(G) == 1 and len(refs) == 5
F = ('change_state', 'action', 'object', 'source', 'target', 'location', 'quantity', 'exact_count')
evd = lambda e: {f: getattr(e, f) for f in F}
EXP = dict(p1=dict(change_state='change', action='disappearance', object='building', source='building', target=None, location=None, quantity='single', exact_count=None),
           p2=dict(change_state='change', action='appearance', object='building', source=None, target='building', location='lower_right', quantity='single', exact_count=None),
           g1=dict(change_state='change', action='constructed', object='building', source=None, target='building', location='upper_right', quantity='single', exact_count=None))
assert evd(P[0]) == EXP['p1'] and evd(P[1]) == EXP['p2'] and evd(G[0]) == EXP['g1'], (evd(P[0]), evd(P[1]), evd(G[0]))
SRCE = (f'results/baselines/{SYS}/LEVIR_CC_test.jsonl ({IMG}); reference {K_REF + 1} of 5 from artifacts/manifests/LEVIR_CC.jsonl; '
        'events from results_v3/parse_cache.json (multi-event parse)')
put('EXAMPLE', dict(system=SYS, image_id=IMG, reference_index_1based=K_REF + 1, n_references=len(refs), candidate=pred, reference=ref,
                    t1=man['t1_path'], t2=man['t2_path']), SRCE)
put('EVENTS', EXP, SRCE)

# per-field scores of the matched pair (p2, g1) and of (p1, g1)
fs2 = event_field_scores_v2(P[1], G[0], T, True, 'unspecified')
fs1 = event_field_scores_v2(P[0], G[0], T, True, 'unspecified')
sp2, sr2 = aggregate_pr(fs2, W)
assert fs2['source'] == (None, None)
FIELDS2 = [('change_state', r'$z$', 'change state'), ('action', r'$a$', 'action'), ('object', r'$o$', 'object'),
           ('target', r'$t$', 'target'), ('location', r'$\ell$', 'location'), ('quantity', r'$q$', 'quantity')]
anc = lambda ax, n: T[ax].ancestors(n)[1:]                                  # A(v): root excluded
A_P2 = dict(object=anc('object', 'building'), action=anc('action', 'appearance'), location=anc('location', 'lower_right'), quantity=anc('quantity', 'single'))
A_G1 = dict(object=anc('object', 'building'), action=anc('action', 'constructed'), location=anc('location', 'upper_right'), quantity=anc('quantity', 'single'))
assert A_P2['action'] == ['change', 'appearance'] and A_G1['action'] == ['change', 'appearance', 'constructed']
assert A_P2['location'] == ['right', 'lower_right'] and A_G1['location'] == ['right', 'upper_right']
assert A_P2['object'] == A_G1['object'] == ['built_environment', 'building'] and A_P2['quantity'] == ['single']
# the drawn fractions are |A(p) n A(g)| / |A(p)| and / |A(g)|; check them against the evaluator
for ax in ('object', 'action', 'location'):
    i = len(set(A_P2[ax]) & set(A_G1[ax]))
    assert abs(fs2[ax][0] - i / len(A_P2[ax])) < 1e-12 and abs(fs2[ax][1] - i / len(A_G1[ax])) < 1e-12, ax
assert fs2['quantity'] == (1.0, 1.0) and fs2['change_state'] == (1.0, 1.0) and fs2['target'] == (1.0, 1.0)
TAB = []
for k, sym, name in FIELDS2:
    hp, hr = fs2[k]
    if k in A_P2 and k != 'quantity':
        i = len(set(A_P2[k]) & set(A_G1[k]))
        hps, hrs = f'{i}/{len(A_P2[k])}', f'{i}/{len(A_G1[k])}'
    elif k == 'target':
        hps, hrs = '2/2', '2/2'                                               # target building = building, A = {built env., building}
        assert anc('object', 'building') == ['built_environment', 'building']
    else:
        hps, hrs = frac(hp), frac(hr)
    TAB.append((sym, name, W[k], hps, hrs, hp, hr))
num_p = sum(W[k] * fs2[k][0] for k, *_ in FIELDS2); den = sum(W[k] for k, *_ in FIELDS2)
assert abs(num_p / den - sp2) < 1e-12
put('FIELDS_p2_g1', {k: fs2[k] for k, *_ in FIELDS2}, 'event_field_scores_v2(p2, g1, root_mode=unspecified)')
put('FIELDS_p1_g1', fs1, 'event_field_scores_v2(p1, g1, root_mode=unspecified)')
put('WEIGHTS', W, 'configs/project.yaml via load_weights()')
put('SPSR_p2_g1', [sp2, sr2], 'aggregate_pr (eq. 2)')

# gate, the two assignments, CE
SP, SR = _score_matrix(P, G, T, W, 'hard', 0.1, True, 'unspecified', True)
ung = event_similarity_pr_v2(P[0], G[0], T, W, gate_mode='none', root_mode='unspecified')
gat = event_similarity_pr_v2(P[0], G[0], T, W, gate_mode='hard', root_mode='unspecified', gate_object_match=True)
assert gat == (0.0, 0.0) and SP[0, 0] == SR[0, 0] == 0.0 and abs(SP[1, 0] - sp2) < 1e-12 and abs(SR[1, 0] - sr2) < 1e-12
rP, cP = linear_sum_assignment(-SP); rR, cR = linear_sum_assignment(-SR)
assert list(rP) == [1] and list(rR) == [1]                                 # p2 matched to g1 in both assignments; p1 unmatched
sc = final_scores(P, G, T, W)
cep, cer, cef = sc['ce_precision'], sc['ce_recall'], sc['ce_f1']
assert abs(cep - SP[1, 0] / 2) < 1e-12 and abs(cer - SR[1, 0] / 1) < 1e-12 and abs(cef - 2 * cep * cer / (cep + cer)) < 1e-12
per_ref = []
for g in refs:
    s = final_scores(P, ev(g), T, W)
    per_ref.append(np.nan if s['not_scorable'] else s['ce_f1'])
mean_f = float(np.nanmean(per_ref))
saved = pd.read_csv(ROOT / 'results_v3' / 'systems' / 'per_image' / f'LEVIR_CC__{SYS}.csv').set_index('image_id').loc[IMG]
assert abs(saved.ce_f - mean_f) < 1e-12 and saved.n_ref_not_scorable == 0
put('GATE_p1_g1', dict(ungated=list(ung), gated=list(gat)), 'event_similarity_pr_v2 with gate_mode none / hard (final configuration)')
put('SP', SP.tolist(), 'event_v2._score_matrix (final configuration)'); put('SR', SR.tolist(), 'event_v2._score_matrix (final configuration)')
put('CE_this_reference', [cep, cer, cef], 'final_scores(P, G_3)')
put('CE_F_per_reference', per_ref, 'final_scores against each of the 5 references')
put('CE_F_mean_over_K', mean_f, f'mean of the 5; equals results_v3/systems/per_image/LEVIR_CC__{SYS}.csv ce_f for {IMG}')

# ------------------------------------------------------------------ the three cases of hP/hR (taxonomy figure)
CASES = [('coarser', 'action', 'appearance', 'constructed'), ('over-specific', 'action', 'constructed', 'appearance'),
         ('sibling', 'location', 'lower_right', 'upper_right')]
CV = {}
for name, ax, p, g in CASES:
    hp, hr, _ = T[ax].hierarchical_scores(p, g, exclude_root=True)
    CV[name] = (hp, hr)
assert CV['coarser'] == (1.0, 2 / 3) and CV['over-specific'] == (2 / 3, 1.0) and CV['sibling'] == (0.5, 0.5)
put('CASES', {n: dict(axis=ax, pred=p, ref=g, hp_hr=CV[n]) for n, ax, p, g in CASES}, 'Taxonomy.hierarchical_scores(exclude_root=True)')
NODES = {ax: len(T[ax].nodes) for ax in ('object', 'action', 'location', 'quantity')}
assert NODES == dict(object=21, action=11, location=12, quantity=6)
put('N_NODES', NODES, 'configs/ontology/*.yaml (root included)')

# ------------------------------------------------------------------ images
for t in ('t1', 't2'):
    Image.open(man[f'{t}_path']).convert('RGB').save(SRC / 'img' / f'f_{t}.png')

# ------------------------------------------------------------------ LaTeX
PRE = r"""\documentclass[border=1pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{amsmath,amssymb,pifont,array,graphicx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,calc}
% colours as in the paper's other figures: blue = reference, orange = candidate (CVD-checked pair); red only with an icon
\definecolor{ink}{HTML}{0B0B0B}\definecolor{ink2}{HTML}{52514E}\definecolor{muted}{HTML}{898781}
\definecolor{grid}{HTML}{E1E0D9}\definecolor{panel}{HTML}{F6F6F3}
\definecolor{cRef}{HTML}{2A78D6}\definecolor{cRefT}{HTML}{1C5BA8}\definecolor{cCand}{HTML}{EB6834}\definecolor{cCandT}{HTML}{B5481C}
\definecolor{cBad}{HTML}{D03B3B}
\newcommand{\xmark}{{\color{cBad}\ding{55}}}
\newcommand{\fs}[1]{\fontsize{#1}{#1}\selectfont\setlength{\baselineskip}{1.15\dimexpr#1pt\relax}}
\newcommand{\method}{\textsc{RICCE}}
\tikzset{
  nd/.style={circle, minimum size=1.9mm, inner sep=0pt, draw=ink2, line width=0.45pt, fill=white},
  rootnd/.style={nd, draw=muted, dashed, line width=0.5pt},
  edge/.style={draw=ink2!75, line width=0.45pt},
  lab/.style={font=\fs{5.6}, text=ink2, inner sep=0.6pt, align=center},
  hullG/.style={line width=3.0mm, line cap=round, line join=round, draw=cRef, draw opacity=0.26},
  hullP/.style={line width=3.0mm, line cap=round, line join=round, draw=cCand, draw opacity=0.30},
  badge/.style={fill=muted, text=white, rounded corners=1pt, minimum size=3.1mm, inner sep=0pt, font=\fs{6.4}\bfseries},
  ttl/.style={anchor=west, font=\fs{7.4}, text=ink, inner sep=0pt},
}
\begin{document}
"""


def tex(s):
    return s.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')


def label_of(ax, n):
    return T[ax].nodes[n].label


# ---- mini trees of the framework figure: hand-placed coordinates (cm, local), structure checked against the ontology
# spec: node -> (x, y, label position, label text); parent links are read from the ontology.
MINI = {
    'object': dict(title=r'object $o$ \textcolor{muted}{(also $s$, $t$; part)}', nodes={
        'entity': (1.6, 2.45, 'right', 'entity (root)'),
        'built_environment': (1.15, 1.8, 'upleft', 'built\\\\environment'),
        'vegetation': (2.2, 1.8, 'below', 'vegetation'),
        'building': (0.6, 1.15, 'right', 'building'),
        'transportation': (1.85, 1.15, 'below', 'transportation\\\\infrastructure'),
        'residential_building': (0.22, 0.5, 'below', 'residential\\\\building'),
        'industrial_building': (1.12, 0.5, 'below', 'industrial\\\\building'),
    }, P='building', G='building'),
    'action': dict(title=r'action $a$ \textcolor{muted}{(part)}', nodes={
        'temporal_state': (0.85, 2.45, 'right', 'temporal state (root)'),
        'no_change': (0.3, 1.8, 'below', 'no change'),
        'change': (1.35, 1.8, 'right', 'change'),
        'appearance': (0.75, 1.15, 'leftfar', 'appearance'),
        'disappearance': (1.8, 1.15, 'right', 'disappearance'),
        'constructed': (0.75, 0.5, 'below', 'constructed'),
        'demolished': (1.8, 0.5, 'below', 'demolished'),
    }, P='appearance', G='constructed', P1='disappearance'),
    'location': dict(title=r'location $\ell$ \textcolor{muted}{(part)}', nodes={
        'image': (1.45, 2.45, 'right', 'image (root)'),
        'left': (0.4, 1.7, 'below', 'left'),
        'center': (1.3, 1.7, 'below', 'center'),
        'right': (2.3, 1.7, 'right', 'right'),
        'upper_right': (1.7, 0.85, 'below', 'upper\\\\right'),
        'center_right': (2.3, 0.85, 'below', 'center\\\\right'),
        'lower_right': (2.9, 0.85, 'below', 'lower\\\\right'),
    }, P='lower_right', G='upper_right'),
    'quantity': dict(title=r'quantity $q$', nodes={
        'quantity': (1.5, 2.45, 'right', 'quantity (root)'),
        'none': (0.2, 1.55, 'below', 'none'),
        'single': (0.8, 1.55, 'below', 'single'),
        'few': (1.68, 1.55, 'below', 'few'),
        'several': (2.27, 1.55, 'below', 'several'),
        'many': (2.85, 1.55, 'below', 'many'),
    }, P='single', G='single', tagside='east', note='an exact count, if both\\\\captions give one, is\\\\compared numerically'),
}
for ax, spec in MINI.items():
    t = T[ax]
    for n in spec['nodes']:
        assert n in t.nodes, (ax, n)
        par = t.nodes[n].parent
        assert par is None or par in spec['nodes'], (ax, n, par)
    assert t.root in spec['nodes']
# the pruned action tree must keep the opposed families the gate uses
assert T['action'].nodes['constructed'].parent == 'appearance' and T['action'].nodes['demolished'].parent == 'disappearance'


def place(x, y, pos):
    return {'right': (f'{x + 0.13:.3f},{y:.3f}', 'west'), 'left': (f'{x - 0.13:.3f},{y:.3f}', 'east'),
            'below': (f'{x:.3f},{y - 0.12:.3f}', 'north'), 'above': (f'{x:.3f},{y + 0.12:.3f}', 'south'),
            'leftfar': (f'{x - 0.22:.3f},{y:.3f}', 'east'), 'upleft': (f'{x - 0.12:.3f},{y + 0.04:.3f}', 'south east')}[pos]


def split_node(x, y, r=0.095):
    return (rf'\fill[cCand] ({x},{y}) -- ++(90:{r}) arc (90:270:{r}) -- cycle;'
            rf'\fill[cRef] ({x},{y}) -- ++(-90:{r}) arc (-90:90:{r}) -- cycle;'
            rf'\draw[ink2, line width=0.45pt] ({x},{y}) circle ({r});')


def mini_tree(ax, ox, oy):
    spec = MINI[ax]; t = T[ax]; nd = spec['nodes']
    xy = lambda n: (ox + nd[n][0], oy + nd[n][1])
    L = [rf'\node[anchor=west, font=\fs{{6.6}}, text=ink, inner sep=0pt] at ({ox:.3f},{oy + 2.74:.3f}) {{{spec["title"]}}};']
    # ancestor sets (root excluded): A(g) blue, A(p) orange, drawn first
    for key, style in (('G', 'hullG'), ('P', 'hullP')):
        path = (A_G1 if key == 'G' else A_P2)[ax]
        pts = [xy(n) for n in path]
        if len(pts) == 1:
            c = 'cRef' if key == 'G' else 'cCand'; op = 0.26 if key == 'G' else 0.30
            L.append(rf'\fill[{c}, fill opacity={op}] ({pts[0][0]:.3f},{pts[0][1]:.3f}) circle (1.5mm);')
        else:
            L.append(rf'\draw[{style}] ' + ' -- '.join(f'({x:.3f},{y:.3f})' for x, y in pts) + ';')
    for n in nd:
        par = t.nodes[n].parent
        if par is not None:
            (x1, y1), (x2, y2) = xy(par), xy(n)
            L.append(rf'\draw[edge] ({x1:.3f},{y1:.3f}) -- ({x2:.3f},{y2:.3f});')
    for n, (x, y, pos, text) in nd.items():
        X, Y = ox + x, oy + y
        if n == t.root:
            L.append(rf'\node[rootnd] at ({X:.3f},{Y:.3f}) {{}};')
        elif n == spec['P'] == spec['G']:
            L.append(split_node(X, Y))
        elif n == spec['P']:
            L.append(rf'\node[nd, fill=cCand, draw=cCandT] at ({X:.3f},{Y:.3f}) {{}};')
        elif n == spec['G']:
            L.append(rf'\node[nd, fill=cRef, draw=cRefT] at ({X:.3f},{Y:.3f}) {{}};')
        elif n == spec.get('P1'):
            L.append(rf'\node[nd, draw=cCandT, line width=0.9pt] at ({X:.3f},{Y:.3f}) {{}};')
        else:
            L.append(rf'\node[nd] at ({X:.3f},{Y:.3f}) {{}};')
        at, anchor = place(X, Y, pos)
        col = 'muted' if n == t.root else 'ink2'
        L.append(rf'\node[lab, text={col}, anchor={anchor}] at ({at}) {{{text}}};')
    # event tags next to the marked nodes, on the side away from the edge to the parent
    def tag(n, s, side):
        X, Y = xy(n)
        par = t.nodes[n].parent
        left = side if side else ('left' if xy(par)[0] >= X else 'right')
        dx, dy, anchor = {'left': (-0.1, 0.08, 'south east'), 'right': (0.1, 0.08, 'south west'), 'east': (0.12, 0.0, 'west')}[left]
        L.append(rf'\node[font=\fs{{5.6}}, inner sep=0pt, anchor={anchor}] at ({X + dx:.3f},{Y + dy:.3f}) {{{s}}};')
    if spec['P'] == spec['G']:
        tag(spec['P'], r'\textcolor{cCandT}{$p_2$}\,\textcolor{cRefT}{$g_1$}', spec.get('tagside'))
    else:
        tag(spec['P'], r'\textcolor{cCandT}{$p_2$}', None)
        tag(spec['G'], r'\textcolor{cRefT}{$g_1$}', None)
    if spec.get('P1'):
        tag(spec['P1'], r'\textcolor{cCandT}{$p_1$}', 'right')
    if spec.get('more'):
        n, txt = spec['more']; X, Y = xy(n)
        L.append(rf'\node[lab, text=muted, anchor=west, align=left] at ({X + 0.42:.3f},{Y + 0.02:.3f}) {{{txt}}};')
    if spec.get('note'):
        L.append(rf'\node[lab, text=muted, anchor=north west, align=left] at ({ox + 0.15:.3f},{oy + 0.95:.3f}) {{{spec["note"]}}};')
    return L


def framework_tex():
    H = 6.8
    L = [PRE, r'\begin{tikzpicture}[x=1cm,y=1cm]', rf'\useasboundingbox (0,0) rectangle (17.4,{H});']
    X1, X2, X3, X4 = 0.0, 4.45, 11.0, 14.45
    hy = H - 0.2
    heads = [(X1, '1', 'Parse captions into events'), (X2, '2', r'Map each field onto its taxonomy'),
             (X3, '3', 'Score fields, then events'), (X4, '4', 'Gate, match, score')]
    for x, n, s in heads:
        L.append(rf'\node[badge] at ({x + 0.16:.2f},{hy:.2f}) {{{n}}};')
        L.append(rf'\node[ttl] at ({x + 0.40:.2f},{hy:.2f}) {{{s}}};')
    for x in (X2 - 0.16, X3 - 0.14, X4 - 0.12):
        L.append(rf'\draw[grid, line width=0.6pt] ({x:.2f},0.0) -- ({x:.2f},{H - 0.45:.2f});')
        L.append(rf'\draw[-{{Stealth[length=1.5mm,width=1.3mm]}}, muted, line width=0.6pt] ({x - 0.09:.2f},{hy:.2f}) -- ({x + 0.09:.2f},{hy:.2f});')

    # ---- panel 1: images, captions, events
    top = H - 0.5
    L.append(rf'\node[anchor=north west, inner sep=0pt] (t1) at (0.08,{top:.2f}) {{\includegraphics[width=1.95cm]{{img/f_t1.png}}}};')
    L.append(rf'\node[anchor=north west, inner sep=0pt] (t2) at (2.18,{top:.2f}) {{\includegraphics[width=1.95cm]{{img/f_t2.png}}}};')
    for nm in ('t1', 't2'):
        L.append(rf'\node[anchor=north west, fill=black, fill opacity=0.55, text opacity=1, text=white, font=\fs{{6.4}}, inner sep=1.2pt] at ({nm}.north west) {{$T_{nm[1]}$}};')
    ev_row = lambda tag_, col, e: (rf'\textcolor{{{col}}}{{${tag_}$}} & {e["action"].replace("_", " ")} & {e["object"].replace("_", " ")} & '
                                   rf'{(e["location"] or "--").replace("_", " ")} & {e["quantity"]}\\')
    head = r'& \textcolor{muted}{$a$} & \textcolor{muted}{$o$} & \textcolor{muted}{$\ell$} & \textcolor{muted}{$q$}\\[-0.8pt]'
    tabfmt = r'@{}l@{\hspace{2.6pt}}l@{\hspace{2.6pt}}l@{\hspace{2.6pt}}l@{\hspace{2.6pt}}l@{}'
    refcap = r'a \textbf{house} is \textbf{built} at the \textbf{top right corner} of the scene'
    candcap = r'a \textbf{building} \textbf{disappears} and a \textbf{house} \textbf{appears} at the \textbf{bottom right corner} of the scene'
    assert refcap.replace(r'\textbf{', '').replace('}', '') + ' .' == ref
    assert candcap.replace(r'\textbf{', '').replace('}', '') == pred
    box = r'anchor=north west, line width=0.6pt, rounded corners=1.5pt, text width=3.9cm, inner sep=2.2pt, font=\fs{6.2}, align=left'
    L.append(rf'''\node[{box}, draw=cRef] (rb) at (0.08,{top - 2.12:.2f}) {{%
\textcolor{{cRefT}}{{\textbf{{reference}} $g$ \,(1 of $K{{=}}{len(refs)}$)}}\\[1pt]
\textit{{{refcap}}}\\[2pt]
{{\fs{{5.8}}\begin{{tabular}}{{{tabfmt}}}{head}{ev_row('g_1', 'cRefT', EXP['g1'])}\end{{tabular}}}}}};''')
    L.append(rf'''\node[{box}, draw=cCand] (cb) at (0.08,{top - 3.62:.2f}) {{%
\textcolor{{cCandT}}{{\textbf{{candidate}} $y$ \,(RSICCformer)}}\\[1pt]
\textit{{{candcap}}}\\[2pt]
{{\fs{{5.8}}\begin{{tabular}}{{{tabfmt}}}{head}{ev_row('p_1', 'cCandT', EXP['p1'])}{ev_row('p_2', 'cCandT', EXP['p2'])}\end{{tabular}}}}}};''')
    L.append(r'\node[anchor=north west, font=\fs{5.6}, text=muted, inner sep=0pt, text width=4.0cm, align=left] at (0.1,0.42) '
             r'{every event here has change state $z$\,=\,change;\\ -- : field absent (not mentioned)};')

    # ---- panel 2: the four taxonomies
    yt, yb = 3.5, 0.36
    L += mini_tree('object', X2 + 0.05, yt)
    L += mini_tree('action', X2 + 3.3, yt)
    L += mini_tree('location', X2 + 0.05, yb)
    L += mini_tree('quantity', X2 + 3.3, yb)
    # opposed families on the action tree (the gate)
    nda = MINI['action']['nodes']
    a1 = (X2 + 3.3 + nda['appearance'][0], yt + nda['appearance'][1])
    a2 = (X2 + 3.3 + nda['disappearance'][0], yt + nda['disappearance'][1])
    L.append(rf'\draw[{{Stealth[length=1.3mm]}}-{{Stealth[length=1.3mm]}}, cBad, line width=0.6pt, dashed] ({a1[0] + 0.14:.3f},{a1[1]:.3f}) -- ({a2[0] - 0.14:.3f},{a2[1]:.3f});')
    L.append(rf'\node[font=\fs{{5.4}}, text=cBad, inner sep=0pt, anchor=north] at ({(a1[0] + a2[0]) / 2 + 0.08:.3f},{a1[1] - 0.08:.3f}) {{\xmark\ opposed}};')
    # legend
    ly = 0.14; lx = X2 + 0.1
    L.append(rf'\node[nd, fill=cCand, draw=cCandT] at ({lx:.2f},{ly}) {{}};\node[lab, anchor=west] at ({lx + 0.1:.2f},{ly}) {{candidate}};')
    L.append(rf'\node[nd, fill=cRef, draw=cRefT] at ({lx + 1.05:.2f},{ly}) {{}};\node[lab, anchor=west] at ({lx + 1.15:.2f},{ly}) {{reference}};')
    L.append(rf'\draw[hullP, line width=2.2mm] ({lx + 2.12:.2f},{ly}) -- ({lx + 2.37:.2f},{ly});\node[lab, anchor=west] at ({lx + 2.48:.2f},{ly}) {{$A(p)$}};')
    L.append(rf'\draw[hullG, line width=2.2mm] ({lx + 3.05:.2f},{ly}) -- ({lx + 3.3:.2f},{ly});\node[lab, anchor=west] at ({lx + 3.41:.2f},{ly}) {{$A(g)$}};')
    L.append(rf'\draw[hullG, line width=2.2mm] ({lx + 3.98:.2f},{ly}) -- ({lx + 4.23:.2f},{ly});\draw[hullP, line width=2.2mm] ({lx + 3.98:.2f},{ly}) -- ({lx + 4.23:.2f},{ly});\node[lab, anchor=west] at ({lx + 4.34:.2f},{ly}) {{both}};')
    L.append(rf'\node[rootnd] at ({lx + 4.95:.2f},{ly}) {{}};\node[lab, anchor=west] at ({lx + 5.05:.2f},{ly}) {{root (not in $A$)}};')

    # ---- panel 3: field and event scores
    x = X3 + 0.05
    AP, AG = r'\textcolor{cCandT}{A(p)}', r'\textcolor{cRefT}{A(g)}'
    L.append(rf'\node[anchor=north west, inner sep=0pt, font=\fs{{7.0}}] at ({x:.2f},{top:.2f}) {{$hP_k=\dfrac{{|{AP}\cap {AG}|}}{{|{AP}|}}$}};')
    L.append(rf'\node[anchor=north west, inner sep=0pt, font=\fs{{7.0}}] at ({x:.2f},{top - 0.72:.2f}) {{$hR_k=\dfrac{{|{AP}\cap {AG}|}}{{|{AG}|}}$}};')
    rows = '\n'.join(rf'{sym}\ {name.replace("change state", "change")} & {w:g} & {hps} & {hrs}\\' for sym, name, w, hps, hrs, hp, hr in TAB)
    L.append(rf'''\node[anchor=north west, inner sep=0pt, font=\fs{{6.4}}] at ({x:.2f},{top - 1.55:.2f}) {{%
\begin{{tabular}}{{@{{}}l@{{\hspace{{6pt}}}}r@{{\hspace{{8pt}}}}c@{{\hspace{{8pt}}}}c@{{}}}}
\multicolumn{{4}}{{@{{}}l}}{{\textcolor{{muted}}{{fields of }}\textcolor{{cCandT}}{{$p_2$}} \textcolor{{muted}}{{vs.}} \textcolor{{cRefT}}{{$g_1$}}}}\\[1pt]
\textcolor{{muted}}{{field}} & \textcolor{{muted}}{{$w_k$}} & \textcolor{{muted}}{{$hP_k$}} & \textcolor{{muted}}{{$hR_k$}}\\[-0.5pt]
\hline\noalign{{\vskip 1pt}}
{rows}
\hline
\end{{tabular}}}};''')
    L.append(rf'''\node[anchor=north west, inner sep=0pt, font=\fs{{6.8}}, align=left] at ({x:.2f},{top - 3.75:.2f}) {{%
$S_P=\dfrac{{\sum_k w_k\,hP_k}}{{\sum_k w_k}}=\mathbf{{{f2(sp2)}}}$\\[4pt] $S_R=\dfrac{{\sum_k w_k\,hR_k}}{{\sum_k w_k}}=\mathbf{{{f2(sr2)}}}$}};''')
    L.append(rf'''\node[anchor=north west, inner sep=0pt, font=\fs{{5.6}}, text=muted, text width=3.2cm, align=left] at ({x:.2f},0.62) {{%
coarser action: full precision, partial recall; sibling location: both partial; $s$ absent on both sides.}};''')

    # ---- panel 4: gate, matching, CE
    x = X4 + 0.05
    def mat(name, M, x0, y0):
        out = [rf'\node[anchor=west, font=\fs{{6.8}}, inner sep=0pt] at ({x0:.2f},{y0:.2f}) {{${name}$}};',
               rf'\node[font=\fs{{6.2}}, text=cRefT, inner sep=0pt] at ({x0 + 0.85:.2f},{y0 - 0.02:.2f}) {{$g_1$}};']
        for i, row in enumerate(M):
            yy = y0 - 0.36 - 0.36 * i
            out.append(rf'\node[font=\fs{{6.2}}, text=cCandT, inner sep=0pt] at ({x0 + 0.2:.2f},{yy:.2f}) {{$p_{i + 1}$}};')
            matched = i == 1
            bx = 'draw=cRef, line width=0.9pt' if matched else 'draw=grid, line width=0.4pt'
            txt = rf'\textbf{{{f2(row[0])}}}' if matched else rf'\xmark\,{f2(row[0])}'
            out.append(rf'\node[{bx}, minimum width=0.78cm, minimum height=0.3cm, inner sep=0pt, font=\fs{{6.4}}] at ({x0 + 0.85:.2f},{yy:.2f}) {{{txt}}};')
        return out
    L += mat('S_P', SP, x, top - 0.12)
    L += mat('S_R', SR, x + 1.45, top - 0.12)
    L.append(rf'''\node[anchor=north west, inner sep=0pt, font=\fs{{5.7}}, text=ink2, text width=2.85cm, align=left] at ({x:.2f},{top - 1.15:.2f}) {{%
\xmark\ \textbf{{gate}}: $p_1$ (disappearance) and $g_1$ (constructed) assert opposite changes of the same object:
$S_P,S_R\!: {f2(ung[0])}, {f2(ung[1])}\to 0$.\\[2pt]
\textcolor{{cRefT}}{{$\square$}} \textbf{{matching}}: one optimal one-to-one assignment per matrix (Hungarian); $p_2\!\leftrightarrow\! g_1$, $p_1$ stays unmatched.}};''')
    L.append(rf'''\node[anchor=north west, inner sep=0pt, font=\fs{{6.8}}, align=left] at ({x:.2f},{top - 3.05:.2f}) {{%
$\mathrm{{CE\text{{-}}P}}=\tfrac{{{f2(SP[1, 0])}\,+\,0}}{{2}}=\mathbf{{{f2(cep)}}}$\\[4pt]
$\mathrm{{CE\text{{-}}R}}=\tfrac{{{f2(SR[1, 0])}}}{{1}}=\mathbf{{{f2(cer)}}}$\\[4pt]
$\mathrm{{CE\text{{-}}F}}=\tfrac{{2\,\mathrm{{P}}\,\mathrm{{R}}}}{{\mathrm{{P}}+\mathrm{{R}}}}=\textcolor{{cRefT}}{{\mathbf{{{f2(cef)}}}}}$}};''')
    L.append(rf'''\node[anchor=north west, inner sep=0pt, font=\fs{{5.7}}, text=ink2, text width=2.85cm, align=left] at ({x:.2f},1.55) {{%
CE-P divides by the $|P|{{=}}2$ candidate events, so the unmatched $p_1$ lowers it; CE-R divides by the $|G|{{=}}1$ reference events.\\[3pt]
mean over the $K{{=}}{len(refs)}$ references:\\ \textbf{{CE-F\,=\,{f2(mean_f)}}}}};''')
    L += [r'\end{tikzpicture}', r'\end{document}']
    return '\n'.join(L)


# ---- full taxonomies: indented node-link trees from the ontology
ROW, DX = 0.265, 0.36
HILITE = dict(object='residential_building', action='constructed', location='upper_right', quantity='few')
TITLES = dict(object=r'\textbf{Object} \textcolor{muted}{(also source, target)}', action=r'\textbf{Action}',
              location=r'\textbf{Location}', quantity=r'\textbf{Quantity}')


def dfs(t, n=None, d=0, out=None):
    out = [] if out is None else out
    n = t.root if n is None else n
    out.append((n, d))
    for c in t.children.get(n, []):
        dfs(t, c, d + 1, out)
    return out


def full_tree(ax, ox, oy):
    t = T[ax]; rows = dfs(t)
    pos = {n: (ox + d * DX, oy - (i + 1) * ROW) for i, (n, d) in enumerate(rows)}
    L = [rf'\node[anchor=south west, font=\fs{{7.0}}, inner sep=0pt] at ({ox - 0.1:.3f},{oy - 0.05:.3f}) {{{TITLES[ax]} \textcolor{{muted}}{{\fs{{6}}{NODES[ax]} nodes}}}};']
    h = HILITE[ax]; A = t.ancestors(h)[1:]
    # ancestor set of the highlighted node along the connectors (root excluded)
    pts = []
    for k, n in enumerate(A):
        x, y = pos[n]
        if k:
            px, py = pos[A[k - 1]]
            pts.append((px, y))
        pts.append((x, y))
    if len(pts) == 1:
        L.append(rf'\fill[cRef, fill opacity=0.26] ({pts[0][0]:.3f},{pts[0][1]:.3f}) circle (1.5mm);')
    else:
        L.append(r'\draw[hullG] ' + ' -- '.join(f'({x:.3f},{y:.3f})' for x, y in pts) + ';')
    for n, d in rows:
        par = t.nodes[n].parent
        if par is not None:
            (px, py), (x, y) = pos[par], pos[n]
            L.append(rf'\draw[edge] ({px:.3f},{py - 0.095:.3f}) -- ({px:.3f},{y:.3f}) -- ({x - 0.095:.3f},{y:.3f});')
    for n, d in rows:
        x, y = pos[n]
        if n == t.root:
            L.append(rf'\node[rootnd] at ({x:.3f},{y:.3f}) {{}};')
            L.append(rf'\node[lab, font=\fs{{6.2}}, text=muted, anchor=west] at ({x + 0.12:.3f},{y:.3f}) {{{tex(t.nodes[n].label)} (root: unspecified)}};')
        else:
            sty = 'nd, fill=cRef, draw=cRefT' if n == h else 'nd'
            L.append(rf'\node[{sty}] at ({x:.3f},{y:.3f}) {{}};')
            L.append(rf'\node[lab, font=\fs{{6.2}}, text=ink, anchor=west] at ({x + 0.12:.3f},{y:.3f}) {{{tex(t.nodes[n].label)}}};')
    yb = oy - (len(rows) + 1) * ROW - 0.08
    names = ', '.join(tex(t.nodes[n].label) for n in A)
    L.append(rf'\node[anchor=north west, font={{\fs{{6.2}}\hyphenpenalty10000\exhyphenpenalty10000}}, inner sep=0pt, text width=4.0cm, align=flush left] at ({ox - 0.1:.3f},{yb + 0.1:.3f}) '
             rf'{{$A(\textcolor{{cRefT}}{{\text{{{tex(t.nodes[h].label)}}}}})=\{{$\textcolor{{cRefT}}{{{names}}}$\}}$, $|A|={len(A)}$}};')
    return L, pos


def taxonomies_tex():
    L = [PRE, r'\begin{tikzpicture}[x=1cm,y=1cm]']
    top = 6.4
    OX = dict(object=0.15, action=4.75, location=9.05, quantity=12.75)
    Lo, _ = full_tree('object', OX['object'], top)
    La, pa = full_tree('action', OX['action'], top)
    Ll, _ = full_tree('location', OX['location'], top)
    Lq, pq = full_tree('quantity', OX['quantity'], top)
    L += Lo + La + Ll + Lq
    # opposed families of the action taxonomy (contradiction gate)
    xr = OX['action'] + 2.95
    for a, b in (('appearance', 'disappearance'), ('expansion', 'contraction')):
        (x1, y1), (x2, y2) = pa[a], pa[b]
        wa = 1.05 if a == 'appearance' else 1.0
        wb = 1.25 if b == 'disappearance' else 1.08
        L.append(rf'\draw[{{Stealth[length=1.2mm]}}-{{Stealth[length=1.2mm]}}, cBad, line width=0.55pt, dashed, rounded corners=2pt] '
                 rf'({x1 + wa:.3f},{y1:.3f}) -- ({xr:.3f},{y1:.3f}) -- ({xr:.3f},{y2:.3f}) -- ({x2 + wb:.3f},{y2:.3f});')
    L.append(rf'\node[font=\fs{{5.8}}, text=cBad, anchor=west, align=left, inner sep=0pt] at ({xr + 0.08:.3f},{(pa["appearance"][1] + pa["contraction"][1]) / 2:.3f}) {{\xmark\ opposed\\families}};')
    # legend under the quantity tree
    lx, ly = OX['quantity'], pq['many'][1] - 1.15
    L.append(rf'\node[rootnd] at ({lx:.2f},{ly:.2f}) {{}};\node[lab, font=\fs{{6.0}}, anchor=west] at ({lx + 0.12:.2f},{ly:.2f}) {{root: mentioned but unspecified; not in $A$}};')
    L.append(rf'\draw[hullG, line width=2.2mm] ({lx - 0.08:.2f},{ly - 0.32:.2f}) -- ({lx + 0.08:.2f},{ly - 0.32:.2f});\node[nd, fill=cRef, draw=cRefT] at ({lx + 0.08:.2f},{ly - 0.32:.2f}) {{}};'
             rf'\node[lab, font=\fs{{6.0}}, anchor=west] at ({lx + 0.22:.2f},{ly - 0.32:.2f}) {{a node $v$ and its ancestor set $A(v)$}};')
    L.append(rf'\draw[{{Stealth[length=1.2mm]}}-{{Stealth[length=1.2mm]}}, cBad, line width=0.55pt, dashed] ({lx - 0.1:.2f},{ly - 0.64:.2f}) -- ({lx + 0.15:.2f},{ly - 0.64:.2f});'
             rf'\node[lab, font=\fs{{6.0}}, anchor=west] at ({lx + 0.22:.2f},{ly - 0.64:.2f}) {{opposed families (contradiction gate)}};')
    # how to read: the definitions and the three cases, computed with the taxonomy code
    co, os_, si = CV['coarser'], CV['over-specific'], CV['sibling']
    yb = 2.15
    L.append(rf'\draw[grid, line width=0.6pt] ({OX["action"] - 0.1:.2f},{yb + 0.12:.2f}) -- (17.3,{yb + 0.12:.2f});')
    L.append(rf'''\node[anchor=north west, inner sep=0pt, font={{\fs{{6.4}}\hyphenpenalty10000}}, text width=4.0cm, align=flush left] at ({OX['action']:.2f},{yb:.2f}) {{%
\textbf{{Ancestor set}} $A(v)$: the nodes on the path from the root to $v$, \emph{{root excluded}}. A child is more specific than its parent; a root label asserts nothing beyond the field and is scored as absent.}};''')
    L.append(rf'''\node[anchor=north west, inner sep=0pt, font=\fs{{6.8}}, align=left] at ({OX['location'] - 0.1:.2f},{yb:.2f}) {{%
for a candidate node \textcolor{{cCandT}}{{$p$}}\\ and a reference node \textcolor{{cRefT}}{{$g$}}:\\[4pt]
$hP=\dfrac{{|A(\textcolor{{cCandT}}{{p}})\cap A(\textcolor{{cRefT}}{{g}})|}}{{|A(\textcolor{{cCandT}}{{p}})|}}$\\[5pt]
$hR=\dfrac{{|A(\textcolor{{cCandT}}{{p}})\cap A(\textcolor{{cRefT}}{{g}})|}}{{|A(\textcolor{{cRefT}}{{g}})|}}$}};''')
    L.append(rf'''\node[anchor=north west, inner sep=0pt, font={{\fs{{6.3}}\hyphenpenalty10000}}, text width=5.3cm, align=flush left] at ({OX['quantity'] - 0.95:.2f},{yb:.2f}) {{%
\begin{{tabular}}{{@{{}}l@{{\hspace{{5pt}}}}l@{{\hspace{{5pt}}}}c@{{}}}}
\textcolor{{muted}}{{case}} & \textcolor{{cCandT}}{{$p$}} \textcolor{{muted}}{{vs.}} \textcolor{{cRefT}}{{$g$}} & \textcolor{{muted}}{{$(hP,hR)$}}\\
coarser & appearance vs.\ constructed & $({frac(co[0])},\,{frac(co[1])})$\\
over-specific & constructed vs.\ appearance & $({frac(os_[0])},\,{frac(os_[1])})$\\
sibling & lower right vs.\ upper right & $({frac(si[0])},\,{frac(si[1])})$\\
\end{{tabular}}\\[4pt]
A coarser answer keeps full precision, an over-specific one full recall; a sibling loses both.}};''')
    L += [r'\end{tikzpicture}', r'\end{document}']
    return '\n'.join(L)


def build(name, text):
    (SRC / f'{name}.tex').write_text(text)
    r = subprocess.run([TECTONIC, '-X', 'compile', f'{name}.tex', '--outdir', str(OUT)], cwd=SRC, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:]); raise SystemExit(f'{name}: LaTeX failed')
    subprocess.run(['pdftoppm', '-png', '-r', '600', '-singlefile', str(OUT / f'{name}.pdf'), str(OUT / name)], check=True)
    info = subprocess.run(['pdfinfo', str(OUT / f'{name}.pdf')], capture_output=True, text=True).stdout
    size = [l for l in info.splitlines() if l.startswith('Page size')][0]
    print(name, size)
    return size


if __name__ == '__main__':
    s1 = build('fig_framework', framework_tex())
    s2 = build('fig_taxonomies', taxonomies_tex())
    put('PAGE_SIZES', dict(fig_framework=s1, fig_taxonomies=s2), 'pdfinfo')
    (OUT / 'figure_values.json').write_text(json.dumps(VAL, indent=1, default=float))
    print(json.dumps({k: VAL[k]['value'] for k in ('SPSR_p2_g1', 'CE_this_reference', 'CE_F_mean_over_K', 'GATE_p1_g1')}, indent=1, default=float))
