"""Version 2 of the framework and taxonomy figures (2026-11-11): only the text needed to read them, all
text black, no table of weights. Same example and same values as scripts/cvpr27_39_framework_figures.py, whose
computation part (everything before its image/LaTeX part) is executed here unchanged, with all its checks.
Version 1 (CVPR2027_Framework_Figures/) is not changed.

  fig_framework: (1) captions and parsed events; (2) the four taxonomies with the candidate's and the reference's
    nodes and ancestor sets, each tree titled with its (hP, hR); (3) hP, hR, S_P, S_R; (4) gate, matching, CE scores.
  fig_taxonomies: the four taxonomies from configs/ontology/*.yaml, one ancestor set per tree shaded, a short legend.
Outputs -> CVPR2027_Framework_Figures_v2/ (figure_values.json is the same set of values as version 1).
"""
import json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / 'scripts' / 'cvpr27_39_framework_figures.py'
MARK = '# ------------------------------------------------------------------ images'
src = V1.read_text(); assert src.count(MARK) == 1
ns = {'__file__': str(V1), '__name__': 'cvpr27_39_values'}
exec(compile(src.split(MARK)[0], str(V1), 'exec'), ns)              # values + checks of version 1
T, W, VAL, A_P2, A_G1, fs2 = ns['T'], ns['W'], ns['VAL'], ns['A_P2'], ns['A_G1'], ns['fs2']
EXP, SP, SR, frac, f2 = ns['EXP'], ns['SP'], ns['SR'], ns['frac'], ns['f2']
sp2, sr2, cep, cer, cef, mean_f, refs, man, pred, ref = (ns[k] for k in ('sp2', 'sr2', 'cep', 'cer', 'cef', 'mean_f', 'refs', 'man', 'pred', 'ref'))

OUT = ROOT / 'CVPR2027_Framework_Figures_v2'
SRC = OUT / 'src'
(SRC / 'img').mkdir(parents=True, exist_ok=True)
TECTONIC = ns['TECTONIC']
from PIL import Image
for t in ('t1', 't2'):
    Image.open(man[f'{t}_path']).convert('RGB').save(SRC / 'img' / f'f_{t}.png')

PRE = r"""\documentclass[border=1pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{amsmath,amssymb,pifont,array,graphicx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,calc}
% all text black; colour only on marks: blue = reference, orange = candidate (CVD-checked pair), red = gate (with an icon)
\definecolor{ink}{HTML}{000000}\definecolor{edgec}{HTML}{6F6E69}\definecolor{grid}{HTML}{D9D8D2}\definecolor{badgec}{HTML}{6F6E69}
\definecolor{cRef}{HTML}{2A78D6}\definecolor{cRefD}{HTML}{1C5BA8}\definecolor{cCand}{HTML}{EB6834}\definecolor{cCandD}{HTML}{B5481C}
\definecolor{cBad}{HTML}{D03B3B}
\newcommand{\xmark}{{\color{cBad}\ding{55}}}
\newcommand{\fs}[1]{\fontsize{#1}{#1}\selectfont\setlength{\baselineskip}{1.15\dimexpr#1pt\relax}}
\tikzset{
  every node/.style={text=ink},
  nd/.style={circle, minimum size=2.1mm, inner sep=0pt, draw=edgec, line width=0.5pt, fill=white},
  rootnd/.style={nd, draw=edgec, dashed, line width=0.55pt},
  edge/.style={draw=edgec, line width=0.5pt},
  lab/.style={font=\fs{6.4}, inner sep=0.6pt, align=center},
  hullG/.style={line width=3.2mm, line cap=round, line join=round, draw=cRef, draw opacity=0.28},
  hullP/.style={line width=3.2mm, line cap=round, line join=round, draw=cCand, draw opacity=0.32},
  badge/.style={fill=badgec, text=white, rounded corners=1pt, minimum size=3.3mm, inner sep=0pt, font=\fs{6.8}\bfseries},
  ttl/.style={anchor=west, font=\fs{7.8}\bfseries, inner sep=0pt},
}
\begin{document}
"""

# ---- mini trees (cm, local to the cell); parents are read from the ontology and checked
MINI = {
    'object': dict(title='Object', ax='object', nodes={
        'entity': (1.55, 2.2, 'right', 'entity'),
        'built_environment': (1.0, 1.62, 'upleft', 'built\\\\environment'),
        'vegetation': (2.3, 1.62, 'below', 'vegetation'),
        'building': (0.65, 1.04, 'right', 'building'),
        'transportation': (2.17, 1.04, 'below', 'transportation\\\\infrastructure'),
        'residential_building': (0.32, 0.46, 'below', 'residential\\\\building'),
        'industrial_building': (1.36, 0.46, 'below', 'industrial\\\\building'),
    }, P='building', G='building'),
    'action': dict(title='Action', ax='action', nodes={
        'temporal_state': (0.95, 2.2, 'right', 'temporal state'),
        'change': (1.2, 1.62, 'right', 'change'),
        'appearance': (0.65, 1.04, 'leftfar', 'appearance'),
        'disappearance': (1.72, 1.04, 'right', 'disappearance'),
        'constructed': (0.65, 0.46, 'below', 'constructed'),
        'demolished': (1.72, 0.46, 'right', 'demolished'),
    }, P='appearance', G='constructed', P1='disappearance', tagG='east'),
    'location': dict(title='Location', ax='location', nodes={
        'image': (1.5, 2.2, 'right', 'image'),
        'left': (0.35, 1.45, 'below', 'left'),
        'center': (1.25, 1.45, 'below', 'center'),
        'right': (2.3, 1.45, 'right', 'right'),
        'upper_right': (1.6, 0.62, 'below', 'upper\\\\right'),
        'center_right': (2.3, 0.62, 'below', 'center\\\\right'),
        'lower_right': (3.0, 0.62, 'below', 'lower\\\\right'),
    }, P='lower_right', G='upper_right'),
    'quantity': dict(title='Quantity', ax='quantity', nodes={
        'quantity': (1.5, 2.2, 'right', 'quantity'),
        'none': (0.15, 1.3, 'below', 'none'),
        'single': (0.72, 1.3, 'below', 'single'),
        'few': (1.72, 1.3, 'below', 'few'),
        'several': (2.3, 1.3, 'below', 'several'),
        'many': (2.95, 1.3, 'below', 'many'),
    }, P='single', G='single', tagside='east'),
}
for ax, spec in MINI.items():
    t = T[ax]
    assert t.root in spec['nodes']
    for n in spec['nodes']:
        assert n in t.nodes and (t.nodes[n].parent is None or t.nodes[n].parent in spec['nodes']), (ax, n)
    assert spec['P'] == {'object': 'building', 'action': 'appearance', 'location': 'lower_right', 'quantity': 'single'}[ax]
    assert spec['G'] == {'object': 'building', 'action': 'constructed', 'location': 'upper_right', 'quantity': 'single'}[ax]


def place(x, y, pos):
    return {'right': (x + 0.14, y, 'west'), 'left': (x - 0.14, y, 'east'), 'below': (x, y - 0.13, 'north'),
            'leftfar': (x - 0.24, y, 'east'), 'upleft': (x - 0.13, y + 0.04, 'south east')}[pos]


def mini_tree(ax, ox, oy):
    spec = MINI[ax]; t = T[ax]; nd = spec['nodes']
    xy = lambda n: (ox + nd[n][0], oy + nd[n][1])
    hp, hr = fs2[ax]
    vals = f'$hP=hR={frac(hp)}$' if hp == hr else f'$hP={frac(hp)}$, $hR={frac(hr)}$'
    L = [rf'\node[anchor=west, font=\fs{{7.4}}, inner sep=0pt] at ({ox:.3f},{oy + 2.58:.3f}) {{\textbf{{{spec["title"]}}}\enspace{{\fs{{6.8}}{vals}}}}};']
    for key, style, col, op in (('G', 'hullG', 'cRef', 0.28), ('P', 'hullP', 'cCand', 0.32)):
        pts = [xy(n) for n in (A_G1 if key == 'G' else A_P2)[ax]]
        if len(pts) == 1:
            L.append(rf'\fill[{col}, fill opacity={op}] ({pts[0][0]:.3f},{pts[0][1]:.3f}) circle (1.6mm);')
        else:
            L.append(rf'\draw[{style}] ' + ' -- '.join(f'({x:.3f},{y:.3f})' for x, y in pts) + ';')
    for n in nd:
        par = t.nodes[n].parent
        if par is not None:
            (x1, y1), (x2, y2) = xy(par), xy(n)
            L.append(rf'\draw[edge] ({x1:.3f},{y1:.3f}) -- ({x2:.3f},{y2:.3f});')
    r = 0.105
    for n, (x, y, pos, text) in nd.items():
        X, Y = ox + x, oy + y
        if n == t.root:
            L.append(rf'\node[rootnd] at ({X:.3f},{Y:.3f}) {{}};')
        elif n == spec['P'] == spec['G']:
            L.append(rf'\fill[cCand] ({X:.3f},{Y:.3f}) -- ++(90:{r}) arc (90:270:{r}) -- cycle;\fill[cRef] ({X:.3f},{Y:.3f}) -- ++(-90:{r}) arc (-90:90:{r}) -- cycle;'
                     rf'\draw[edgec, line width=0.5pt] ({X:.3f},{Y:.3f}) circle ({r});')
        elif n == spec['P']:
            L.append(rf'\node[nd, fill=cCand, draw=cCandD] at ({X:.3f},{Y:.3f}) {{}};')
        elif n == spec['G']:
            L.append(rf'\node[nd, fill=cRef, draw=cRefD] at ({X:.3f},{Y:.3f}) {{}};')
        elif n == spec.get('P1'):
            L.append(rf'\node[nd, draw=cCandD, line width=1.0pt] at ({X:.3f},{Y:.3f}) {{}};')
        else:
            L.append(rf'\node[nd] at ({X:.3f},{Y:.3f}) {{}};')
        lx, ly, anchor = place(X, Y, pos)
        L.append(rf'\node[lab, anchor={anchor}] at ({lx:.3f},{ly:.3f}) {{{text}}};')

    def tag(n, s, side):
        X, Y = xy(n)
        side = side or ('left' if xy(t.nodes[n].parent)[0] >= X else 'right')
        dx, dy, anchor = {'left': (-0.11, 0.08, 'south east'), 'right': (0.11, 0.08, 'south west'), 'east': (0.14, 0.0, 'west')}[side]
        L.append(rf'\node[font=\fs{{6.6}}, inner sep=0pt, anchor={anchor}] at ({X + dx:.3f},{Y + dy:.3f}) {{{s}}};')
    if spec['P'] == spec['G']:
        tag(spec['P'], r'$p_2{,}g_1$', spec.get('tagside'))
    else:
        tag(spec['P'], r'$p_2$', None); tag(spec['G'], r'$g_1$', spec.get('tagG'))
    if spec.get('P1'):
        tag(spec['P1'], r'$p_1$', 'right')
    return L


def framework_tex():
    H = 6.4
    L = [PRE, r'\begin{tikzpicture}[x=1cm,y=1cm]', rf'\useasboundingbox (0,0) rectangle (17.4,{H});']
    X1, X2, X3, X4 = 0.0, 4.6, 11.25, 14.3
    hy = H - 0.2
    for x, n, s in ((X1, '1', 'Parse into events'), (X2, '2', 'Map fields onto taxonomies'), (X3, '3', 'Score'), (X4, '4', 'Gate and match')):
        L.append(rf'\node[badge] at ({x + 0.17:.2f},{hy:.2f}) {{{n}}};\node[ttl] at ({x + 0.42:.2f},{hy:.2f}) {{{s}}};')
    for x in (X2 - 0.15, X3 - 0.15, X4 - 0.15):
        L.append(rf'\draw[grid, line width=0.6pt] ({x:.2f},0.05) -- ({x:.2f},{H - 0.45:.2f});')
        L.append(rf'\draw[-{{Stealth[length=1.6mm,width=1.4mm]}}, edgec, line width=0.7pt] ({x - 0.1:.2f},{hy:.2f}) -- ({x + 0.1:.2f},{hy:.2f});')

    # ---- 1: images, captions, events
    top = H - 0.5
    L.append(rf'\node[anchor=north west, inner sep=0pt] (t1) at (0.05,{top:.2f}) {{\includegraphics[width=1.8cm]{{img/f_t1.png}}}};')
    L.append(rf'\node[anchor=north west, inner sep=0pt] (t2) at (2.1,{top:.2f}) {{\includegraphics[width=1.8cm]{{img/f_t2.png}}}};')
    for nm in ('t1', 't2'):
        L.append(rf'\node[anchor=north west, fill=white, fill opacity=0.8, text opacity=1, font=\fs{{7}}, inner sep=1.2pt] at ({nm}.north west) {{$T_{nm[1]}$}};')
    row = lambda tg, e: (rf'${tg}$: {e["object"].replace("_", " ")}, {e["action"].replace("_", " ")}, '
                         rf'{(e["location"] or "--").replace("_", " ")}, {e["quantity"]}')
    refcap = 'a house is built at the top right corner of the scene'
    candcap = 'a building disappears and a house appears at the bottom right corner of the scene'
    assert refcap + ' .' == ref and candcap == pred
    box = r'line width=0.8pt, rounded corners=1.5pt, text width=4.2cm, inner sep=2.4pt, font=\fs{6.7}, align=left'
    L.append(rf"\node[{box}, draw=cRef, anchor=north west] (rb) at (0.05,{top - 1.95:.2f}) {{\textbf{{Reference}}\\[0.5pt]\textit{{``{refcap}''}}\\[2.5pt]{{\fs{{6.2}}{row('g_1', EXP['g1'])}}}}};")
    L.append(rf"\node[{box}, draw=cCand, anchor=north west] at ([yshift=-0.14cm]rb.south west) {{\textbf{{Candidate}}\\[0.5pt]\textit{{``{candcap}''}}\\[2.5pt]"
             rf"{{\fs{{6.2}}{row('p_1', EXP['p1'])}\\[0.5pt]{row('p_2', EXP['p2'])}}}}};")

    # ---- 2: the four taxonomies
    yt, yb = 3.25, 0.3
    L += mini_tree('object', X2 + 0.1, yt)
    L += mini_tree('action', X2 + 3.3, yt)
    L += mini_tree('location', X2 + 0.1, yb)
    L += mini_tree('quantity', X2 + 3.25, yb)
    nda = MINI['action']['nodes']
    a1 = (X2 + 3.3 + nda['appearance'][0], yt + nda['appearance'][1]); a2 = (X2 + 3.3 + nda['disappearance'][0], yt + nda['disappearance'][1])
    L.append(rf'\draw[{{Stealth[length=1.4mm]}}-{{Stealth[length=1.4mm]}}, cBad, line width=0.7pt, dashed] ({a1[0] + 0.15:.3f},{a1[1]:.3f}) -- ({a2[0] - 0.15:.3f},{a2[1]:.3f});')
    L.append(rf'\node[font=\fs{{7}}, inner sep=0pt] at ({(a1[0] + a2[0]) / 2:.3f},{a1[1] - 0.17:.3f}) {{\xmark}};')
    ly, lx = 0.12, X2 + 0.04
    items = [(0.0, r'\node[nd, fill=cCand, draw=cCandD] at (#,ly) {};', 'candidate'), (1.32, r'\node[nd, fill=cRef, draw=cRefD] at (#,ly) {};', 'reference'),
             (2.85, r'\draw[hullP, line width=2.3mm] (#-0.1,ly) -- (#+0.12,ly);', '$A(p)$'), (3.75, r'\draw[hullG, line width=2.3mm] (#-0.1,ly) -- (#+0.12,ly);', '$A(g)$'),
             (4.72, r'\node[rootnd] at (#,ly) {};', 'root'), (5.55, r'\draw[{Stealth[length=1.2mm]}-{Stealth[length=1.2mm]}, cBad, line width=0.6pt, dashed] (#-0.12,ly) -- (#+0.14,ly);', 'opposed')]
    for dx, mark, text in items:
        x = lx + dx
        L.append(mark.replace('#', f'{x:.2f}').replace('ly', f'{ly}'))
        L.append(rf'\node[lab, anchor=west] at ({x + 0.2:.2f},{ly}) {{{text}}};')

    # ---- 3: scores of one event pair
    x = X3 + 0.02
    L.append(rf'''\node[anchor=north west, inner sep=0pt, font=\fs{{7.4}}, align=left] at ({x:.2f},{top - 0.05:.2f}) {{%
$hP_k=\dfrac{{|A(p)\cap A(g)|}}{{|A(p)|}}$\\[9pt]
$hR_k=\dfrac{{|A(p)\cap A(g)|}}{{|A(g)|}}$\\[16pt]
$S_P=\dfrac{{\sum_k w_k\,hP_k}}{{\sum_k w_k}}$\\[3pt]\hspace*{{2.2em}}$=\mathbf{{{f2(sp2)}}}$\\[9pt]
$S_R=\dfrac{{\sum_k w_k\,hR_k}}{{\sum_k w_k}}$\\[3pt]\hspace*{{2.2em}}$=\mathbf{{{f2(sr2)}}}$}};''')
    L.append(rf'\node[anchor=north west, inner sep=0pt, font=\fs{{6.6}}, align=left] at ({x:.2f},{top - 5.1:.2f}) {{for $(p_2,g_1)$, over all fields $k$}};')

    # ---- 4: gate, matching, scores
    x = X4 + 0.02
    def mat(name, M, x0, y0):
        out = [rf'\node[anchor=west, font=\fs{{7.6}}, inner sep=0pt] at ({x0:.2f},{y0:.2f}) {{${name}$}};',
               rf'\node[font=\fs{{7}}, inner sep=0pt] at ({x0 + 0.9:.2f},{y0:.2f}) {{$g_1$}};']
        for i, rowv in enumerate(M):
            yy = y0 - 0.4 - 0.4 * i
            out.append(rf'\node[font=\fs{{7}}, inner sep=0pt] at ({x0 + 0.22:.2f},{yy:.2f}) {{$p_{i + 1}$}};')
            bx = 'draw=cRef, line width=1.0pt' if i == 1 else 'draw=grid, line width=0.5pt'
            txt = rf'\textbf{{{f2(rowv[0])}}}' if i == 1 else r'\xmark\,0'
            out.append(rf'\node[{bx}, minimum width=0.82cm, minimum height=0.33cm, inner sep=0pt, font=\fs{{7.2}}] at ({x0 + 0.9:.2f},{yy:.2f}) {{{txt}}};')
        return out
    assert SP[0, 0] == SR[0, 0] == 0.0
    L += mat('S_P', SP, x, top - 0.12)
    L += mat('S_R', SR, x + 1.6, top - 0.12)
    L.append(rf'\node[anchor=north west, inner sep=0pt, font=\fs{{6.8}}, align=left] at ({x:.2f},{top - 1.25:.2f}) {{\xmark\ gate: opposite changes\\[1pt]\textcolor{{cRef}}{{$\square$}} optimal one-to-one match}};')
    L.append(rf'''\node[anchor=north west, inner sep=0pt, font=\fs{{7.6}}, align=left] at ({x:.2f},{top - 2.2:.2f}) {{%
$\mathrm{{CE\text{{-}}P}}=\tfrac{{{f2(SP[1, 0])}+0}}{{2}}=\mathbf{{{f2(cep)}}}$\\[7pt]
$\mathrm{{CE\text{{-}}R}}=\tfrac{{{f2(SR[1, 0])}}}{{1}}=\mathbf{{{f2(cer)}}}$\\[7pt]
$\mathrm{{CE\text{{-}}F}}=\mathbf{{{f2(cef)}}}$}};''')
    L.append(rf'\node[anchor=north west, inner sep=0pt, font=\fs{{6.8}}, align=left] at ({x:.2f},{top - 4.2:.2f}) {{mean over the {len(refs)} references:\\ CE-F $=\mathbf{{{f2(mean_f)}}}$}};')
    L += [r'\end{tikzpicture}', r'\end{document}']
    return '\n'.join(L)


# ---- full taxonomies
ROW, DX = 0.29, 0.38
HILITE = dict(object='residential_building', action='constructed', location='upper_right', quantity='few')


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
    L = [rf'\node[anchor=south west, font=\fs{{8}}\bfseries, inner sep=0pt] at ({ox - 0.1:.3f},{oy - 0.08:.3f}) {{{ax.capitalize()}}};']
    A = t.ancestors(HILITE[ax])[1:]
    pts = []
    for k, n in enumerate(A):
        x, y = pos[n]
        if k:
            pts.append((pos[A[k - 1]][0], y))
        pts.append((x, y))
    if len(pts) == 1:
        L.append(rf'\fill[cRef, fill opacity=0.28] ({pts[0][0]:.3f},{pts[0][1]:.3f}) circle (1.6mm);')
    else:
        L.append(r'\draw[hullG] ' + ' -- '.join(f'({x:.3f},{y:.3f})' for x, y in pts) + ';')
    for n, d in rows:
        par = t.nodes[n].parent
        if par is not None:
            (px, py), (x, y) = pos[par], pos[n]
            L.append(rf'\draw[edge] ({px:.3f},{py - 0.105:.3f}) -- ({px:.3f},{y:.3f}) -- ({x - 0.105:.3f},{y:.3f});')
    for n, d in rows:
        x, y = pos[n]
        sty = 'rootnd' if n == t.root else ('nd, fill=cRef, draw=cRefD' if n == HILITE[ax] else 'nd')
        L.append(rf'\node[{sty}] at ({x:.3f},{y:.3f}) {{}};')
        L.append(rf'\node[lab, font=\fs{{7}}, anchor=west] at ({x + 0.13:.3f},{y:.3f}) {{{t.nodes[n].label}}};')
    return L, pos


def taxonomies_tex():
    L = [PRE, r'\begin{tikzpicture}[x=1cm,y=1cm]']
    top = 6.45
    OX = dict(object=0.15, action=4.6, location=8.75, quantity=12.15)
    for ax in OX:
        Lx, p = full_tree(ax, OX[ax], top)
        L += Lx
        if ax == 'action':
            pa = p
    xr = OX['action'] + 2.95
    for a, b in (('appearance', 'disappearance'), ('expansion', 'contraction')):
        (x1, y1), (x2, y2) = pa[a], pa[b]
        wa = {'appearance': 1.32, 'expansion': 1.22}[a]; wb = {'disappearance': 1.6, 'contraction': 1.34}[b]
        L.append(rf'\draw[{{Stealth[length=1.3mm]}}-{{Stealth[length=1.3mm]}}, cBad, line width=0.65pt, dashed, rounded corners=2pt] '
                 rf'({x1 + wa:.3f},{y1:.3f}) -- ({xr:.3f},{y1:.3f}) -- ({xr:.3f},{y2:.3f}) -- ({x2 + wb:.3f},{y2:.3f});')
        L.append(rf'\node[font=\fs{{7}}, inner sep=0pt] at ({xr + 0.16:.3f},{(y1 + y2) / 2:.3f}) {{\xmark}};')
    # legend
    lx, ly = OX['quantity'], top - 3.0
    L.append(rf'\node[rootnd] at ({lx:.2f},{ly:.2f}) {{}};\node[lab, font=\fs{{7}}, anchor=west] at ({lx + 0.2:.2f},{ly:.2f}) {{root (unspecified; not in $A$)}};')
    L.append(rf'\draw[hullG, line width=2.3mm] ({lx - 0.1:.2f},{ly - 0.42:.2f}) -- ({lx + 0.1:.2f},{ly - 0.42:.2f});\node[nd, fill=cRef, draw=cRefD] at ({lx + 0.1:.2f},{ly - 0.42:.2f}) {{}};'
             rf'\node[lab, font=\fs{{7}}, anchor=west] at ({lx + 0.3:.2f},{ly - 0.42:.2f}) {{node $v$ and its ancestor set $A(v)$}};')
    L.append(rf'\draw[{{Stealth[length=1.3mm]}}-{{Stealth[length=1.3mm]}}, cBad, line width=0.65pt, dashed] ({lx - 0.12:.2f},{ly - 0.84:.2f}) -- ({lx + 0.16:.2f},{ly - 0.84:.2f});'
             rf'\node[lab, font=\fs{{7}}, anchor=west] at ({lx + 0.3:.2f},{ly - 0.84:.2f}) {{opposed families (gate)}};')
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
    VAL['PAGE_SIZES'] = dict(value=dict(fig_framework=s1, fig_taxonomies=s2), source='pdfinfo')
    VAL['TREE_TITLES_hP_hR'] = dict(value={ax: fs2[ax] for ax in MINI}, source='event_field_scores_v2(p2, g1) via scripts/cvpr27_39 (checked there)')
    (OUT / 'figure_values.json').write_text(json.dumps(VAL, indent=1, default=float))
