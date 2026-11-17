"""Fig. 1 (motivation), version 2 (2026-11-11): less text, all text black, no LLM judge, a clearer (b)
and a simpler (c). The values are computed and checked by the value part of scripts/cvpr27_22_make_overview_figures.py,
executed here unchanged; CVPR2027_Figures/ (version 1) is not changed.

  (a) human-study pair P0078: the reference, caption A (one-word reversal) and caption B (correct but coarser), and one
      table: BLEU-4, CIDEr-D, CSM-4, BERTScore, CE-P, CE-R, CE-F and the annotators' votes, with the caption each prefers.
  (b) RICCE-Bench semantic-validity suite (16,538 pairs): % of pairs in which a metric scores the contradictory caption
      at least as high as the correct one (= 1 - strict accuracy of Tab. 1; ties count as wrong). RICCE: 0% (gate).
  (c) Spearman rho with the annotators' mean overall rating on all 896 scorable rated captions, 95% cluster-bootstrap CI (Tab. 2).
With --trial, also a trial build of a local copy of the paper sources (CVPR2027_Final_Submission/) with this figure as Fig. 1.
Outputs -> CVPR2027_Motivation_Figure_v2/
"""
import json, re, shutil, subprocess, sys, tempfile
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / 'scripts' / 'cvpr27_22_make_overview_figures.py'
MARK = '# ------------------------------------------------------------------ images'
src = V1.read_text(); assert src.count(MARK) == 1
ns = {'__file__': str(V1), '__name__': 'cvpr27_22_values'}
exec(compile(src.split(MARK)[0], str(V1), 'exec'), ns)              # values + checks of version 1
M, FAIL, hv, VAL = ns['M'], ns['FAIL'], ns['hv'], ns['VAL']
TECTONIC = ns['TECTONIC']
OUT = ROOT / 'CVPR2027_Motivation_Figure_v2'; SRC = OUT / 'src'
(SRC / 'img').mkdir(parents=True, exist_ok=True)
SUB = ROOT / 'CVPR2027_Final_Submission'
for t in ('t1', 't2'):
    Image.open(M[t]).convert('RGB').save(SRC / 'img' / f'm_{t}.png')

# ---- values
A, B = M['a'], M['b']                                               # a = reversed caption (A), b = coarse caption (B)
assert A['ptype'] == 'P4_action_reversal' and all(v == 'B' for v in M['votes'].values())
COLS = [('BLEU-4', 'bleu4', 2), ('CIDEr-D', 'cider_d', 2), ('CSM-4', 'csm4', 2), ('BERTScore', 'bertscore', 3),
        ('CE-P', 'ce_p', 2), ('CE-R', 'ce_r', 2), ('CE-F', 'ce_f', 2)]
for name, k, _ in COLS:                                             # every string metric prefers A, every CE score prefers B
    assert (A[k] > B[k]) == (not name.startswith('CE')), name
votes_A = sum(v == 'A' for v in M['votes'].values()); votes_B = sum(v == 'B' for v in M['votes'].values())
sv = {r.split(',')[0]: r.split(',') for r in (ROOT / 'results_cvpr_final' / 'ordering' / 'semantic_validity_ordering.csv').read_text().splitlines()[1:]}
assert float(sv['CE-F'][2]) == 1.0 and int(sv['CE-F'][1]) == 16538                   # RICCE fails 0 of 16,538 (its gate)
C_ORDER = ['ce_f1', 'sbert', 'bleu4', 'cider_d', 'bertscore_f', 'spice_f', 'csm4']      # LLM judge removed
C_NAME = dict(ce_f1='CE-F (ours)', sbert='Sentence-BERT', bleu4='BLEU-4', bertscore_f='BERTScore', cider_d='CIDEr-D', csm4='CSM-4', spice_f='SPICE')
C = [(C_NAME[m], *hv[m, 'all'][:3]) for m in C_ORDER]
assert all(hv[m, 'all'][3] == 896 for m in C_ORDER)
assert C[0][1] == max(c[1] for c in C) and [c[1] for c in C[1:]] == sorted([c[1] for c in C[1:]], reverse=True)
VALS = dict(caption_A=A['caption'], caption_B=B['caption'], reference=M['ref'], votes=M['votes'],
            table={name: [A[k], B[k]] for name, _, k in [(n, None, k) for n, k, _ in COLS]},
            b_wrong_share=FAIL, b_ricce_strict_accuracy=1.0, c_spearman_all={n: [r, lo, hi] for n, r, lo, hi in C})
SRCS = dict(caption_A=VAL['M_REV_caption']['source'], table=ns['SRCP'], b_wrong_share=VAL['B_FAIL']['source'],
            b_ricce_strict_accuracy='results_cvpr_final/ordering/semantic_validity_ordering.csv (CE-F strict)',
            c_spearman_all=VAL['C_SPEARMAN']['source'], votes=VAL['M_VOTES']['source'])

# ---- LaTeX
PRE = r"""\documentclass[border=1pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{amsmath,amssymb,pifont,array,graphicx,booktabs}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,calc}
% all text black; colour only on marks: blue = RICCE, grey = other metrics; red/green only with a cross/check icon
\definecolor{ink}{HTML}{000000}\definecolor{cOurs}{HTML}{2A78D6}\definecolor{cBase}{HTML}{8A8983}
\definecolor{grid}{HTML}{E1E0D9}\definecolor{axisc}{HTML}{9C9B95}
\definecolor{cBad}{HTML}{D03B3B}\definecolor{cGood}{HTML}{0C8A0C}\definecolor{hlBad}{HTML}{F8D4D4}\definecolor{hlGood}{HTML}{D2EDD2}
\newcommand{\xmark}{{\color{cBad}\ding{55}}}
\newcommand{\cmark}{{\color{cGood}\ding{51}}}
\newcommand{\fs}[1]{\fontsize{#1}{#1}\selectfont\setlength{\baselineskip}{1.15\dimexpr#1pt\relax}}
\newcommand{\method}{\textsc{RICCE}}
\newcommand{\bench}{\textsc{RICCE-Bench}}
\tikzset{ttl/.style={anchor=north west, font=\fs{8}, inner sep=0pt, align=left}}
\begin{document}
"""


def esc(s):
    return s.replace('&', r'\&').replace('%', r'\%')


def cap_tex(c, word, hl):
    c = c[:-2] + '.' if c.endswith(' .') else c
    assert word in c
    return c.replace(word, r'{\setlength{\fboxsep}{0.8pt}\colorbox{%s}{\textbf{%s}}}' % (hl, word), 1)


def fmt(x, d):
    return f'{x:.{d}f}'


def figure_tex():
    H = 4.75
    L = [PRE, r'\begin{tikzpicture}[x=1cm,y=1cm]', rf'\useasboundingbox (0,0) rectangle (17.4,{H});']
    # ---------------- (a)
    L.append(rf'\node[ttl] at (0,{H:.2f}) {{\textbf{{(a) One image pair, two captions}}}};')
    top = H - 0.42
    L.append(rf'\node[anchor=north west, inner sep=0pt] (t1) at (0,{top:.2f}) {{\includegraphics[width=1.82cm]{{img/m_t1.png}}}};')
    L.append(rf'\node[anchor=north west, inner sep=0pt] (t2) at (1.92,{top:.2f}) {{\includegraphics[width=1.82cm]{{img/m_t2.png}}}};')
    for nm, lab in (('t1', 'before'), ('t2', 'after')):
        L.append(rf'\node[anchor=north west, fill=white, fill opacity=0.85, text opacity=1, font=\fs{{7}}, inner sep=1.2pt] at ({nm}.north west) {{$T_{nm[1]}$ {lab}}};')
    box = r'anchor=north west, rounded corners=1.5pt, line width=0.7pt, text width=5.25cm, inner sep=2.2pt, font=\fs{7}, align=left'
    ref = M['ref'][:-2] + '.' if M['ref'].endswith(' .') else M['ref']
    L.append(rf'\node[{box}, draw=axisc] (rf) at (3.92,{top:.2f}) {{\textbf{{Reference:}} \textit{{{esc(ref)}}}}};')
    L.append(rf'\node[{box}, draw=cBad] (ca) at ([yshift=-0.1cm]rf.south west) {{\textbf{{A}} \xmark\ \textit{{{cap_tex(A["caption"], "demolished", "hlBad")}}}}};')
    L.append(rf'\node[{box}, draw=cGood] (cb) at ([yshift=-0.1cm]ca.south west) {{\textbf{{B}} \cmark\ \textit{{{cap_tex(B["caption"], "built environment", "hlGood")}}}}};')
    # score table
    hdr = ' & '.join(rf'\textbf{{{n}}}' if n == 'CE-F' else n for n, _, _ in COLS)
    rowA = ' & '.join(fmt(A[k], d) for _, k, d in COLS)
    rowB = ' & '.join(fmt(B[k], d) for _, k, d in COLS)
    pref = ' & '.join((r'A\,\xmark' if A[k] > B[k] else r'B\,\cmark') for _, k, _ in COLS)
    L.append(rf'''\node[anchor=north west, inner sep=0pt, font=\fs{{7}}] at (0,1.62) {{%
\setlength{{\tabcolsep}}{{3.6pt}}\begin{{tabular}}{{@{{}}l|cccc|ccc|c@{{}}}}
 & \multicolumn{{4}}{{c|}}{{word and embedding metrics}} & \multicolumn{{3}}{{c|}}{{\method}} & \\
caption & {hdr} & people\\\hline
A \xmark\ reversed & {rowA} & {votes_A} of 3\\
B \cmark\ coarser & {rowB} & {votes_B} of 3\\\hline
prefers & {pref} & B\,\cmark\\
\end{{tabular}}}};''')

    # ---------------- (b)
    x0 = 9.5
    L.append(rf'\node[ttl] at ({x0:.2f},{H:.2f}) {{\textbf{{(b) Metrics prefer contradictions}}}};')
    L.append(rf'\node[anchor=north west, inner sep=0pt, font=\fs{{6.6}}, text width=3.9cm, align=left] at ({x0:.2f},{H - 0.36:.2f}) '
             r'{\% of 16{,}538 pairs where a contradictory caption scores $\geq$ the correct one};')
    bx, bw, by0, bh = x0 + 1.32, 2.2, H - 1.2, 0.3            # bar origin, width for 60%, first bar y, step
    for g in (0, 0.2, 0.4, 0.6):
        xx = bx + bw * g / 0.6
        L.append(rf'\draw[grid, line width=0.4pt] ({xx:.3f},{by0 + 0.18:.3f}) -- ({xx:.3f},{by0 - bh * (len(FAIL) - 1) - 0.18:.3f});')
        L.append(rf'\node[font=\fs{{6.4}}, anchor=north, inner sep=1pt] at ({xx:.3f},{by0 - bh * (len(FAIL) - 1) - 0.2:.3f}) {{{int(g * 100)}\%}};')
    for i, (name, v) in enumerate(FAIL):
        yy = by0 - i * bh
        L.append(rf'\node[font=\fs{{7}}, anchor=east, inner sep=1pt] at ({bx - 0.05:.3f},{yy:.3f}) {{{name}}};')
        L.append(rf'\fill[cBase] ({bx:.3f},{yy - 0.1:.3f}) rectangle ({bx + bw * v / 0.6:.3f},{yy + 0.1:.3f});')
        L.append(rf'\node[font=\fs{{7}}, anchor=west, inner sep=1pt] at ({bx + bw * v / 0.6:.3f},{yy:.3f}) {{{round(v * 100):d}\%}};')
    yb = by0 - bh * (len(FAIL) - 1) - 0.66
    L.append(rf'\node[font=\fs{{7}}, anchor=east, inner sep=1pt, ] at ({bx - 0.05:.3f},{yb:.3f}) {{\textbf{{\method}}}};')
    L.append(rf'\node[font=\fs{{7}}, anchor=west, inner sep=1pt] at ({bx - 0.02:.3f},{yb:.3f}) {{\textbf{{0\%}} (its gate)}};')

    # ---------------- (c)
    x0 = 13.75
    L.append(rf'\node[ttl] at ({x0:.2f},{H:.2f}) {{\textbf{{(c) Agreement with people}}}};')
    L.append(rf'\node[anchor=north west, inner sep=0pt, font=\fs{{6.6}}, text width=3.6cm, align=left] at ({x0:.2f},{H - 0.36:.2f}) '
             r'{Spearman $\rho$ with human ratings of 896 captions (95\% CI)};')
    cx, cw, cy0, ch = x0 + 1.42, 1.85, H - 1.2, 0.3           # bar origin, width for rho = 0.8
    for g in (0, 0.2, 0.4, 0.6, 0.8):
        xx = cx + cw * g / 0.8
        L.append(rf'\draw[grid, line width=0.4pt] ({xx:.3f},{cy0 + 0.18:.3f}) -- ({xx:.3f},{cy0 - ch * (len(C) - 1) - 0.18:.3f});')
        L.append(rf'\node[font=\fs{{6.4}}, anchor=north, inner sep=1pt] at ({xx:.3f},{cy0 - ch * (len(C) - 1) - 0.2:.3f}) {{{g:g}}};')
    for i, (name, r, lo, hi) in enumerate(C):
        yy = cy0 - i * ch
        ours = i == 0
        col = 'cOurs' if ours else 'cBase'
        lab = ('\\textbf{' + name + '}') if ours else name
        val = ('\\textbf{' + f'{r:.2f}' + '}') if ours else f'{r:.2f}'
        L.append(rf'\node[font=\fs{{7}}, anchor=east, inner sep=1pt] at ({cx - 0.05:.3f},{yy:.3f}) {{{lab}}};')
        L.append(rf'\fill[{col}] ({cx:.3f},{yy - 0.1:.3f}) rectangle ({cx + cw * r / 0.8:.3f},{yy + 0.1:.3f});')
        L.append(rf'\draw[ink, line width=0.5pt] ({cx + cw * lo / 0.8:.3f},{yy:.3f}) -- ({cx + cw * hi / 0.8:.3f},{yy:.3f});')
        L.append(rf'\node[font=\fs{{7}}, anchor=west, inner sep=1pt] at ({cx + cw * hi / 0.8 + 0.02:.3f},{yy:.3f}) {{{val}}};')
    L += [r'\end{tikzpicture}', r'\end{document}']
    return '\n'.join(L)


CAPTION = r'''\captionof{figure}{\textbf{Why change captions need a different metric.} (a)~Two captions of one LEVIR-CC image pair from our human study, scored against the reference the study used. Caption A reverses the change with one word; caption B is correct but coarser. All three annotators chose B, yet BLEU-4, CIDEr-D, CSM-4 and BERTScore prefer A. \method prefers B, and its precision and recall (\cep\ 1.00, \cer\ 0.83) show that B is correct but less specific. We selected this pair among the four study pairs in which all annotators rejected an action reversal that BLEU-4, CIDEr-D and BERTScore preferred. (b)~On \bench, these metrics score a contradictory caption at least as high as the correct one in 22--49\% of 16{,}538 pairs. (c)~Rank correlation with the annotators' ratings; prior hierarchical metrics computed on \method's parses reach 0.58--0.60 (\cref{tab:human}).}'''


def build():
    (SRC / 'fig_motivation.tex').write_text(figure_tex())
    r = subprocess.run([TECTONIC, '-X', 'compile', 'fig_motivation.tex', '--outdir', str(OUT)], cwd=SRC, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:]); raise SystemExit('LaTeX failed')
    subprocess.run(['pdftoppm', '-png', '-r', '600', '-singlefile', str(OUT / 'fig_motivation.pdf'), str(OUT / 'fig_motivation')], check=True)
    info = subprocess.run(['pdfinfo', str(OUT / 'fig_motivation.pdf')], capture_output=True, text=True).stdout
    return [l for l in info.splitlines() if l.startswith('Page size')][0]


def trial():
    def once(t, a, b):
        assert t.count(a) == 1, a[:60]
        return t.replace(a, b)

    def compile_(W, B, extra=0):
        if extra:
            c = (W / 'sec' / '5_conclusion.tex').read_text()
            (W / 'sec' / '5_conclusion.tex').write_text(once(c, '\\label{end-of-body}', '\\par\\vspace*{%d\\baselineskip}\\label{end-of-body}' % extra))
        r = subprocess.run([TECTONIC, '-o', str(B), '--keep-intermediates', '--keep-logs', 'main.tex'], cwd=W, capture_output=True, text=True)
        if extra:
            (W / 'sec' / '5_conclusion.tex').write_text(c)
        if r.returncode:
            print(r.stderr[-3000:]); raise SystemExit('trial build failed')
        aux = (B / 'main.aux').read_text()
        return int(re.search(r'\\newlabel\{end-of-body\}\{\{[^}]*\}\{(\d+)\}', aux).group(1)), (B / 'main.log').read_text(errors='replace')

    def slack(W, B):
        lo, hi = 0, 16
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if compile_(W, B, mid)[0] <= 8 else (lo, mid - 1)
        return lo

    with tempfile.TemporaryDirectory(dir=str(OUT)) as tmp:
        W = Path(tmp) / 'paper'
        shutil.copytree(SUB, W, ignore=shutil.ignore_patterns('*.zip'))
        shutil.copy2(OUT / 'fig_motivation.pdf', W / 'figures' / 'fig_motivation.pdf')
        m = (W / 'main.tex').read_text()
        i = m.index('\\captionof{figure}{\\textbf{Why change captions'); j = m.index('\\label{fig:teaser}')
        m = m[:i] + CAPTION + '\n' + m[j:]
        (W / 'main.tex').write_text(m)
        B = Path(tmp) / 'b'; B.mkdir()
        end, log = compile_(W, B)
        shutil.copy2(B / 'main.pdf', OUT / 'trial_main_with_new_fig1.pdf')
        undef = re.findall(r"(?:Reference|Citation) `([^']+)' .*undefined", log)
        over = len(re.findall(r'Overfull \\hbox', log))
        sl = slack(W, B)
    return [f'trial build of CVPR2027_Final_Submission with this figure as Fig. 1 and the new caption',
            f'body ends on page {end} (limit 8)', f'undefined references/citations: {undef or "none"}', f'overfull hboxes: {over}',
            f'spare body lines on page 8 (tectonic/XeTeX): {sl}']


if __name__ == '__main__':
    size = build(); print('fig_motivation', size)
    (OUT / 'figure_values.json').write_text(json.dumps(dict(values=VALS, sources=SRCS, page_size=size), indent=1, default=float))
    (OUT / 'snippet_for_main.tex').write_text('%% Fig. 1, version 2: copy fig_motivation.pdf into figures/ (it replaces the old file) and replace the\n'
                                              '%% \\captionof{figure}{...} line of the teaser in main.tex with:\n' + CAPTION + '\n')
    if '--trial' in sys.argv:
        rep = trial(); (OUT / 'trial_build_report.txt').write_text('\n'.join(rep) + '\n'); print('\n'.join(rep))
