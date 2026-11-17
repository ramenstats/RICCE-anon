"""Compact markdown table: score drop from P0_original (pooled 3 datasets, paired bootstrap 95% CI) for the
perturbations where event-level structure matters, plus real-caption binding / source-target diagnostics."""
from pathlib import Path
import pandas as pd
R = Path(__file__).resolve().parents[1] / 'results_cvpr_final' / 'baselines'
M = ['ricce', 'kiritchenko_micro', 'kiritchenko_macro', 'lca_f', 'hcm', 'taxaware_single', 'taxaware_axisflat']
o = pd.read_csv(R / 'perturbation_summary.csv'); a = o[o.dataset == 'ALL']
lines = ['## Score drop vs P0_original (F; pooled LEVIR+SECOND+DUBAI E04 sets; [95% paired-bootstrap CI], 2000 resamples)', '',
         '| type | n | ' + ' | '.join(M) + ' |', '|' + '---|' * (len(M) + 2)]
for t in sorted(a.type.unique()):
    if t == 'P0_original': continue
    s = a[a.type == t].set_index('metric')
    lines.append(f'| {t} | {int(s.n.iloc[0])} | ' + ' | '.join(
        f"{s.loc[m,'drop']:.3f} [{s.loc[m,'drop_ci_lo']:.3f},{s.loc[m,'drop_ci_hi']:.3f}]" for m in M) + ' |')
d = pd.read_csv(R / 'diagnostics_summary.csv')
lines += ['', '## Real-caption diagnostics (mean F; fraction of items scored exactly 1.0 in parentheses)', '',
          '| group | n | ' + ' | '.join(M) + ' |', '|' + '---|' * (len(M) + 2)]
for g in ['identity', 'clause_order', 'bind_swap', 'st_swap']:
    s = d[d.group == g].set_index('metric')
    lines.append(f'| {g} | {int(s.n.iloc[0])} | ' + ' | '.join(f"{s.loc[m,'mean_F']:.3f} ({s.loc[m,'frac_F_eq_1']:.2f})" for m in M) + ' |')
(R / 'compact_table.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))
