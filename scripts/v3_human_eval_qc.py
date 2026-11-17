"""Quality control of the image-grounded human-study annotation files, run BEFORE any
metric-vs-human analysis. Uses the analyst-only key only for face validity (does a
rating drop on the axis a constructed error targets?), never to change a rating.

Checks, per annotator file:
  * format: scripts/make_annotation_sheets.py --check (values, NA policy, Task A/B columns,
    repeated-pattern and consecutive-repeat screens);
  * what the repeated 10-rating patterns are and which item types they sit on;
  * face validity: targeted-axis mean on each perturbation type vs the correct/paraphrase controls;
  * pairwise agreement between annotators (identical rows, exact per-axis, Task B) and a
    shifted-copy test (row i of one file equal to row i+k of another);
  * row-offset test: each annotator's ratings vs the other two annotators' mean on shifted rows (detects
    ratings typed against the wrong row);
  * Task B: preference on perturbation pairs, where one caption contains a constructed error; split by what
    the other caption is (correct original, paraphrase, coarse ancestor, or a sibling substitution, itself an object error).

Input : artifacts_cvpr_final/human_eval/image_grounded/annotation_sheets/annotations_ann{1,2,3}_full.csv
Output: results_v3/human_eval/qc/{qc_summary.json, face_validity.csv, agreement.csv,
        repeated_patterns.csv, pert_pair_preferences.csv, row_offset_test.csv}"""
import collections, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.evaluation.human_protocol import AXES

H = ROOT / 'artifacts_cvpr_final' / 'human_eval' / 'image_grounded'
OUT = ROOT / 'results_v3' / 'human_eval' / 'qc'; OUT.mkdir(parents=True, exist_ok=True)
ANN = ['ann1', 'ann2', 'ann3']
FILES = {a: H / 'annotation_sheets' / f'annotations_{a}_full.csv' for a in ANN}
D = {a: pd.read_csv(p, dtype=str, keep_default_na=False, encoding='utf-8-sig') for a, p in FILES.items()}
KEY = {json.loads(l)['item_id']: json.loads(l) for l in open(H / 'hidden' / 'answer_key.jsonl')}
num = lambda s: pd.to_numeric(s.replace('NA', np.nan), errors='coerce')
ptype = lambda i: KEY[i].get('perturbation_type') or ('real:' + str(KEY[i].get('system')))
summary = {'files': {a: str(p.relative_to(ROOT)) for a, p in FILES.items()}}

# 1. format check (the same checker the annotators were given)
summary['format_check'] = {}
for a, p in FILES.items():
    r = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'make_annotation_sheets.py'), '--check', str(p)], capture_output=True, text=True)
    summary['format_check'][a] = dict(exit_code=r.returncode, output=r.stdout.strip().splitlines())

# 2. repeated patterns and the item types they sit on
rows = []
for a, df in D.items():
    A = df[df.task == 'A'].copy(); A['pat'] = A[AXES].agg('|'.join, axis=1)
    for p, n in A.pat.value_counts().head(10).items():
        types = collections.Counter(A[A.pat == p].item_id.map(ptype)).most_common(5)
        rows.append(dict(annotator=a, pattern=p, n=int(n), item_types=json.dumps(types)))
    summary.setdefault('distinct_patterns', {})[a] = dict(n_rows=len(A), distinct=int(A.pat.nunique()), na_cells=int((A[AXES] == 'NA').sum().sum()))
pd.DataFrame(rows).to_csv(OUT / 'repeated_patterns.csv', index=False)

# 3. face validity: targeted axis per constructed error type vs controls
TARGET = {'P4_action_reversal': 'action', 'P5_location_swap': 'location', 'P6_count_corruption': 'quantity', 'P7_event_omission': 'completeness',
          'P8_hallucination': 'hallucination', 'P9_change_state_reversal': 'change_nochange', 'P3_sibling': 'object',
          'P1_ancestor': 'specificity', 'P2_distant_ancestor': 'specificity'}
rows = []
for a, df in D.items():
    A = df[df.task == 'A'].copy(); A['pt'] = A.item_id.map(ptype)
    ctrl = A[A.pt.isin(['P0_original', 'P0_paraphrase'])]
    for pt, ax in TARGET.items():
        g = A[A.pt == pt]
        rows.append(dict(annotator=a, perturbation_type=pt, targeted_axis=ax, n_items=len(g), n_NA=int((g[ax] == 'NA').sum()),
                         targeted_mean=num(g[ax]).mean(), control_mean=num(ctrl[ax]).mean(), difference=num(g[ax]).mean() - num(ctrl[ax]).mean(),
                         overall_mean=num(g.overall).mean(), control_overall_mean=num(ctrl.overall).mean()))
fv = pd.DataFrame(rows); fv.to_csv(OUT / 'face_validity.csv', index=False)

# 4. agreement + shifted-copy test
rows = []
M = {a: D[a][D[a].task == 'A'].set_index('item_id') for a in ANN}
for p, q in [('ann1', 'ann2'), ('ann1', 'ann3'), ('ann2', 'ann3')]:
    x, y = M[p], M[q].loc[M[p].index]
    same = (x[AXES].values == y[AXES].values).all(1)
    pats = collections.Counter(x[AXES][same].agg('|'.join, axis=1))
    bx = D[p][D[p].task == 'B'].set_index('item_id').preference; by = D[q][D[q].task == 'B'].set_index('item_id').preference.loc[bx.index]
    sx = x[AXES].agg('|'.join, axis=1).values; sy = y[AXES].agg('|'.join, axis=1).values  # same item order
    fx = D[p][D[p].task == 'A'][AXES].agg('|'.join, axis=1).values; fy = D[q][D[q].task == 'A'][AXES].agg('|'.join, axis=1).values  # file order
    n = len(fx)
    shift = {k: float(((fx[:n - k] == fy[k:]) if k >= 0 else (fx[-k:] == fy[:n + k])).mean()) for k in range(-40, 41)}
    best_k = max((k for k in shift if k != 0), key=lambda k: shift[k])
    r = dict(pair=f'{p}-{q}', identical_rows=int(same.sum()), identical_rows_share=float(same.mean()),
             identical_rows_correct_no_change_pattern=int(pats.get('NA|NA|NA|NA|NA|5|NA|5|5|5', 0)),
             task_B_same_preference=float((bx == by).mean()), shift0_identical=shift[0], best_nonzero_shift=best_k, best_nonzero_shift_identical=shift[best_k])
    for ax in AXES:
        r[f'exact_{ax}'] = float((x[ax] == y[ax]).mean())
    d = (num(x.overall) - num(y.overall)).abs().value_counts().sort_index()
    r['abs_overall_diff_counts'] = json.dumps({int(k): int(v) for k, v in d.items()})
    rows.append(r)
pd.DataFrame(rows).to_csv(OUT / 'agreement.csv', index=False)

# 5. Task B perturbation pairs: does the annotator prefer the caption WITHOUT the constructed error?
rows = []
ERR = {'P4_action_reversal', 'P5_location_swap', 'P6_count_corruption', 'P7_event_omission', 'P8_hallucination', 'P9_change_state_reversal'}
for a, df in D.items():
    B = df[df.task == 'B'].set_index('item_id')
    for pid, k in KEY.items():
        if k['kind'] != 'pair' or k['source'] != 'pert': continue
        ta, tb = k['perturbation_type_a'], k['perturbation_type_b']
        err_side = 'A' if ta in ERR and tb not in ERR else ('B' if tb in ERR and ta not in ERR else None)
        if err_side is None: continue
        pref = B.loc[pid, 'preference']
        other = tb if err_side == 'A' else ta
        rows.append(dict(annotator=a, pair_id=pid, category=k['category'], error_side=err_side, other_type=other, preference=pref,
                         prefers_error_free=float(pref in ('A', 'B') and pref != err_side) if pref in ('A', 'B') else np.nan))
pp = pd.DataFrame(rows); pp.to_csv(OUT / 'pert_pair_preferences.csv', index=False)
summary['pert_pairs'] = {a: dict(n=int((pp.annotator == a).sum()), strict_choices=int(pp[pp.annotator == a].prefers_error_free.notna().sum()),
                                 share_preferring_error_free=float(pp[pp.annotator == a].prefers_error_free.mean()),
                                 pref_counts=pp[pp.annotator == a].preference.value_counts().to_dict()) for a in ANN}
summary['pert_pairs_by_other_side'] = {a: {o: dict(n=int(len(g)), strict=int(g.prefers_error_free.notna().sum()), share_preferring_other=float(g.prefers_error_free.mean()))
                                            for o, g in pp[pp.annotator == a].groupby('other_type')} for a in ANN}
# 6. row-offset test: is each annotator's Overall aligned with its own items? Spearman of the annotator's rating on row i
#    with the mean of the other two annotators on row i+k (file order; all files share the item order)
from scipy.stats import spearmanr
A_ = {a: D[a][D[a].task == 'A'].reset_index(drop=True) for a in ANN}
assert all((A_[ANN[0]].item_id == A_[a].item_id).all() for a in ANN)
rows = []
for a in ANN:
    others = [b for b in ANN if b != a]
    for ax in ('overall', 'object', 'action', 'change_nochange'):
        o = num(A_[a][ax]).to_numpy(); oth = np.nanmean(np.vstack([num(A_[b][ax]).to_numpy() for b in others]), axis=0); n = len(o)
        for k in range(-6, 7):
            x, y = (o[:n - k], oth[k:]) if k >= 0 else (o[-k:], oth[:n + k])
            m = ~np.isnan(x) & ~np.isnan(y)
            rows.append(dict(annotator=a, axis=ax, offset=k, n=int(m.sum()), spearman_with_others_mean=float(spearmanr(x[m], y[m]).statistic)))
ro = pd.DataFrame(rows); ro.to_csv(OUT / 'row_offset_test.csv', index=False)
summary['row_offset_best'] = {a: {ax: int(g.loc[g.spearman_with_others_mean.idxmax(), 'offset']) for ax, g in ro[ro.annotator == a].groupby('axis')} for a in ANN}
summary['timing_columns_filled'] = {a: int(((D[a].seconds_spent != '') | (D[a].timestamp_utc != '')).sum()) for a in ANN}
(OUT / 'qc_summary.json').write_text(json.dumps(summary, indent=1, default=float))
print(json.dumps({k: v for k, v in summary.items() if k != 'format_check'}, indent=1, default=float))
print(fv.pivot_table(index=['perturbation_type', 'targeted_axis'], columns='annotator', values='difference').round(2).to_string())
print(pd.DataFrame(rows[:0]) if False else pd.read_csv(OUT / 'agreement.csv')[['pair', 'identical_rows', 'identical_rows_correct_no_change_pattern', 'task_B_same_preference', 'best_nonzero_shift_identical', 'exact_overall', 'exact_change_nochange']].to_string())
