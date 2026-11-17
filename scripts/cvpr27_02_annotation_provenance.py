"""CVPR'27, H0: objective, outcome-blind description of both annotation rounds.

Only format and procedure are checked here; no metric score and no constructed-error
label is read, so nothing in this script can depend on whether ratings favour any metric.
For each returned file (round 1 = annotation_sheets/rejected_2026-10-26, round 2 =
collected/*_2026-10-29.csv):
  * file facts: SHA-256, size, modification time, rows, annotator ids;
  * format: every Task A cell is 1-5 or NA (NA only where the guidelines allow it), every
    Task B row has a preference in {A, B, Tie, CannotDetermine}, Task B rows leave the
    rating columns empty, Task A rows leave preference empty;
  * procedure: number of distinct 10-rating rows, longest run of identical consecutive
    rows, NA cells on scales where NA is allowed;
  * independence: share of Task A rows whose 10 ratings are identical between two
    annotators' files, and between the two rounds of the same annotator.
Outputs: submission_cvpr27_new_version/02_HUMAN_AUDIT/{file_manifest.csv, round_comparison.csv,
         cross_file_identity.csv}"""
import csv, datetime, hashlib, itertools, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.evaluation.human_protocol import AXES, NA_AXES
H = ROOT / 'artifacts_cvpr_final' / 'human_eval' / 'image_grounded'
OUT = ROOT / 'submission_cvpr27_new_version' / '02_HUMAN_AUDIT'; OUT.mkdir(parents=True, exist_ok=True)
FILES = {('round1', a): p for a in ('ann1', 'ann2', 'ann3') for p in (H / 'annotation_sheets' / 'rejected_2026-10-26').glob(f'annotations_{a}_full*.csv')}
FILES |= {('round2', a): H / 'collected' / f'annotations_{a}_2026-10-29.csv' for a in ('ann1', 'ann2', 'ann3')}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    return list(csv.DictReader(open(p, encoding='utf-8-sig', newline='')))


man, comp, pats = [], [], {}
for (rnd, ann), p in sorted(FILES.items()):
    rows = load(p)
    A_ = [r for r in rows if r['task'] == 'A']; B_ = [r for r in rows if r['task'] == 'B']
    man.append(dict(round=rnd, annotator=ann, path=str(p.relative_to(ROOT)), sha256=sha(p), bytes=p.stat().st_size,
                    mtime=datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec='seconds'),
                    rows=len(rows), task_A_rows=len(A_), task_B_rows=len(B_), annotator_ids='|'.join(sorted({r['annotator_id'] for r in rows}))))
    bad_val = sum(1 for r in A_ for a in AXES if r[a].strip().upper() not in {'1', '2', '3', '4', '5', 'NA'})
    na_wrong = sum(1 for r in A_ for a in AXES if r[a].strip().upper() == 'NA' and a not in NA_AXES)
    na_allowed = sum(1 for r in A_ for a in AXES if r[a].strip().upper() == 'NA' and a in NA_AXES)
    b_bad = sum(1 for r in B_ if r['preference'].strip() not in {'A', 'B', 'Tie', 'CannotDetermine'})
    b_rated = sum(1 for r in B_ if any(r[a].strip() for a in AXES))
    a_pref = sum(1 for r in A_ if r['preference'].strip())
    P = [tuple(r[a].strip().upper() for a in AXES) for r in A_]
    pats[(rnd, ann)] = dict(zip([r['item_id'] for r in A_], P))
    run = best = 1
    for i in range(1, len(P)):
        run = run + 1 if P[i] == P[i - 1] else 1; best = max(best, run)
    comp.append(dict(round=rnd, annotator=ann, invalid_rating_cells=bad_val, na_on_scales_without_na=na_wrong,
                     na_cells_on_na_scales=na_allowed, taskB_invalid_or_empty_preference=b_bad, taskB_rows_with_ratings=b_rated,
                     taskA_rows_with_preference=a_pref, distinct_rating_rows=len(set(P)), distinct_share=round(len(set(P)) / max(1, len(P)), 3),
                     longest_identical_run=best, timing_columns_filled=sum(1 for r in rows if r['seconds_spent'].strip() or r['timestamp_utc'].strip())))
pd.DataFrame(man).to_csv(OUT / 'file_manifest.csv', index=False)
pd.DataFrame(comp).to_csv(OUT / 'round_comparison.csv', index=False)

ident = []
for x, y in itertools.combinations(sorted(pats), 2):
    common = set(pats[x]) & set(pats[y])
    same = sum(pats[x][i] == pats[y][i] for i in common)
    ident.append(dict(file_1='/'.join(x), file_2='/'.join(y), common_items=len(common), identical_rating_rows=same,
                      share_identical=round(same / max(1, len(common)), 3)))
pd.DataFrame(ident).to_csv(OUT / 'cross_file_identity.csv', index=False)
print(pd.DataFrame(comp).to_string()); print(pd.DataFrame(ident).to_string())
