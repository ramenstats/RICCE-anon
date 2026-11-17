"""Blank, spreadsheet-editable annotation sheets for the image-grounded human study.

  python scripts/make_annotation_sheets.py            # write sheets for ann1..ann3, pilot + full
  python scripts/make_annotation_sheets.py --check F  # validate a filled sheet F

Sheets go to artifacts_cvpr_final/human_eval/image_grounded/annotation_sheets/.
Every rating cell is EMPTY: no rating of any kind is generated here. Column
order: the three key columns, read-only context columns (images, candidate,
references), then the rating columns of annotation_template.csv. The analysis
(ricce/evaluation/human_analysis.py) reads columns by name, so the context
columns are ignored. Item order is the file order (the HTML tool shuffles per
annotator; here annotators may work in any order).
"""
import argparse, csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.evaluation.human_protocol import AXES, NA_AXES, CSV_COLUMNS

D = ROOT / 'artifacts_cvpr_final' / 'human_eval' / 'image_grounded'
OUT = D / 'annotation_sheets'
CTX = ['dataset', 'image_id', 'T1_image', 'T2_image', 'candidate_or_A', 'candidate_B', 'references_hint']
RATE = CSV_COLUMNS[3:]            # axes..., preference, notes, timestamp_utc, seconds_spent
HEADER = CSV_COLUMNS[:3] + CTX + RATE


def jl(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]


def rows_for(annotator, items, pairs):
    out = []
    for r in items:
        out.append({'annotator_id': annotator, 'task': 'A', 'item_id': r['item_id'], 'dataset': r['dataset'], 'image_id': r['image_id'],
                    'T1_image': '../' + r['t1_thumb'], 'T2_image': '../' + r['t2_thumb'], 'candidate_or_A': r['candidate'], 'candidate_B': '',
                    'references_hint': ' | '.join(r['references'])})
    for r in pairs:
        out.append({'annotator_id': annotator, 'task': 'B', 'item_id': r['pair_id'], 'dataset': r['dataset'], 'image_id': r['image_id'],
                    'T1_image': '../' + r['t1_thumb'], 'T2_image': '../' + r['t2_thumb'], 'candidate_or_A': r['candidate_a'], 'candidate_B': r['candidate_b'],
                    'references_hint': ' | '.join(r['references'])})
    return out


def write(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=HEADER); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in HEADER})


def check(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig', newline='')))
    miss = [c for c in CSV_COLUMNS[:3] + RATE[:-3] if rows and c not in rows[0]]
    if miss:
        print('MISSING COLUMNS', miss); return 1
    bad, empty = [], 0
    ids = {r['annotator_id'] for r in rows}
    for r in rows:
        iid = r['item_id']
        if r['task'] == 'A':
            for a in AXES:
                v = r[a].strip()
                if v == '':
                    bad.append((iid, a, 'EMPTY (type NA if allowed)')); continue
                if v.upper() == 'NA':
                    if a not in NA_AXES: bad.append((iid, a, 'NA not allowed on this axis'))
                elif v not in {'1', '2', '3', '4', '5'}:
                    bad.append((iid, a, f'invalid value {v!r}'))
        else:
            v = r['preference'].strip()
            if v not in {'A', 'B', 'Tie', 'CannotDetermine'}:
                bad.append((iid, 'preference', f'invalid/empty {v!r}'))
    # --- stricter checks (a returned file failed these on 2026-10-26) ---
    warn = []
    A_ = [r for r in rows if r['task'] == 'A']; B_ = [r for r in rows if r['task'] == 'B']
    for r in B_:
        if any(r[a].strip() for a in AXES):
            bad.append((r['item_id'], 'ratings', 'Task B row must leave the ten rating columns EMPTY (only preference)'))
    for r in A_:
        if r['preference'].strip():
            bad.append((r['item_id'], 'preference', 'Task A row must leave preference EMPTY'))
    if A_:
        pats = [tuple(r[a].strip() for a in AXES) for r in A_]
        distinct = len(set(pats)); top = max(pats.count(p) for p in set(pats))
        n_na = sum(1 for r in A_ for a in AXES if r[a].strip().upper() == 'NA')
        warn.append(f'Task A: {len(A_)} rows, {distinct} distinct 10-rating patterns, most repeated pattern {top}x, {n_na} NA cells')
        if distinct < 0.5 * len(A_):
            bad.append(('ALL', 'patterns', f'only {distinct} distinct rating patterns over {len(A_)} rows: ratings look copied / filled down, not given item by item'))
        if n_na == 0 and len(A_) > 100:
            bad.append(('ALL', 'NA', 'no NA anywhere: items with no location/quantity/object, or no-change scenes, need NA'))
        run = 1
        for i in range(1, len(pats)):
            run = run + 1 if pats[i] == pats[i - 1] else 1
            if run >= 4:
                bad.append((A_[i]['item_id'], 'repeat', 'same 10 ratings on 4+ consecutive rows')); break
    if B_:
        c = {k: sum(1 for r in B_ if r['preference'].strip() == k) for k in ('A', 'B', 'Tie', 'CannotDetermine')}
        warn.append(f'Task B: {len(B_)} rows, preference counts {c}')
    for r in rows:
        if r['seconds_spent'].strip() == '' and r['timestamp_utc'].strip() == '':
            warn.append('seconds_spent / timestamp_utc are empty (optional, but they are the only trace that rows were viewed)'); break
    print(f'{path}: {len(rows)} rows, annotator ids {sorted(ids)}, {len(bad)} problems')
    for w in warn: print('  info:', w)
    for b in bad[:40]: print('  ', b)
    if len(bad) > 40: print(f'   ... {len(bad)-40} more')
    return 1 if bad or len(ids) != 1 else 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--check'); a = ap.parse_args()
    if a.check:
        sys.exit(check(a.check))
    OUT.mkdir(exist_ok=True)
    scopes = {'pilot': (jl(D / 'pilot_items.jsonl'), jl(D / 'pilot_pairwise_items.jsonl')), 'full': (jl(D / 'items.jsonl'), jl(D / 'pairwise_items.jsonl'))}
    for scope, (it, pr) in scopes.items():
        for k in (1, 2, 3):
            p = OUT / f'annotations_ann{k}_{scope}.csv'
            write(p, rows_for(f'ann{k}', it, pr)); print(p.relative_to(ROOT), len(it) + len(pr), 'rows')
