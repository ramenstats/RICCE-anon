"""Merge extra per-item metric scores (e.g. Kiritchenko / LCA-F / HCM / TaxAware from the baselines agent)
into the ordering evaluation and regenerate ordering_accuracy_by_metric.csv / _by_family.csv.

Input CSV(s): columns  sample_id, metric, score   (sample_id = RICCE-Bench item id; higher = better).
Usage:
  python scripts/merge_extra_metrics_into_ordering.py --extra path/to/extra_scores.csv [more.csv ...]
Extra files are COPIED to results_cvpr_final/ordering/scores/extra_<name>.csv and folded in with all base scores.
Items whose score is missing are dropped constraint-wise (n_dropped_nan). Extra metrics may cover a subset of items.
"""
import argparse, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.evaluation.ordering_report import run_report
import glob
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--extra', nargs='+', required=True)
ap.add_argument('--out', default=str(ROOT / 'results_cvpr_final' / 'ordering'))
ap.add_argument('--items', default=str(ROOT / 'artifacts_cvpr_final' / 'riccebench' / 'items.jsonl'))
ap.add_argument('--n-boot', type=int, default=1000)
a = ap.parse_args()
out = Path(a.out)
(out / 'scores').mkdir(parents=True, exist_ok=True)
for f in a.extra:
    d = pd.read_csv(f)
    missing = {'sample_id', 'metric', 'score'} - set(d.columns)
    if missing:
        sys.exit(f'{f}: missing columns {missing}')
    shutil.copy(f, out / 'scores' / f'extra_{Path(f).name}')
files = sorted(glob.glob(str(out / 'scores' / '*.csv')))
df, fam = run_report(a.items, files, out, a.n_boot)
print(df[(df.dataset == 'ALL') & (df.suites == 'core')][['metric', 'n_constraints', 'accuracy_strict', 'accuracy_tiehalf']].to_string())
