"""Score a JSONL of {id, reference, candidate, dataset, group, ...} with
RICCE and the flat prior-work baselines; write a per-item CSV.

`reference` may be a string or a list of strings (score = mean over refs;
kiritchenko micro counts are summed over refs).
Usage: python scripts/run_baselines_on_jsonl.py IN.jsonl OUT.csv [--metrics a,b] [--limit N]
"""
import argparse, csv, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.baselines.registry import BaselineContext, score_pair, NAMES


def score_item(item, metrics, ctx):
    refs = item['reference'] if isinstance(item['reference'], list) else [item['reference']]
    out = {}
    for m in metrics:
        rs = [score_pair(m, item['candidate'], r, ctx) for r in refs]
        for k in ('P', 'R', 'F'):
            out[f'{m}_{k}'] = sum(r[k] for r in rs) / len(rs)
        if m == 'kiritchenko_micro':
            for k in ('_inter', '_np', '_nr'):
                out[f'{m}{k}'] = sum(r[k] for r in rs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--metrics', default=','.join(NAMES))
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    metrics = a.metrics.split(',')
    ctx = BaselineContext()
    rows = []
    with open(a.inp) as f:
        for i, line in enumerate(f):
            if a.limit and i >= a.limit: break
            it = json.loads(line)
            r = {k: v for k, v in it.items() if k not in ('reference', 'candidate')}
            r.update(score_item(it, metrics, ctx))
            rows.append(r)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    print(f'wrote {len(rows)} rows -> {a.out}', flush=True)
    os._exit(0)  # avoid a torch/tokenizers thread spin at interpreter exit

if __name__ == '__main__':
    main()
