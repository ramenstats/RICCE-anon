"""Build JSONL inputs (schema of run_baselines_on_jsonl.py) from existing assets:
E04 perturbation sets and frozen system predictions (test split, all refs)."""
import csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts_cvpr_final' / 'baselines' / 'inputs'; OUT.mkdir(parents=True, exist_ok=True)
DATASETS = ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC']
SYSTEMS = ['qwen', 'sft_seed42', 'chg2cap', 'swmcc']
for d in DATASETS:
    with open(OUT / f'e04_{d}.jsonl', 'w') as f:
        for r in csv.DictReader(open(ROOT / 'results' / 'E04' / d / 'per_sample_metrics.csv')):
            f.write(json.dumps({'id': f"{r['image_id']}|{r['type']}", 'dataset': d, 'group': r['type'],
                                'reference': r['gt'], 'candidate': r['prediction'],
                                'e04_ce_f1': float(r['ce_f1'])}) + '\n')
    man = {}
    for l in open(ROOT / 'artifacts' / 'manifests' / f'{d}.jsonl'):
        x = json.loads(l)
        if x['split'] == 'test': man[x['image_id']] = x['references']
    for s in SYSTEMS:
        n = 0
        with open(OUT / f'sys_{s}_{d}.jsonl', 'w') as f:
            for l in open(ROOT / 'results' / 'baselines' / s / f'{d}_test.jsonl'):
                x = json.loads(l)
                if x['image_id'] in man:
                    f.write(json.dumps({'id': x['image_id'], 'dataset': d, 'group': s,
                                        'reference': man[x['image_id']], 'candidate': x['prediction']}) + '\n'); n += 1
        print(s, d, n)
