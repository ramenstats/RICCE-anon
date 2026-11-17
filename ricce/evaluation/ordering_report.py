"""Assemble ordering tables from items + per-item score CSVs (long format: sample_id,metric,score)."""
import glob, json
from pathlib import Path
import pandas as pd
from .ordering import build_constraints, evaluate_metric, evaluate_by_family, flatten

LABELS = {'bleu4': 'BLEU-4', 'meteor': 'METEOR', 'rougeL': 'ROUGE-L', 'cider_d_item': 'CIDEr-D (official corpus-IDF, per-item)',
          'bertscore': 'BERTScore-F1', 'sbert': 'SBERT-cos', 'CE-P': 'RICCE CE-P', 'CE-R': 'RICCE CE-R', 'CE-F': 'RICCE CE-F',
          'spice': 'SPICE (subset)', 'llm_judge': 'LLM-judge Qwen2.5-3B (subset)'}


def load_items(path):
    return pd.DataFrame([json.loads(l) for l in open(path)])


def load_scores_wide(score_files):
    frames = []
    for f in score_files:
        d = pd.read_csv(f)
        frames.append(d[['sample_id', 'metric', 'score']])
    if not frames:
        return pd.DataFrame()
    long = pd.concat(frames).drop_duplicates(['sample_id', 'metric'], keep='last')
    return long.pivot(index='sample_id', columns='metric', values='score')


def run_report(items_path, score_files, out_dir, n_boot=1000, seed=0):
    out_dir = Path(out_dir)
    items = load_items(items_path)
    cons = build_constraints(items)
    wide = load_scores_wide(score_files)
    rows, frows = [], []
    for m in wide.columns:
        s = wide[m].reindex(items['sample_id'])
        s.index = items['sample_id']
        rows += evaluate_metric(items, cons, s, m, n_boot, seed, suites=('core',))
        rows += evaluate_metric(items, cons, s, m, n_boot, seed, suites=('invariance',))
        rows += evaluate_metric(items, cons, s, m, n_boot, seed, suites=('extended',))
        frows += evaluate_by_family(items, cons, s, m, n_boot=300, seed=seed)
    df = flatten(rows)
    df['metric_label'] = df['metric'].map(lambda x: LABELS.get(x, x))
    fam = flatten(frows)
    fam['metric_label'] = fam['metric'].map(lambda x: LABELS.get(x, x))
    sub_f = out_dir / 'subset_sample_ids.csv'
    if sub_f.exists():
        sub = set(pd.read_csv(sub_f)['sample_id'])
        it_sub = items[items['sample_id'].isin(sub)]
        c_sub = build_constraints(it_sub)
        srows = []
        for m in wide.columns:
            s = wide[m].reindex(items['sample_id'])
            s.index = items['sample_id']
            srows += [r for r in evaluate_metric(it_sub, c_sub, s, m, n_boot, seed, suites=('core',)) if r['dataset'] == 'ALL']
        sdf = flatten(srows)
        sdf['metric_label'] = sdf['metric'].map(lambda x: LABELS.get(x, x))
        sdf.to_csv(out_dir / 'ordering_accuracy_subset_matched.csv', index=False)
    df.to_csv(out_dir / 'ordering_accuracy_by_metric.csv', index=False)
    fam.to_csv(out_dir / 'ordering_accuracy_by_family.csv', index=False)
    cons.groupby(['suite', 'group', 'kind']).size().rename('n_constraints').reset_index().to_csv(out_dir / 'constraint_counts.csv', index=False)
    wide.to_csv(out_dir / 'per_item_scores_wide.csv')
    return df, fam
