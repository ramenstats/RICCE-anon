"""CVPR'27: canonical BERTScore for RICCE-Bench and the teaser.

bert_score 0.3.x asks the RoBERTa tokenizer for add_prefix_space=True through encode(); under
transformers 5 that argument is ignored, so the first word of every sentence is tokenized without
its leading-space marker. The RICCE-Bench BERTScore column (results_cvpr_final/ordering/scores/
bertscore.csv, 2026-09-23) and the teaser values (results_cvpr_final/teaser/teaser_scores.json)
were computed in this state. A tokenizer built with add_prefix_space=True restores canonical
BERTScore (it reproduces the stored E08 LEVIR-CC values, which predate the problem, to < 1e-6).
This script recomputes both with the canonical tokenizer and re-derives every BERTScore ordering
number with the paper's own code (ricce.evaluation.ordering_report.run_report and the
semantic-validity rule of scripts/run_semantic_validity_ordering.py). The frozen results are not
modified. Output: submission_cvpr27_new_version/07_BASELINES/bertscore_canonical/"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.evaluation.ordering_report import run_report
OUT = ROOT / 'submission_cvpr27_new_version' / '07_BASELINES' / 'bertscore_canonical'; OUT.mkdir(parents=True, exist_ok=True)
ITEMS = ROOT / 'artifacts_cvpr_final' / 'riccebench' / 'items.jsonl'


def scorer():
    from bert_score import BERTScorer
    from transformers import AutoTokenizer
    torch.set_num_threads(48)          # CPU: the shared GPU is far slower here; CPU and GPU values agree (checked on the human-study captions)
    s = BERTScorer(lang='en', rescale_with_baseline=False, device='cpu')
    s._tokenizer = AutoTokenizer.from_pretrained('roberta-large', add_prefix_space=True, use_fast=False)
    return s


sc = scorer()
items = pd.DataFrame([json.loads(l) for l in open(ITEMS)])
f = []
for i in range(0, len(items), 4000):
    _, _, x = sc.score(list(items.candidate[i:i + 4000]), list(items.reference[i:i + 4000]), batch_size=128); f += x.tolist()
    print('bench', i, flush=True)
pd.DataFrame(dict(sample_id=items.sample_id, metric='bertscore', score=f)).to_csv(OUT / 'bertscore.csv', index=False)
old = pd.read_csv(ROOT / 'results_cvpr_final' / 'ordering' / 'scores' / 'bertscore.csv').set_index('sample_id').score
d = (pd.Series(f, index=items.sample_id) - old.reindex(items.sample_id)).abs()
run_report(ITEMS, [OUT / 'bertscore.csv'], OUT, n_boot=1000, seed=0)

# semantic-validity suite, same rule and bootstrap as scripts/run_semantic_validity_ordering.py
it = items.set_index('sample_id')
s = pd.DataFrame(dict(source_id=it.source_id, ptype=it.perturbation_type, score=f), index=it.index)
GOOD, BAD = ['P0_paraphrase', 'P1_near_ancestor'], ['P4_action_reversal', 'P9_change_state_reversal', 'negation_flip', 'source_target_swap']
piv = s.pivot_table(index='source_id', columns='ptype', values='score')
rows = []
for gd in GOOD:
    for bd in BAD:
        x = piv[[gd, bd]].dropna()
        rows.append(pd.DataFrame({'source_id': x.index.values, 'bad': bd, 'win': (x[gd] > x[bd] + 1e-9).astype(float)}))
r = pd.concat(rows, ignore_index=True)
ps = r.groupby('source_id').win.agg(['sum', 'count']); idx = np.random.default_rng(0).integers(0, len(ps), (2000, len(ps)))
b = ps['sum'].values[idx].sum(1) / ps['count'].values[idx].sum(1)
sv = dict(metric='bertscore', n_constraints=len(r), strict=r.win.mean(), strict_lo=np.percentile(b, 2.5), strict_hi=np.percentile(b, 97.5))

# teaser
tz = json.load(open(ROOT / 'results_cvpr_final' / 'teaser' / 'teaser_scores.json'))
teaser = []
for x in tz['items']:
    _, _, y = sc.score([x['caption']], [tz['reference']]); teaser.append(dict(kind=x['kind'], caption=x['caption'], stored=x['bertscore'], canonical=float(y[0])))
# text-only pilot (results_v3/human_pilot): Spearman of canonical BERTScore with the pilot rating.
# Spearman is invariant to BERTScore's linear baseline rescaling, so raw F1 is sufficient.
from scipy.stats import spearmanr
pil = pd.read_csv(ROOT / 'results_v3' / 'human_pilot' / 'per_item.csv')
man = {ds: {json.loads(l)['image_id']: json.loads(l) for l in open(ROOT / 'artifacts' / 'manifests' / f'{ds}.jsonl')} for ds in pil.dataset.unique()}
preds = []
for r in pil.itertuples():
    e = pd.read_csv(ROOT / 'results' / 'E08' / r.dataset / r.system / 'per_sample_metrics.csv', usecols=['image_id', 'prediction']).set_index('image_id')
    preds.append(str(e.loc[r.image_id, 'prediction']))
_, _, pf = sc.score(preds, [man[r.dataset][r.image_id]['references'] for r in pil.itertuples()], batch_size=64)
pilot = dict(n=len(pil), spearman_canonical=float(spearmanr(pf.numpy(), pil.human).statistic),
             spearman_stored=float(spearmanr(pil.bertscore, pil.human).statistic))
json.dump(dict(bench_n=len(items), bench_abs_diff_vs_stored=dict(max=float(d.max()), median=float(d.median()), mean=float(d.mean())),
               semantic_validity=sv, teaser=teaser, pilot=pilot), open(OUT / 'summary.json', 'w'), indent=1)
print(json.dumps(json.load(open(OUT / 'summary.json')), indent=1))
