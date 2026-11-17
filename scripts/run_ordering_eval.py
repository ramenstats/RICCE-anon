"""Score RICCE-Bench items with each metric and compute pairwise ordering accuracy.
Cheap metrics: bleu4 meteor rougeL cider_d bertscore sbert ricce  (CPU only)
Subset metrics (fixed random subset of source captions, --subset-n per dataset): spice llm_judge (checkpointed).
Usage: python scripts/run_ordering_eval.py --metrics bleu4 meteor ...   |   --report-only
"""
import argparse, json, os, random, sys, time, glob
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.evaluation.ordering_report import load_items, run_report
from ricce.schema import Event

OUT = ROOT / 'results_cvpr_final' / 'ordering'
SC = OUT / 'scores'
ITEMS = ROOT / 'artifacts_cvpr_final' / 'riccebench' / 'items.jsonl'


def save(metric_rows, name):
    SC.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows, columns=['sample_id', 'metric', 'score']).to_csv(SC / f'{name}.csv', index=False)


def subset_items(items, n, seed):
    out = []
    for ds, g in items.groupby('dataset'):
        sids = sorted(g['source_id'].unique())
        pick = set(random.Random(f'{seed}|{ds}|subset').sample(sids, min(n, len(sids))))
        out.append(g[g['source_id'].isin(pick)])
    return pd.concat(out)


def run_metric(m, items, args):
    t = time.time()
    if m in ('bleu4', 'meteor', 'rougeL'):
        from ricce.metrics.conventional import sentence_bleu_scores, meteor, rouge_l
        f = {'bleu4': lambda c, r: sentence_bleu_scores(c, [r])['bleu4'], 'meteor': lambda c, r: meteor(c, [r]),
             'rougeL': lambda c, r: rouge_l(c, [r])}[m]
        save([(r.sample_id, m, f(r.candidate, r.reference)) for r in items.itertuples()], m)
    elif m == 'cider_d':
        from ricce.metrics.conventional import cider_d_corpus
        rows = []
        for ds, g in items.groupby('dataset'):
            ids = list(g['sample_id'])
            refs = {r.sample_id: [r.reference.strip()] for r in g.itertuples()}
            preds = {r.sample_id: r.candidate.strip() for r in g.itertuples()}
            _, per = cider_d_corpus(ids, [preds[i] for i in ids], refs)
            rows += [(i, 'cider_d_item', per[i]) for i in ids]
        save(rows, 'cider_d_item')
    elif m == 'bertscore':
        from bert_score import BERTScorer
        sc = BERTScorer(lang='en', rescale_with_baseline=False, device='cpu')
        cand, ref = list(items['candidate']), list(items['reference'])
        out = []
        for i in range(0, len(cand), 2000):
            _, _, f = sc.score(cand[i:i + 2000], ref[i:i + 2000], batch_size=128)
            out += f.tolist()
            print('bertscore', i, flush=True)
        save(list(zip(items['sample_id'], ['bertscore'] * len(out), out)), 'bertscore')
    elif m == 'sbert':
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        texts = sorted(set(items['candidate']) | set(items['reference']))
        emb = dict(zip(texts, model.encode(texts, normalize_embeddings=True, batch_size=256)))
        save([(r.sample_id, 'sbert', float(emb[r.candidate] @ emb[r.reference])) for r in items.itertuples()], 'sbert')
    elif m == 'ricce':
        from ricce.metrics.event import ricce_scores
        from ricce.benchmarks.env import load_taxes, load_weights
        taxes, w = load_taxes(), load_weights()
        E = lambda L: [Event(**e) for e in L]
        rows = []
        for r in items.itertuples():
            s = ricce_scores(E(r.candidate_event), E(r.source_event), taxes, w)
            rows += [(r.sample_id, 'CE-P', s['ce_precision']), (r.sample_id, 'CE-R', s['ce_recall']), (r.sample_id, 'CE-F', s['ce_f1'])]
        save(rows, 'ricce')
    elif m in ('spice', 'llm_judge'):
        sub = subset_items(items, args.subset_n, args.seed)
        sub[['sample_id']].to_csv(OUT / 'subset_sample_ids.csv', index=False)
        ck = SC / f'{m}.csv'
        done = set(pd.read_csv(ck)['sample_id']) if ck.exists() else set()
        todo = sub[~sub['sample_id'].isin(done)]
        print(m, 'subset items', len(sub), 'todo', len(todo), flush=True)
        chunk = args.chunk
        for i in range(0, len(todo), chunk):
            g = todo.iloc[i:i + chunk]
            if m == 'spice':
                from ricce.metrics.spice_llm_judge import spice_corpus
                # zero-padded ids: pycocoevalcap sorts ids lexicographically and spice_corpus zips scores in INPUT order,
                # so ids must already be in lexicographic order or scores get mis-assigned (latent bug in spice_corpus).
                ids = [f'{k:06d}' for k in range(len(g))]
                refs = {f'{k:06d}': [r.strip()] for k, r in enumerate(g['reference'])}
                preds = [c.strip() for c in g['candidate']]
                _, per, reason = spice_corpus(ids, preds, refs)
                sc = [per[k] for k in ids]
                if reason:
                    print('SPICE reason:', reason, flush=True)
                    if all(x != x for x in sc):
                        sys.exit('SPICE unavailable: ' + str(reason))
            else:
                from ricce.metrics.spice_llm_judge import llm_judge_batch
                sc, reason = llm_judge_batch([c.strip() for c in g['candidate']], [[r.strip()] for r in g['reference']])
                if reason:
                    print('judge reason:', reason, flush=True)
            new = pd.DataFrame({'sample_id': g['sample_id'].values, 'metric': m, 'score': sc})
            new.to_csv(ck, mode='a', header=not ck.exists(), index=False)
            print(m, 'checkpoint', i + len(g), '/', len(todo), f'{time.time()-t:.0f}s', flush=True)
    else:
        raise SystemExit('unknown metric ' + m)
    print(m, 'done', f'{time.time()-t:.0f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics', nargs='*', default=[])
    ap.add_argument('--report-only', action='store_true')
    ap.add_argument('--subset-n', type=int, default=300)
    ap.add_argument('--chunk', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-boot', type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_items(ITEMS)
    for m in args.metrics:
        run_metric(m, items, args)
    if args.report_only or not args.metrics:
        files = sorted(glob.glob(str(SC / '*.csv')))
        df, fam = run_report(ITEMS, files, OUT, args.n_boot)
        print(df[(df.dataset == 'ALL') & (df.suites == 'core')][['metric', 'n_constraints', 'accuracy_strict', 'accuracy_tiehalf']].to_string())


if __name__ == '__main__':
    main()
