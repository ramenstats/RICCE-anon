"""v3: reference-aggregation protocols and the (non-LLM) mapper
cascade under the FINAL evaluator (ricce/metrics/final.py).

Reference aggregation (SFT, Chg2Cap, RSCaMa on LEVIR-CC test; parses from the
shared cache): single reference, mean over references (used throughout),
max over references, consensus graph (references clustered with
ricce/metrics/consensus.py, unchanged), and the support-weighted consensus
graph, where CE-R is the optimum of the support-weighted recall assignment
(sum_j q_j S_R / sum_j q_j) -- the dual-assignment analogue of the earlier
support-weighted variant.

Mapper cascade (Qwen zero-shot, SFT, RSCaMa): every caption re-parsed under
five mapper configurations (alias / +n-gram / +embeddings [frozen] /
+n-gram+embeddings / +embeddings+ancestor fallback), scored with the final
evaluator; unresolved share is per field-resolution attempt. The LLM stage is
not re-run (it needs fresh LLM inference); its coverage numbers are a mapper
property and do not depend on the evaluator.
Outputs -> results_v3/refagg_cascade/."""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.schema import Event
from ricce.utils import read_jsonl
from ricce.benchmarks.env import load_taxes, load_weights, load_cfg
from ricce.evaluation.parse_store import ParseStore
from ricce.metrics.final import FINAL_CONFIG, final_scores
from ricce.metrics.event_v2 import _score_matrix
from ricce.metrics.consensus import consensus_events

OUT = ROOT / 'results_v3' / 'refagg_cascade'; OUT.mkdir(parents=True, exist_ok=True)
T, W = load_taxes(), load_weights()
man = {r['image_id']: r for r in read_jsonl(ROOT / 'artifacts' / 'manifests' / 'LEVIR_CC.jsonl') if r.get('split') == 'test'}
store = ParseStore(ROOT / 'results_v3' / 'parse_cache.json')


def prf(s):
    return (np.nan, np.nan, np.nan) if s['not_scorable'] else (s['ce_precision'], s['ce_recall'], s['ce_f1'])


def support_weighted(pe, cons):
    ge = [Event(**{k: v for k, v in d.items() if k != 'support'}) for d in cons]
    q = np.array([d.get('support', 1.0) for d in cons], float)
    base = final_scores(pe, ge, T, W)
    if base['not_scorable']:
        return (np.nan,) * 3
    if not pe or not ge:
        return (0.0, 0.0, 0.0)
    cfg = {k: v for k, v in FINAL_CONFIG.items() if k != 'dual_matching'}
    SP, SR = _score_matrix(pe, ge, T, W, cfg['gate_mode'], 0.1, True, cfg['root_mode'], cfg['gate_object_match'])
    r, c = linear_sum_assignment(-(SR * q[None, :]))
    rec = float((SR[r, c] * q[c]).sum() / q.sum())
    p = base['ce_precision']
    return p, rec, (2 * p * rec / (p + rec) if p + rec else 0.0)


ref_rows = []
for sy in ('sft_seed42', 'chg2cap', 'rscama'):
    acc = {k: [] for k in ('single', 'mean', 'max', 'consensus', 'consensus_weighted')}
    gaps = []
    for r in read_jsonl(ROOT / 'results' / 'baselines' / sy / 'LEVIR_CC_test.jsonl'):
        refs = man[r['image_id']]['references']
        pe = store.events(str(r['prediction']), 'multi'); ges = [store.events(g, 'multi') for g in refs]
        per = np.array([prf(final_scores(pe, g, T, W)) for g in ges], float)
        if np.isnan(per[:, 2]).all():
            continue
        acc['single'].append(per[0]); acc['mean'].append(np.nanmean(per, 0)); acc['max'].append(np.nanmax(per, 0))
        cons = consensus_events(ges, T, W)
        d = prf(final_scores(pe, [Event(**{k: v for k, v in x.items() if k != 'support'}) for x in cons], T, W))
        acc['consensus'].append(d); acc['consensus_weighted'].append(support_weighted(pe, cons))
        gaps.append((float(np.nanstd(per[:, 2])), abs(np.nanmean(per[:, 2]) - d[2]) if not np.isnan(d[2]) else np.nan))
    g = np.array(gaps, float); ok = ~np.isnan(g).any(1)
    for k, v in acc.items():
        v = np.array(v, float)
        ref_rows.append(dict(system=sy, protocol=k, n=int((~np.isnan(v[:, 2])).sum()), ce_p=float(np.nanmean(v[:, 0])),
                             ce_r=float(np.nanmean(v[:, 1])), ce_f=float(np.nanmean(v[:, 2]))))
    ref_rows.append(dict(system=sy, protocol='pearson_refdisagreement_vs_mean_consensus_gap', n=int(ok.sum()), ce_f=float(np.corrcoef(g[ok, 0], g[ok, 1])[0, 1])))
    print(sy, 'done', flush=True)
pd.DataFrame(ref_rows).to_csv(OUT / 'reference_aggregation.csv', index=False)
print(pd.DataFrame(ref_rows).round(3).to_string(), flush=True)

# ---------------------------------------------------------------- mapper cascade (parallel re-parse per configuration)
import multiprocessing as mp
cfgp = load_cfg()['parser']; emb = cfgp.get('embedding_model'); thr = float(cfgp.get('similarity_threshold', 0.58))
CONF = {'alias only': dict(use_embeddings=False), 'alias + n-gram': dict(use_embeddings=False, use_ngram=True),
        'alias + embeddings (frozen)': dict(use_embeddings=True, embedding_model=emb),
        'alias + n-gram + embeddings': dict(use_embeddings=True, embedding_model=emb, use_ngram=True),
        'alias + embeddings + ancestor fallback': dict(use_embeddings=True, embedding_model=emb, use_ancestor_fallback=True)}
_P = {}


def _init(kw):
    os.environ['OMP_NUM_THREADS'] = '1'; os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch; torch.set_num_threads(1)
    from ricce.parsing.hybrid import HybridParser
    _P['p'] = HybridParser(T['object'], T['action'], T['location'], T['quantity'], similarity_threshold=thr, **kw)


def _parse(t):
    p = _P['p']; before = dict(p.stage_counts)
    ev = [e.to_dict() for e in p.parse(t, allow_multi_event=True)]
    after = p.stage_counts
    return t, ev, {k: after[k] - before.get(k, 0) for k in after}


SYSC = ('qwen', 'sft_seed42', 'rscama')
texts = set()
preds = {}
for sy in SYSC:
    preds[sy] = [(r['image_id'], str(r['prediction'])) for r in read_jsonl(ROOT / 'results' / 'baselines' / sy / 'LEVIR_CC_test.jsonl')]
    texts |= {p for _, p in preds[sy]}
for r in man.values():
    texts |= set(r['references'])
texts = sorted(texts)
crow = []
for cname, kw in CONF.items():
    parsed, stage = {}, {}
    with mp.get_context('fork').Pool(48, initializer=_init, initargs=(kw,)) as pool:
        for t, ev, sc in pool.imap_unordered(_parse, texts, chunksize=64):
            parsed[t] = [Event(**e) for e in ev]; stage[t] = sc
    for sy in SYSC:
        used = {p for _, p in preds[sy]} | {g for iid, _ in preds[sy] for g in man[iid]['references']}
        tot = {}
        for t in used:
            for k, v in stage[t].items():
                tot[k] = tot.get(k, 0) + v
        f = []
        for iid, p in preds[sy]:
            v = [prf(final_scores(parsed[p], parsed[g], T, W)) for g in man[iid]['references']]
            v = np.array(v, float)
            if not np.isnan(v[:, 2]).all():
                f.append(np.nanmean(v, 0))
        f = np.array(f)
        crow.append(dict(config=cname, system=sy, n=len(f), ce_p=f[:, 0].mean(), ce_r=f[:, 1].mean(), ce_f=f[:, 2].mean(),
                         unresolved_share=tot.get('unresolved', 0) / max(1, sum(tot.values())), stage_counts=json.dumps(tot)))
        print(cname, sy, round(f[:, 2].mean(), 4), flush=True)
pd.DataFrame(crow).to_csv(OUT / 'mapper_cascade.csv', index=False)
print('DONE')
