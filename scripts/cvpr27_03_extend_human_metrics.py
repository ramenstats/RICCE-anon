"""CVPR'27, B1: score every caption of the image-grounded human study
with every text metric, using exactly the references the analyst key used.

Why: in the key built by scripts/build_human_eval_items.py,
  * the 612 constructed hard cases have no official CIDEr-D: E04 stores only
    `cider_diagnostic` (a single-pair n-gram cosine without IDF, see
    ricce/metrics/conventional.py), and metrics_of() copied it into the
    `cider_d` field. The v4 "CIDEr-D, all items" correlation therefore pooled two
    different metrics;
  * BERTScore exists for 136 LEVIR-CC outputs only, SPICE and the LLM judge for 18.

References (unchanged from the key, see scripts/v3_human_eval_rescore_key.py):
hard cases and perturbation pairs -> the image's first manifest reference;
real outputs and real pairs -> every reference of the image.

Metrics written (one row per caption: 898 items + 2 x 380 pair sides):
  cider_d     official CIDEr-D formula (pycocoevalcap CiderScorer) with document
              frequencies from the dataset's test-split references, i.e. the same
              corpus E08 used for the real outputs. Check: reproduces the key's
              CIDEr-D of every real output.
  bertscore_f roberta-large F1, no baseline rescaling, max over references
              (as scripts/run_conventional_e08.py). Check: reproduces the 136 key values.
  sbert       all-MiniLM-L6-v2 cosine, max over references (conventional.sentence_bert).
  spice_f     official SPICE (pycocoevalcap, Java). Check: the 18 key values.
  llm_judge   local Qwen2.5-3B-Instruct text judge, same prompt and greedy decoding as
              ricce/metrics/spice_llm_judge.py (1-5). Check: the 18 key values.
  cider_diag  the old proxy, kept so the v4 numbers can be reproduced.

Usage: python scripts/cvpr27_03_extend_human_metrics.py [--stages cider,bert,sbert,spice,judge]
Output: submission_cvpr27_new_version/07_BASELINES/human_captions_text_metrics.csv
        submission_cvpr27_new_version/07_BASELINES/human_captions_text_metrics_checks.json"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl
from ricce.metrics.conventional import cider_diagnostic
from ricce.evaluation.human_analysis import in_study

H = ROOT / 'artifacts_cvpr_final' / 'human_eval' / 'image_grounded'
OUT = ROOT / 'submission_cvpr27_new_version' / '07_BASELINES'; OUT.mkdir(parents=True, exist_ok=True)
CSV, CHK = OUT / 'human_captions_text_metrics.csv', OUT / 'human_captions_text_metrics_checks.json'
DS = ('LEVIR_CC', 'SECOND_CC', 'DUBAI_CC')
key = [k for k in map(json.loads, open(H / 'hidden' / 'answer_key_final_evaluator.jsonl')) if in_study(k)]
items = {r['item_id']: r for r in read_jsonl(H / 'items.jsonl')}
pairs = {r['pair_id']: r for r in read_jsonl(H / 'pairwise_items.jsonl')}
man = {ds: {r['image_id']: r for r in read_jsonl(ROOT / 'artifacts' / 'manifests' / f'{ds}.jsonl')} for ds in DS}


def refs_for(k):
    m = man[k['dataset']][k['image_id']]
    return m['references'][:1] if k['source'] in ('E04_perturbation', 'pert') else list(m['references'])


def jobs():
    rows = []
    for k in key:
        base = dict(dataset=k['dataset'], image_id=k['image_id'], source=k['source'], references=refs_for(k))
        if k['kind'] == 'item':
            rows.append(dict(job=f"{k['item_id']}", kind='item', item_id=k['item_id'], side='', system=k['system'] or 'perturbed',
                             candidate=items[k['item_id']]['candidate'], key_metrics=k['metrics'], **base))
        else:
            p = pairs[k['item_id']]
            for s in ('a', 'b'):
                rows.append(dict(job=f"{k['item_id']}{s}", kind='pair', item_id=k['item_id'], side=s, system=k[f'system_{s}'] or 'perturbed',
                                 candidate=p[f'candidate_{s}'], key_metrics=k[f'metrics_{s}'], **base))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ CIDEr-D with a fixed IDF corpus
def cider_fixed_corpus(cands, refs_list, corpus_refs):
    from pycocoevalcap.cider.cider_scorer import CiderScorer, cook_refs, cook_test
    sc = CiderScorer(n=4, sigma=6.0)
    sc.crefs = [cook_refs(r) for r in corpus_refs]
    sc.compute_doc_freq()                                   # document frequency over the corpus, as E08
    ref_len = np.log(float(len(sc.crefs)))
    sc.crefs = [cook_refs(r) for r in refs_list]
    sc.ctest = [cook_test(c) for c in cands]
    return _compute_cider(sc, ref_len)


def _compute_cider(sc, ref_len):
    """Body of CiderScorer.compute_cider (pycocoevalcap), which would reset ref_len to
    log(len(crefs)); here ref_len stays log(number of corpus images)."""
    from collections import defaultdict
    n, sigma = sc.n, sc.sigma

    def counts2vec(cnts):
        vec = [defaultdict(float) for _ in range(n)]; length = 0; norm = [0.0] * n
        for ng, tf in cnts.items():
            df = np.log(max(1.0, sc.document_frequency[ng])); k = len(ng) - 1
            vec[k][ng] = float(tf) * (ref_len - df); norm[k] += vec[k][ng] ** 2
            if k == 1: length += tf
        return vec, [np.sqrt(x) for x in norm], length

    def sim(vh, vr, nh, nr, lh, lr):
        delta = float(lh - lr); val = np.zeros(n)
        for k in range(n):
            for ng, c in vh[k].items():
                val[k] += min(vh[k][ng], vr[k][ng]) * vr[k][ng]
            if nh[k] != 0 and nr[k] != 0: val[k] /= (nh[k] * nr[k])
            val[k] *= np.e ** (-(delta ** 2) / (2 * sigma ** 2))
        return val

    out = []
    for test, refs in zip(sc.ctest, sc.crefs):
        vh, nh, lh = counts2vec(test); s = np.zeros(n)
        for ref in refs:
            vr, nr, lr = counts2vec(ref); s += sim(vh, vr, nh, nr, lh, lr)
        out.append(float(np.mean(s) / len(refs) * 10.0))
    return out


def stage_cider(J):
    J['cider_d'] = np.nan
    for ds in DS:
        corpus = [r['references'] for r in man[ds].values() if r.get('split') == 'test']
        idx = J.index[J.dataset == ds]
        J.loc[idx, 'cider_d'] = cider_fixed_corpus(list(J.loc[idx, 'candidate']), list(J.loc[idx, 'references']), corpus)
    J['cider_diag'] = [cider_diagnostic(c, r) for c, r in zip(J.candidate, J.references)]


def stage_bert(J):
    import torch
    from bert_score import BERTScorer
    from transformers import AutoTokenizer
    sc = BERTScorer(lang='en', rescale_with_baseline=False, device='cuda' if torch.cuda.is_available() else 'cpu')
    # bert_score 0.3.x passes add_prefix_space=True to encode(), which transformers 5 ignores, so the first
    # word loses its leading-space token. A tokenizer built with add_prefix_space=True restores canonical
    # BERTScore; it reproduces the stored E08 values to < 1e-6 (checked on 300 LEVIR-CC items).
    sc._tokenizer = AutoTokenizer.from_pretrained('roberta-large', add_prefix_space=True, use_fast=False)
    _, _, f = sc.score(list(J.candidate), [list(r) for r in J.references], batch_size=64)
    J['bertscore_f'] = [float(x) for x in f]
    J.attrs['bert_model'] = sc.model_type


def stage_sbert(J):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    out = []
    for c, rs in zip(J.candidate, J.references):
        e = m.encode([c] + list(rs), normalize_embeddings=True); out.append(float(np.max(e[1:] @ e[0])))
    J['sbert'] = out


def stage_spice(J):
    from ricce.metrics.spice_llm_judge import spice_corpus
    ids = list(J.job)
    _, per, reason = spice_corpus(ids, list(J.candidate), dict(zip(ids, [list(r) for r in J.references])))
    if reason: print('SPICE unavailable:', reason)
    J['spice_f'] = [per.get(i, np.nan) for i in ids]


def stage_judge(J):
    from ricce.metrics.spice_llm_judge import llm_judge_batch
    s, reason = llm_judge_batch(list(J.candidate), [list(r) for r in J.references], batch_size=16)
    if reason: print('LLM judge:', reason)
    J['llm_judge'] = s


STAGES = dict(cider=stage_cider, bert=stage_bert, sbert=stage_sbert, spice=stage_spice, judge=stage_judge)
COLS = dict(cider=['cider_d', 'cider_diag'], bert=['bertscore_f'], sbert=['sbert'], spice=['spice_f'], judge=['llm_judge'])


def checks(J):
    out = {}
    km = lambda m: J.key_metrics.map(lambda d: d.get(m, np.nan)).astype(float)
    real = J.source.isin(['E08_model_output', 'real'])
    for col, kname, mask, what in (('cider_d', 'cider_d', real, 'real outputs: official CIDEr-D from E08'),
                                   ('cider_diag', 'cider_d', ~real, 'hard cases: key cider_d equals the cider_diagnostic proxy'),
                                   ('bertscore_f', 'bertscore_f', None, 'BERTScore values present in the key'),
                                   ('spice_f', 'spice_f', None, 'SPICE values present in the key'),
                                   ('llm_judge', 'llm_judge', None, 'LLM-judge values present in the key')):
        if col not in J: continue
        k = km(kname); m = k.notna() & J[col].notna() & (mask if mask is not None else True)
        d = (J.loc[m, col] - k[m]).abs()
        out[col] = dict(what=what, n_compared=int(m.sum()), max_abs_diff=float(d.max()) if len(d) else None,
                        n_above_1e6=int((d > 1e-6).sum()), n_above_1e3=int((d > 1e-3).sum()),
                        exact_agreement_rate=float((d < 1e-9).mean()) if len(d) else None)
    out['coverage'] = {c: int(J[c].notna().sum()) for c in J.columns if c in sum(COLS.values(), [])}
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--stages', default='cider,bert,sbert,spice,judge')
    a = ap.parse_args()
    J = jobs()
    if CSV.exists():                                   # resume: keep finished stages
        old = pd.read_csv(CSV).set_index('job')
        for cols in COLS.values():
            for c in cols:
                if c in old: J[c] = J.job.map(old[c])
    for s in a.stages.split(','):
        t = time.time(); STAGES[s](J); print(f'{s}: {time.time() - t:.0f}s', flush=True)
        J.drop(columns=['key_metrics']).assign(references=J.references.map(json.dumps)).to_csv(CSV, index=False)
    c = checks(J); c['bert_model'] = J.attrs.get('bert_model', 'roberta-large (bert_score default for lang=en)')
    CHK.write_text(json.dumps(c, indent=1)); print(json.dumps(c, indent=1))
