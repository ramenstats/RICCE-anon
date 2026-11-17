"""Runtime/cost measurements. Every number is a wall-clock time from this script on this machine.
Usage: python scripts/run_runtime.py [--n 1000] [--llm-n 100] [--bert-n 200] [--skip-llm]"""
import os, sys, json, time, argparse, random, platform, statistics
ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=1000); ap.add_argument('--llm-n', type=int, default=100)
ap.add_argument('--bert-n', type=int, default=200); ap.add_argument('--skip-llm', action='store_true'); ap.add_argument('--llm-gpu', action='store_true', default=True)
args = ap.parse_args()
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from ricce.evaluation.mapper_v2 import *   # forces CUDA_VISIBLE_DEVICES='' unless set by caller
import torch
from ricce.metrics.event import ricce_scores, match_events, match_events_greedy, event_similarity_pr
from ricce.metrics.conventional import evaluate_caption, cider_d_corpus, bertscore
from scipy.optimize import linear_sum_assignment

OUT = ROOT / 'results_cvpr_final/runtime'; OUT.mkdir(parents=True, exist_ok=True)
cfg = load_cfg(); taxes = load_taxes(cfg); W = cfg['metric']['weights']; EMB = cfg['parser']['embedding_model']
LLM = cfg['vlm']['model_name_or_path']
T = lambda: time.perf_counter()
res = {'hardware': {'cpu': [l.split(':')[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')][0], 'cpu_threads_visible': os.cpu_count(),
                    'torch_num_threads': torch.get_num_threads(), 'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'), 'python': platform.python_version(), 'torch': torch.__version__,
                    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'gpu': (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                    'loadavg_at_start_1_5_15min': os.getloadavg(), 'note': 'shared server; other jobs may add noise. Single process, batch size 1 (HybridParser encodes one caption per call).'},
       'embedding_model': EMB, 'embedding_batch_size': 1, 'llm_fallback_model': LLM, 'n_captions_requested': args.n, 'conditions': {}}
# ------- data: n reference captions sampled from test splits of the 3 datasets (real captions) + real predictions
rnd = random.Random(0); caps, items = [], []
for ds in DATASETS:
    ms = [r for r in read_jsonl(ROOT / f'artifacts/manifests/{ds}.jsonl') if r['split'] == 'test']
    pr = {r['image_id']: r['prediction'] for r in read_jsonl(ROOT / f'results/baselines/sft_seed42/{ds}_test.jsonl')}
    ms = [r for r in ms if r['image_id'] in pr]; rnd.shuffle(ms)
    for r in ms[: args.n // 3 + (1 if ds == DATASETS[0] else 0)]:
        caps.append(rnd.choice(r['references'])); items.append((pr[r['image_id']], r['references']))
caps, items = caps[: args.n], items[: args.n]
res['n_captions_actual'] = len(caps); res['sample'] = {'captions': 'random reference captions from test splits (seed 0), ~1/3 per dataset', 'items': 'sft_seed42 predictions + 5 references (same images)', 'n_items': len(items)}
per1000 = lambda sec, n: sec * 1000.0 / n

def run_parse(p, texts):
    t = T()
    for c in texts: p.parse(c)
    return T() - t

def cond(name, fn, n, repeats=1, note=''):
    ts = [fn() for _ in range(repeats)]; med = statistics.median(ts)
    res['conditions'][name] = {'n': n, 'repeats': repeats, 'seconds_each': ts, 'median_seconds': med, 'sec_per_1000': per1000(med, n), 'note': note}
    print(name, res['conditions'][name]['sec_per_1000'], flush=True)

# 1. rules/alias only (CPU)
p_rules = build_parser(taxes, cfg, use_embeddings=False); cond('rules_alias_only_cpu', lambda: run_parse(build_parser(taxes, cfg, use_embeddings=False), caps), len(caps), 3)
# 2. + ngram (CPU)
cond('plus_ngram_cpu', lambda: run_parse(build_parser(taxes, cfg, use_embeddings=False, use_ngram=True), caps), len(caps), 3)
# 3. + semantic embeddings: model load, cold cache (label embeddings not yet cached), warm cache (same instance, second pass)
t = T(); p_sem = build_parser(taxes, cfg, use_ngram=False); res['embedding_model_load_seconds'] = T() - t
cold = run_parse(p_sem, caps); res['conditions']['semantic_cold_cache_cpu'] = {'n': len(caps), 'seconds_each': [cold], 'sec_per_1000': per1000(cold, len(caps)), 'note': 'first pass on a fresh parser; includes computing+caching taxonomy label embeddings (4 taxonomies); caption embeddings are never cached; excludes model load'}
warm = [run_parse(p_sem, caps) for _ in range(3)]; res['conditions']['semantic_warm_cache_cpu'] = {'n': len(caps), 'seconds_each': warm, 'median_seconds': statistics.median(warm), 'sec_per_1000': per1000(statistics.median(warm), len(caps)), 'note': 'same instance, label-embedding cache warm'}
print('semantic', res['conditions']['semantic_cold_cache_cpu']['sec_per_1000'], res['conditions']['semantic_warm_cache_cpu']['sec_per_1000'], flush=True)
# stage counts on the semantic pass (fresh counter for one warm pass)
p_sem.stage_counts = {k: 0 for k in p_sem.stage_counts}; run_parse(p_sem, caps); sc = dict(p_sem.stage_counts); tot = sum(sc.values())
res['stage_counts_semantic_pipeline_1000'] = {'counts': sc, 'total_field_resolutions': tot, 'frac_reaching_LLM_if_enabled': sc['unresolved'] / tot,
    'explanation': 'a field resolution reaches the LLM fallback iff exact, ngram(off here) and semantic all fail, i.e. it is counted as unresolved when the LLM is off'}
p_full = build_parser(taxes, cfg, use_ngram=True); run_parse(p_full, caps); sc2 = dict(p_full.stage_counts)
res['stage_counts_ngram_semantic_1000'] = {'counts': sc2, 'total': sum(sc2.values()), 'frac_reaching_LLM_if_enabled': sc2['unresolved'] / sum(sc2.values())}

# 5. full RICCE scoring (parse pred + 5 refs [warm semantic parser] + Hungarian) and scoring-only
parser = build_parser(taxes, cfg); run_parse(parser, caps[:20])
def full_score():
    t = T()
    for pred, refs in items:
        pe = parser.parse(pred); [ricce_scores(pe, parser.parse(g), taxes, W) for g in refs]
    return T() - t
cond('full_ricce_end_to_end_cpu', full_score, len(items), 1, 'per item: parse prediction + parse 5 refs (semantic parser, warm label cache) + 5 Hungarian scorings')
pre = [(parser.parse(p), [parser.parse(g) for g in refs]) for p, refs in items]
def score_only():
    t = T()
    for pe, ges in pre: [ricce_scores(pe, ge, taxes, W) for ge in ges]
    return T() - t
cond('ricce_scoring_only_preparsed_cpu', score_only, len(items), 3, 'event matching + aggregation only (5 refs/item)')
# 6. conventional metrics
def conv_basic():
    t = T()
    for pred, refs in items: evaluate_caption(pred, refs, False)
    return T() - t
cond('conventional_bleu_meteor_rouge_cpu', conv_basic, len(items), 1, 'BLEU1-4 + METEOR + ROUGE-L + single-pair CIDEr diagnostic (evaluate_caption, expensive=False)')
def cider():
    ids = [str(i) for i in range(len(items))]; t = T(); cider_d_corpus(ids, {j: items[j][0] for j in range(len(items))}, {str(i): items[i][1] for i in range(len(items))}); return T() - t
try:
    ids = [str(i) for i in range(len(items))]
    t = T(); o, _ = cider_d_corpus(ids, [x[0] for x in items], {i: items[int(i)][1] for i in ids}); dt = T() - t
    res['conditions']['cider_d_corpus_cpu'] = {'n': len(items), 'seconds_each': [dt], 'sec_per_1000': per1000(dt, len(items)), 'value_finite': bool(np.isfinite(o))}
except Exception as e: res['conditions']['cider_d_corpus_cpu'] = {'status': 'failed', 'error': str(e)}
bn = min(args.bert_n, len(items))
try:
    t = T(); bertscore(items[0][0], items[0][1]); res['bertscore_model_load_plus_first_call_seconds_cpu'] = T() - t
    t = T(); vals = [bertscore(p, r) for p, r in items[:bn]]; dt = T() - t
    res['conditions']['bertscore_roberta_large_cpu'] = {'n': bn, 'seconds_each': [dt], 'sec_per_1000': per1000(dt, bn), 'value_finite': bool(np.all(np.isfinite(vals))),
                                                       'note': f'MEASURED on n={bn} then scaled linearly to 1000 (extrapolation, labelled); model load excluded'}
except Exception as e: res['conditions']['bertscore_roberta_large_cpu'] = {'status': 'failed', 'error': str(e)}

# 7. matching: greedy vs Hungarian on real multi-event captions (allow_multi_event=True)
pairs = []
for pred, refs in items:
    pe = parser.parse(pred, allow_multi_event=True)
    for g in refs:
        ge = parser.parse(g, allow_multi_event=True)
        if len(pe) >= 2 or len(ge) >= 2: pairs.append((pe, ge))
def time_fn(fn):
    t = T()
    for pe, ge in pairs: fn(pe, ge, taxes, W)
    return T() - t
mt = {'n_pairs_with_multi_event_side': len(pairs), 'size_dist': {f'{len(a)}x{len(b)}': 0 for a, b in pairs[:0]}}
for a, b in pairs: k = f'{len(a)}x{len(b)}'; mt['size_dist'][k] = mt['size_dist'].get(k, 0) + 1
if pairs:
    h = [time_fn(match_events) for _ in range(3)]; g_ = [time_fn(match_events_greedy) for _ in range(3)]
    mt['hungarian_seconds_median'] = statistics.median(h); mt['greedy_seconds_median'] = statistics.median(g_)
    mt['hungarian_ms_per_pair'] = 1000 * statistics.median(h) / len(pairs); mt['greedy_ms_per_pair'] = 1000 * statistics.median(g_) / len(pairs)
    # assignment step only, on precomputed S_F matrices
    mats = [match_events(a, b, taxes, W)[1] for a, b in pairs]
    t = T(); [linear_sum_assignment(-m) for m in mats if m.size]; mt['hungarian_assignment_only_ms_per_pair'] = 1000 * (T() - t) / len(mats)
    def greedy_only(m):
        used_r, used_c = set(), set()
        for s, i, j in sorted(((m[i, j], i, j) for i in range(m.shape[0]) for j in range(m.shape[1])), key=lambda x: -x[0]):
            if i in used_r or j in used_c: continue
            used_r.add(i); used_c.add(j)
    t = T(); [greedy_only(m) for m in mats if m.size]; mt['greedy_assignment_only_ms_per_pair'] = 1000 * (T() - t) / len(mats)
    f = lambda fn: [ricce_scores(a, b, taxes, W, matcher=fn)['ce_f1'] for a, b in pairs]
    fh, fg = f('hungarian'), f('greedy'); mt['frac_pairs_cef_differs'] = float(np.mean(np.abs(np.array(fh) - np.array(fg)) > 1e-9)); mt['mean_cef_hungarian'] = float(np.mean(fh)); mt['mean_cef_greedy'] = float(np.mean(fg))
res['matching_greedy_vs_hungarian'] = mt
res['parser_config'] = dict(cfg['parser']); res['hardware']['loadavg_at_end_of_cpu_part'] = os.getloadavg()
json.dump(res, open(OUT / 'summary_cpu.json', 'w'), indent=1, default=str)
print('CPU part done', flush=True)

# 4. LLM fallback on a small sample (GPU). Not extrapolated: reports measured n=llm_n plus stage counts.
if not args.skip_llm:
    import subprocess
    env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = os.environ.get('LLM_GPU', '0')
    code = f'''
import sys,json,time,os
sys.path.insert(0,{str(ROOT)!r})
from ricce.evaluation.mapper_v2 import *
import torch
cfg=load_cfg();taxes=load_taxes(cfg)
caps=json.load(open({str(OUT / "_llm_caps.json")!r}))
p=build_parser(taxes,cfg,use_ngram=True,use_llm_fallback=True,llm_model={LLM!r})
t=time.perf_counter(); p._load_llm(); load=time.perf_counter()-t
ok=bool(p._llm)
t=time.perf_counter()
for c in caps: p.parse(c)
dt=time.perf_counter()-t
sc=dict(p.stage_counts); tot=sum(sc.values())
json.dump({{'llm_loaded':ok,'llm_load_seconds':load,'n_captions':len(caps),'seconds':dt,'sec_per_caption':dt/len(caps),'stage_counts':sc,'total_field_resolutions':tot,'frac_reaching_LLM':(sc['llm_fallback']+sc['unresolved'])/tot,'frac_llm_resolved_of_reached':sc['llm_fallback']/max(1,sc['llm_fallback']+sc['unresolved']),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'batch_size':1,'max_new_tokens':32,'quantization':'none (torch_dtype=auto, bf16)'}},open({str(OUT / "summary_llm.json")!r},'w'),indent=1,default=str)
'''
    json.dump(caps[: args.llm_n], open(OUT / '_llm_caps.json', 'w'))
    r = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
    print(r.stdout[-500:], r.stderr[-800:])
    (OUT / '_llm_caps.json').unlink()
