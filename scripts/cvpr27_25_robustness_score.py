"""Robustness to simulated acquisition interference, step 3: scores.

Scores every caption of scripts/cvpr27_24_robustness_infer.py exactly as the
frozen benchmark (Tab. 5, scripts/v3_recompute_systems.py) does: final
evaluator (FINAL_CONFIG), multi-event parses, per-image mean over the
references, not-scorable pairs excluded. BLEU-4 (mean sentence BLEU, NLTK
smoothing 1) and corpus CIDEr-D use the functions behind Tab. 5
(ricce.metrics.conventional).

Checks:
  * clean captions are compared with the frozen predictions in results/baselines/
    for every (system, dataset) cell; re-running on another device or entry script
    flips a few beam-search near-ties, so the number of identical captions and the
    clean-minus-Tab. 5 difference of every metric are recorded; for cells whose
    clean captions are all identical, CE-P/CE-R/CE-F, BLEU-4 and CIDEr-D must equal
    Tab. 5 (results_v3/systems/summary.csv) to 1e-9 (the run fails otherwise);
  * parses taken from the shared cache equal fresh parses on a 300-text sample
    (the run fails otherwise).
The clean column of every comparison is this run's own clean pass (same code,
checkpoint, decoding and device as the corrupted passes), not the frozen file.
Diagnostics per scenario, using the definitions of the cross-dataset analysis
(scripts/v3_recompute_systems.py, section 3): a scene is "changed" if fewer than
half of its parsed references describe no change; "missed change" = a caption
whose parse is no-change on a changed scene; "hallucinated change" = a caption
that parses to a change event on an unchanged scene; "gated" = the hard
contradiction gate acts against at least one reference.
Statistics: 'corrupted' = mean over the 25 scenario means (Real-CD's
"corrupted average"); 95% CIs of the clean-to-corrupted change from a paired
bootstrap over images (10,000 resamples) on images scorable in all 26 inputs;
system comparisons under interference by paired bootstrap with Holm per dataset.
Outputs -> results_cvpr27_robustness/: per_image_scores.csv.gz, per_condition.csv,
per_system.csv, pairwise_corrupted.csv, robustness.json
Usage: python scripts/cvpr27_25_robustness_score.py [--allow-partial]
"""
import argparse, itertools, json, multiprocessing as mp, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl
from ricce.benchmarks.env import load_taxes, load_weights
from ricce.evaluation.parse_store import ParseStore
from ricce.metrics.final import FINAL_CONFIG, gate_activations
from ricce.metrics.event_v2 import ricce_scores_v2
from ricce.metrics.conventional import sentence_bleu_scores, cider_d_corpus

ap = argparse.ArgumentParser()
ap.add_argument('--allow-partial', action='store_true')
ap.add_argument('--procs', type=int, default=48)
a = ap.parse_args()

R = ROOT / 'results_cvpr27_robustness'
PRED = R / 'predictions'
DS = ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC']
SYSTEMS = ['chg2cap', 'swmcc', 'rsiccformer', 'promptcc', 'rscama']
TYPES = 'FGIMS'
SCEN = [f'{x}-{y}' for x in TYPES for y in TYPES]
CONDS = ['clean'] + [f's3_{c}' for c in SCEN]
SWEEP = [f's{v}_{x}-{x}' for v in (1, 5) for x in TYPES]  # LEVIR-CC only
B = 10000
T, W = load_taxes(), load_weights()
man = {ds: {r['image_id']: r for r in read_jsonl(ROOT / 'artifacts' / 'manifests' / f'{ds}.jsonl') if r.get('split') == 'test'} for ds in DS}
checks = {}

# ------------------------------------------------------------------ predictions
P = {}
for sy in SYSTEMS:
    for ds in DS:
        for c in CONDS + (SWEEP if ds == 'LEVIR_CC' else []):
            f = PRED / sy / ds / f'{c}.jsonl'
            if f.with_suffix('.done').exists():
                P[(sy, ds, c)] = {r['image_id']: str(r['prediction']) for r in read_jsonl(f)}
cells = sorted({(sy, ds) for sy, ds, _ in P})
missing = [(sy, ds, c) for sy, ds in cells for c in CONDS + (SWEEP if ds == 'LEVIR_CC' else []) if (sy, ds, c) not in P]
print(len(P), 'prediction files,', len(cells), 'cells,', len(missing), 'missing', flush=True)
if missing and not a.allow_partial:
    sys.exit(f'incomplete: {missing[:5]} ...')

# frozen identity of the clean runs
ident = {}
for sy, ds in cells:
    if (sy, ds, 'clean') not in P:
        continue
    fz = {r['image_id']: str(r['prediction']) for r in read_jsonl(ROOT / 'results' / 'baselines' / sy / f'{ds}_test.jsonl') if r['image_id'] in man[ds]}
    new = P[(sy, ds, 'clean')]
    assert set(new) == set(man[ds]), (sy, ds)
    ident[f'{ds}|{sy}'] = dict(n=len(new), identical=sum(new[k].strip() == fz[k].strip() for k in new))
checks['clean_identical_to_frozen'] = ident
print('clean vs frozen:', {k: f"{v['identical']}/{v['n']}" for k, v in ident.items()}, flush=True)

# ------------------------------------------------------------------ parses
store = ParseStore(R / 'parse_cache.json')
texts = sorted({t for d in P.values() for t in d.values()} | {g for ds in DS for r in man[ds].values() for g in r['references']})
main = json.load(open(ROOT / 'results_v3' / 'parse_cache.json'))['multi']
seeded = 0
for t in texts:
    if t in main and t not in store.d['multi']:
        store.seed('multi', t, main[t]); seeded += 1
t0 = time.time()
n_new = store.ensure(texts, 'multi', n_proc=32)
store.save()
print(f'parses: {len(texts)} texts, {seeded} seeded from the shared cache, {n_new} parsed ({time.time() - t0:.0f}s)', flush=True)
# determinism: fresh parses of cached texts equal the cache
rs = random.Random(0)
sample = rs.sample(sorted(t for t in texts if t in main), min(300, sum(t in main for t in texts)))
fresh = ParseStore(R / '_nonexistent_parse_check.json')
fresh.ensure(sample, 'multi', n_proc=32)
mism = [t for t in sample if fresh.d['multi'][t] != main[t]]
checks['parse_cache_vs_fresh'] = dict(n=len(sample), mismatches=len(mism))
assert not mism, mism[:3]
EV = {t: store.events(t, 'multi') for t in texts}


# ------------------------------------------------------------------ per (caption, image) scores
def stratum(e):
    if e.change_state == 'no_change':
        return 'no_change'
    core = int(e.object is not None and e.object != T['object'].root) + int(e.action is not None and e.action != T['action'].root)
    return {2: 'full', 1: 'partial', 0: 'failed'}[core]


def ev_stratum(es):
    if not es:
        return 'failed'
    if all(e.change_state == 'no_change' for e in es):
        return 'no_change'
    s = [stratum(e) for e in es]
    if all(x in ('full', 'no_change') for x in s):
        return 'full'
    return 'failed' if all(x == 'failed' for x in s) else 'partial'


def score(key):
    ds, iid, p = key
    pe, vals, gated = EV[p], [], 0
    for g in man[ds][iid]['references']:
        s = ricce_scores_v2(pe, EV[g], T, W, **FINAL_CONFIG)
        vals.append((np.nan,) * 3 if s['not_scorable'] else (s['ce_precision'], s['ce_recall'], s['ce_f1']))
        gated += bool(gate_activations(pe, EV[g], T, W))
    v = np.array(vals, float)
    m = [np.nanmean(v[:, k]) if np.isfinite(v[:, k]).any() else np.nan for k in range(3)]
    return key, (*m, int(np.isnan(v[:, 2]).sum()), gated, sentence_bleu_scores(p, man[ds][iid]['references'])['bleu4'])


keys = sorted({(ds, iid, p) for (sy, ds, c), d in P.items() for iid, p in d.items()})
t0 = time.time()
with mp.get_context('fork').Pool(a.procs) as pool:
    S = dict(pool.imap_unordered(score, keys, chunksize=256))
print(f'scored {len(keys)} unique (caption, image) pairs in {time.time() - t0:.0f}s', flush=True)
ref_changed = {ds: {iid: float(np.mean([ev_stratum(EV[g]) == 'no_change' for g in r['references']])) < 0.5 for iid, r in man[ds].items()} for ds in DS}

rows = []
for (sy, ds, c), d in P.items():
    for iid, p in d.items():
        cep, cer, cef, nns, gated, b4 = S[(ds, iid, p)]
        rows.append((ds, sy, c, iid, p, cep, cer, cef, nns, gated, b4, ev_stratum(EV[p]), ref_changed[ds][iid]))
D = pd.DataFrame(rows, columns=['dataset', 'system', 'condition', 'image_id', 'prediction', 'ce_p', 'ce_r', 'ce_f',
                                'n_ref_not_scorable', 'gate_refs', 'bleu4', 'pred_stratum', 'ref_changed'])
D.to_csv(R / 'per_image_scores.csv.gz', index=False)

# ------------------------------------------------------------------ per condition
crow = []
for (ds, sy, c), g in D.groupby(['dataset', 'system', 'condition']):
    sc = g.dropna(subset=['ce_p', 'ce_r', 'ce_f'])
    cid, _ = cider_d_corpus(list(g.image_id), list(g.prediction), {i: man[ds][i]['references'] for i in g.image_id})
    ch, nc = g[g.ref_changed], g[~g.ref_changed]
    sch, snc = sc[sc.ref_changed], sc[~sc.ref_changed]
    crow.append(dict(dataset=ds, system=sy, condition=c, t1=c[3] if c != 'clean' else '', t2=c[5] if c != 'clean' else '',
                     n=len(sc), ce_p=sc.ce_p.mean(), ce_r=sc.ce_r.mean(), ce_f=sc.ce_f.mean(), bleu4=g.bleu4.mean(), cider_d=cid,
                     n_changed=len(ch), n_unchanged=len(nc), ce_f_changed=sch.ce_f.mean(), ce_f_unchanged=snc.ce_f.mean(),
                     missed_change=(ch.pred_stratum == 'no_change').mean() if len(ch) else np.nan,
                     hallucinated_change=nc.pred_stratum.isin(['full', 'partial']).mean() if len(nc) else np.nan,
                     gated=(g.gate_refs > 0).mean()))
C = pd.DataFrame(crow)
C.to_csv(R / 'per_condition.csv', index=False)

# clean == Tab. 5
tab5 = pd.read_csv(ROOT / 'results_v3' / 'systems' / 'summary.csv').set_index(['dataset', 'system'])
mx, dev = {}, {}
for r in C[C.condition == 'clean'].itertuples():
    t = tab5.loc[(r.dataset, r.system)]
    d = {m: float(getattr(r, m) - t[m]) for m in ('ce_p', 'ce_r', 'ce_f', 'bleu4', 'cider_d')}
    key = f'{r.dataset}|{r.system}'
    dev[key] = dict(**ident[key], **{f'clean_minus_tab5_{m}': v for m, v in d.items()})
    if ident[key]['identical'] == ident[key]['n']:
        mx[key] = max(abs(v) for v in d.values())
checks['clean_vs_tab5_identical_cells_max_abs_diff'] = mx
checks['clean_vs_tab5_all_cells'] = dev
assert all(v < 1e-9 for v in mx.values()), mx
print('clean == Tab. 5 exactly for', len(mx), 'caption-identical cells; other cells:',
      {k: (f"{v['identical']}/{v['n']}", round(v['clean_minus_tab5_ce_f'], 4)) for k, v in dev.items() if k not in mx}, flush=True)

# ------------------------------------------------------------------ per system
rng = np.random.default_rng(0)
METRICS = ['ce_p', 'ce_r', 'ce_f', 'bleu4', 'cider_d', 'missed_change', 'hallucinated_change', 'gated', 'ce_f_changed', 'ce_f_unchanged']
srow, wide = [], {}
for (ds, sy), g in C.groupby(['dataset', 'system']):
    g = g[g.condition.isin(CONDS)]
    if len(g) != len(CONDS):
        continue
    cl = g[g.condition == 'clean'].iloc[0]; co = g[g.condition != 'clean']
    row = dict(dataset=ds, system=sy, n=int(cl.n))
    for m in METRICS:
        row[f'clean_{m}'] = cl[m]; row[f'corr_{m}'] = co[m].mean()
        row[f'corr_{m}_min'] = co[m].min(); row[f'corr_{m}_max'] = co[m].max()
    for m in ('ce_p', 'ce_r', 'ce_f', 'bleu4', 'cider_d'):
        row[f'retained_{m}'] = row[f'corr_{m}'] / row[f'clean_{m}']
    row['worst_scenario'] = co.loc[co.ce_f.idxmin(), 'condition'][3:]; row['best_scenario'] = co.loc[co.ce_f.idxmax(), 'condition'][3:]
    same = co[co.t1 == co.t2]; mixed = co[co.t1 != co.t2]
    row['corr_ce_f_same_type'] = same.ce_f.mean(); row['corr_ce_f_mixed_type'] = mixed.ce_f.mean()
    for k in TYPES:
        row[f'ce_f_t1_{k}'] = co[co.t1 == k].ce_f.mean(); row[f'ce_f_t2_{k}'] = co[co.t2 == k].ce_f.mean()
    # paired bootstrap over images: per-image mean over the 25 scenarios minus clean
    pi = D[(D.dataset == ds) & (D.system == sy) & D.condition.isin(CONDS)].pivot(index='image_id', columns='condition', values=['ce_p', 'ce_r', 'ce_f'])
    pi = pi.dropna()
    row['n_paired'] = len(pi)
    for m in ('ce_p', 'ce_r', 'ce_f'):
        d = (pi[m][[f's3_{c}' for c in SCEN]].mean(axis=1) - pi[m]['clean']).values
        bm = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
        row[f'delta_{m}'] = float(d.mean()); row[f'delta_{m}_lo'] = float(np.percentile(bm, 2.5)); row[f'delta_{m}_hi'] = float(np.percentile(bm, 97.5))
    # T1-vs-T2 asymmetry: scenario (x on T1, clean-equivalent?) -> compare A-B with B-A over the 10 unordered mixed pairs
    asym = [(co[co.condition == f's3_{x}-{y}'].ce_f.iloc[0] - co[co.condition == f's3_{y}-{x}'].ce_f.iloc[0]) for x, y in itertools.combinations(TYPES, 2)]
    row['asym_mean_abs'] = float(np.mean(np.abs(asym))); row['asym_max_abs'] = float(np.max(np.abs(asym)))
    srow.append(row)
    wide[(ds, sy)] = pi['ce_f'][[f's3_{c}' for c in SCEN]].mean(axis=1)
Ssum = pd.DataFrame(srow)
Ssum.to_csv(R / 'per_system.csv', index=False)
if Ssum.empty:
    json.dump(dict(checks=checks), open(R / 'robustness.json', 'w'), indent=1, default=float)
    sys.exit('no complete (system, dataset) cell yet; checks written')

# ------------------------------------------------------------------ comparisons under interference
pw, res = [], {}
for ds in DS:
    sys_ds = sorted(sy for d, sy in wide if d == ds)
    pairs = []
    for x, y in itertools.combinations(sys_ds, 2):
        j = wide[(ds, x)].index.intersection(wide[(ds, y)].index)
        d = (wide[(ds, x)].loc[j] - wide[(ds, y)].loc[j]).values
        bm = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
        p = max(2 * min((bm <= 0).mean(), (bm >= 0).mean()), 1.0 / (B + 1))
        pairs.append(dict(dataset=ds, a=x, b=y, n=len(d), mean_diff=float(d.mean()), ci_lo=float(np.percentile(bm, 2.5)), ci_hi=float(np.percentile(bm, 97.5)), p_raw=p))
    order = np.argsort([q['p_raw'] for q in pairs]); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(pairs) - rank) * pairs[i]['p_raw'])); pairs[i]['p_holm'] = run
    pw += pairs
    s = Ssum[Ssum.dataset == ds].set_index('system')
    if len(s) >= 3:
        res[f'{ds}_rank_clean_vs_corrupted_spearman'] = float(spearmanr(s.clean_ce_f, s.corr_ce_f)[0])
        for m in ('bleu4', 'cider_d'):
            res[f'{ds}_corrupted_rank_cef_vs_{m}_spearman'] = float(spearmanr(s.corr_ce_f, s[f'corr_{m}'])[0])
            res[f'{ds}_retention_rank_cef_vs_{m}_spearman'] = float(spearmanr(s.retained_ce_f, s[f'retained_{m}'])[0])
    res[f'{ds}_order_clean'] = list(s.sort_values('clean_ce_f', ascending=False).index)
    res[f'{ds}_order_corrupted'] = list(s.sort_values('corr_ce_f', ascending=False).index)
pd.DataFrame(pw).to_csv(R / 'pairwise_corrupted.csv', index=False)
# severity sweep (LEVIR-CC, same type on both images): severity 1 / 3 / 5
sw = C[(C.dataset == 'LEVIR_CC') & (C.condition.str[3] == C.condition.str[5]) & (C.condition != 'clean')].copy()
sw['severity'] = sw.condition.str[1].astype(int)
SW = sw.groupby(['system', 'severity'])[['ce_p', 'ce_r', 'ce_f', 'bleu4', 'cider_d', 'missed_change', 'hallucinated_change', 'gated']].mean().reset_index()
SW['n_types'] = sw.groupby(['system', 'severity']).size().values
SW.to_csv(R / 'severity_sweep.csv', index=False)
res['severity_sweep_complete'] = bool((SW.n_types == 5).all()) and set(SW.severity) == {1, 3, 5}
res['checks'] = checks
res['n_unique_caption_image_pairs'] = len(keys)
res['n_parsed_new'] = n_new
json.dump(res, open(R / 'robustness.json', 'w'), indent=1, default=float)
pd.set_option('display.width', 250)
print(Ssum[['dataset', 'system', 'n', 'clean_ce_p', 'corr_ce_p', 'clean_ce_r', 'corr_ce_r', 'clean_ce_f', 'corr_ce_f', 'delta_ce_f', 'delta_ce_f_lo', 'delta_ce_f_hi',
            'clean_missed_change', 'corr_missed_change', 'clean_hallucinated_change', 'corr_hallucinated_change', 'clean_gated', 'corr_gated']].round(3).to_string(), flush=True)
print(json.dumps({k: v for k, v in res.items() if k != 'checks'}, indent=1), flush=True)
print('DONE')
